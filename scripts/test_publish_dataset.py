"""Protect exact rows and provenance when distributing compressed CSV shards."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import zstandard

from publish_dataset import package


class PublicationTests(unittest.TestCase):
    def test_round_trip_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            # Larger than the read buffer; quoted commas and CRLF stay unchanged.
            original = b'name,enriched_text\r\n' + b'a,"[C]<1,2,3>"\r\n' * 600000
            path = source / 'merged_train.csv'
            path.write_bytes(original)
            (source / 'manifest.json').write_text(json.dumps({'files': [{
                'path': str(path), 'rows': 600000, 'bytes': len(original),
                'sha256': hashlib.sha256(original).hexdigest()}]}))
            release = package(source, output, target_bytes=1024**2)
            shards = release['splits'][0]['shards']
            self.assertGreater(len(shards), 1)
            recovered = []
            for index, shard in enumerate(shards):
                with (output / shard['path']).open('rb') as stream:
                    with zstandard.ZstdDecompressor().stream_reader(stream) as reader:
                        data = reader.read()
                recovered.append(data if index == 0 else data.split(b'\n', 1)[1])
            self.assertEqual(b''.join(recovered), original)
            self.assertEqual(package(source, output), release)
            (output / shards[0]['path']).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                package(source, output)

    def test_wrong_source_hash_refuses_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            original = b'name,enriched_text\na,[C]<0,0,0>\n'
            path = source / 'merged_train.csv'
            path.write_bytes(original)
            (source / 'manifest.json').write_text(json.dumps({'files': [{
                'path': str(path), 'rows': 1, 'bytes': len(original), 'sha256': '0' * 64}]}))
            with self.assertRaisesRegex(ValueError, 'hash/count mismatch'):
                package(source, output)
            self.assertFalse((output / 'release.json').exists())


if __name__ == '__main__':
    unittest.main()
