import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
SCRIPT=Path(__file__).with_name('audit_inchikey_overlap.py')
spec=importlib.util.spec_from_file_location('inchi_audit',SCRIPT)
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

class AuditTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(a.identity('[C][C][O]')[0],a.identity('[O][C][C]')[0])
        self.assertEqual(a.identity('O=c1cccc[nH]1')[0],a.identity('Oc1ccccn1')[0])
        for x,y in [('C[C@H](O)F','C[C@@H](O)F'),('CC(O)F','C[C@H](O)F'),('[13CH3]CO','CCO')]:
            self.assertNotEqual(a.identity(x)[0],a.identity(y)[0])
        self.assertTrue(a.identity('C1CC')[2])

    def test_counts_failures_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def write(name,rows):
                p=root/name
                with p.open('w',newline='') as f:
                    w=csv.writer(f); w.writerow(['name','enriched_text']); w.writerows(rows)
                return p
            ref=write('ref.csv',[('ethanol','[C][C][O]'),('tautomer','O=c1cccc[nH]1'),('stereo','C[C@H](O)F'),('bad','C1CC')])
            ref2=write('ref2.csv',[('also_ethanol','[C][C][O]')])
            train=write('train.csv',[(f'source_{i}',s) for i,s in enumerate(['[C]<0,0,0>[C][O]','[O][C][C]','Oc1ccccn1','C[C@H](O)F','C[C@@H](O)F','CC(O)F','[13CH3]CO','C1CC','C1CO'])])
            out=root/'out'
            cmd=[sys.executable,str(SCRIPT),'--train',str(train),'--reference',str(ref),'--reference',str(ref2),'--output',str(out),'--workers','2','--batch-size','2']
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            report=json.loads((out/'summary.json').read_text()); group=report['groups']['all_references']
            self.assertEqual(report['rows_scanned'],9)
            self.assertEqual(group['exact_text']['training_rows'],3)
            self.assertEqual(group['inchikey']['training_rows'],4)
            self.assertEqual(group['additional']['training_rows'],2)
            self.assertEqual(group['exact_without_valid_key']['training_rows'],1)
            self.assertEqual(report['counts']['conversion_failure_rows'],2)
            before=(out/'matches.csv').read_bytes()
            with sqlite3.connect(out/'audit.sqlite') as db:
                state=json.loads(db.execute("SELECT value FROM metadata WHERE key='state'").fetchone()[0])
                state.update(rows=4,counts={'rows':4,'exact_text_rows':2,'converted_rows':4,'inchikey_match_rows':4,'additional_rows':2},warnings={})
                with train.open('rb') as f:
                    for _ in range(5): f.readline()
                    state['offset']=f.tell()
                db.execute('DELETE FROM hits WHERE first_row>4'); db.execute('DELETE FROM failures')
                db.execute("UPDATE metadata SET value=? WHERE key='state'",(json.dumps(state),))
            subprocess.run(cmd+['--resume'],check=True,capture_output=True,text=True)
            self.assertEqual(before,(out/'matches.csv').read_bytes())
            self.assertEqual(json.loads((out/'summary.json').read_text())['groups'],report['groups'])

if __name__=='__main__': unittest.main()
