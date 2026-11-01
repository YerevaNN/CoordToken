"""Compatibility entrypoint for already-queued jobs 1051/1052.

The original launch.py is archived alongside this file before installation.
The queued Slurm script calls launch.py WIDTH-env gate 3; retain its job ID.
"""
import os
from pathlib import Path
import sys

assert sys.argv[1:] == ['gate', '3'], 'Only the original startup invocation may dispatch'
assert os.environ.get('NABLA_WIDTH') in {'512','1024'}
new = Path(__file__).resolve().parent.parent/'nabla_b200_20261023/run_all.py'
assert new.is_file()
os.execv(sys.executable,[sys.executable,'-u',str(new)])
