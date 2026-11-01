"""Exact frozen mixed-corpus validation inputs, in the same order and vocabulary."""
import json
import numpy as np
from nabla_data import OriginalNabla

class SharedValidation(OriginalNabla):
 def __init__(self,root,vocab):
  self.root=root;self.manifest=json.loads((root/'manifest.json').read_text())
  assert self.manifest['status']=='complete' and self.manifest['vocab']==vocab
  self.rows=self.manifest['rows'];self.vocab=vocab;self.dsmap={'nablaDFT':0};self.datasets=['nablaDFT']
  for name in ['tokens','coords','atoms','keys']:setattr(self,name,np.load(root/(name+'.npy'),mmap_mode='r'))
