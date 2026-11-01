"""Original Transformer/iFSQ architecture with optional graph-independent work tokens."""
from dataclasses import dataclass, asdict
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from prepare import ROOT
sys.path.insert(0,str(ROOT))
from quantizer import FSQ

@dataclass
class Config:
    vocab_size:int=192
    width:int=1024
    layers:int=8
    workspace:int=0
    bits:int=12
    dropout:float=0.1

class Tokenizer(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;c=config
        self.tok_emb=nn.Linear(c.vocab_size+3,c.width)
        def block():
            layer=nn.TransformerEncoderLayer(c.width,c.width//64,c.width*4,dropout=c.dropout,batch_first=True,norm_first=True,bias=False)
            return nn.TransformerEncoder(layer,c.layers,enable_nested_tensor=False)
        self.enc,self.dec=block(),block()
        self.pre_q=nn.Sequential(nn.Linear(c.width,c.width//8),nn.ReLU(),nn.Linear(c.width//8,c.bits),nn.Tanh())
        self.quant=FSQ([2]*c.bits,return_indices=True,preserve_symmetry=True,keep_num_codebooks_dim=False)
        self.post_q=nn.Sequential(nn.Linear(c.vocab_size+c.bits,256),nn.ReLU(),nn.Linear(256,c.width))
        self.out=nn.Linear(c.width,3)
        if c.workspace:
            self.encoder_workspace=nn.Parameter(torch.randn(c.workspace,c.width)*0.02)
            self.decoder_workspace=nn.Parameter(torch.randn(c.workspace,c.width)*0.02)
        self.register_buffer('pos_emb_cache',torch.empty(1,0,c.width),persistent=False)
    def positions(self,n,device,dtype):
        if self.pos_emb_cache.size(1)<n or self.pos_emb_cache.dtype!=dtype or self.pos_emb_cache.device!=device:
            pos=torch.arange(n,device=device,dtype=torch.float32)[:,None]
            freqs=torch.exp(torch.arange(0,self.config.width,2,device=device,dtype=torch.float32)*(-__import__('math').log(10000.)/self.config.width))
            pe=torch.empty(1,n,self.config.width,device=device)
            pe[0,:,0::2]=torch.sin(pos*freqs);pe[0,:,1::2]=torch.cos(pos*freqs);self.pos_emb_cache=pe.to(dtype)
        return self.pos_emb_cache[:,:n]
    def stack(self,module,h,pad,workspace=None):
        n=h.size(1)
        if workspace is not None:
            h=torch.cat([h,workspace.to(h.dtype)[None].expand(h.size(0),-1,-1)],1)
            pad=F.pad(pad,(0,workspace.size(0)),value=False)
        return module(h,src_key_padding_mask=pad)[:,:n]
    def identities(self,tokens):
        return F.one_hot(tokens.clamp_min(0),self.config.vocab_size).float()*tokens.ge(0).unsqueeze(-1)
    def decode(self,tokens,indices):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        backend=[SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION] if tokens.is_cuda else SDPBackend.MATH
        with sdpa_kernel(backend):
            quant=self.quant.indices_to_codes(indices)
            return self.decode_quant(tokens,quant)
    def decode_quant(self,tokens,quant):
        identity=self.identities(tokens);d=self.post_q(torch.cat([identity.to(quant.dtype),quant],-1));pe=self.positions(tokens.size(1),d.device,d.dtype)
        h=self.stack(self.dec,d+pe,tokens.lt(0),getattr(self,'decoder_workspace',None))
        with torch.autocast(device_type=h.device.type,enabled=False):return self.out(h.float())
    def forward(self,tokens,coords,atoms=None):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        backend=[SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION] if tokens.is_cuda else SDPBackend.MATH
        with sdpa_kernel(backend):
            identity=self.identities(tokens);h=self.tok_emb(torch.cat([identity,coords],-1));pe=self.positions(tokens.size(1),h.device,h.dtype)
            z=self.stack(self.enc,h+pe,tokens.lt(0),getattr(self,'encoder_workspace',None));quant,indices=self.quant(self.pre_q(z))
            return self.decode_quant(tokens,quant),indices
    def architecture(self):return {**asdict(self.config),'parameters':sum(p.numel() for p in self.parameters())}
