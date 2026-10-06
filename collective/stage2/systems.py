"""Hash-checked frozen evaluation systems shared by prospective checkpoint analyses."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2 import native_random as nr
from replication.measure import network_data,load_model
from srange.heads import PairHead
from srange import provenance as pv

class Frozen:
    def __init__(self,record,device,native=True):
        self.rec=pv.verify_record(record);self.device=device;self.native=native
        r=self.rec
        if native:
            assert r['features']=='random' and r['implementation_sha256']==nr.IMPLEMENTATION_SHA
            self.ds=nr.directed_data(r['network'])
            split,s=nr.load_or_make_split(nr.STORE,self.ds.name,self.ds.meta['processed_sha256'],len(self.ds.edges),r['seed'])
            assert s==r['split_sha256']
            self.tr,self.va,self.te=(split[x] for x in ('train','val','test'))
            self.X=nr.features(self.ds,self.tr,r['arch'],r['seed']).to(device)
            assert nr.tensor_sha256(self.X)==r['features_sha256']
            self.g=nr.graph(self.ds.n,self.ds.edges[self.tr],self.ds.signs[self.tr],device)
            self.enc=nr.NativeSystem(r['arch'],self.X,self.g,self.ds.edges[self.tr],self.ds.signs[self.tr],r['T'],r['seed'],c=.35 if r['network']=='bitcoin_alpha' else .45)
            ck=pv.load_checkpoint(r['checkpoint']['sha256']);self.enc.load_state_dict(ck['state']);self.enc.eval()
            self.seed=r['seed'];self.arch=r['arch'];self.network=r['network']
            self.recorded_logits=r['test_logits_sha256'];self.previous_queries=np.array(r['query_positions'])
        else:
            self.ds,self.tr,self.te,self.X,self.g=network_data(r,device)
            split,s=nr.load_or_make_split(nr.STORE,r['args']['dataset'],self.ds.meta['processed_sha256'],len(self.ds.edges),r['args']['seed'])
            self.va=split['val'];ck=pv.load_checkpoint(r['checkpoint']['sha256'])
            self.enc,self.head=load_model(ck,device,bool(r['model'].get('memory_efficient',False)),PairHead)
            self.seed=r['args']['seed'];self.arch=r['args']['arch'];self.network=r['args']['dataset']
            self.recorded_logits=r['evaluation']['test_logits_sha256'];self.previous_queries=np.array(r['targets']['test_edge_positions'])
        self.edges=self.ds.edges[self.tr];self.signs=self.ds.signs[self.tr].astype(np.float32)
    def forward(self,pairs,signs=None):
        q=torch.as_tensor(pairs,device=self.device)
        graph=self.g if signs is None else self.g.with_signs(torch.as_tensor(signs,device=self.device))
        if self.native:return self.enc.scores(q,None if signs is None else graph)
        return self.head(self.enc(self.X,graph),q)
    def verify(self):
        with torch.no_grad():s=self.forward(self.ds.edges[self.te]).cpu().numpy()
        assert pv.array_sha256(s)==self.recorded_logits,'frozen full-test function differs'
        return s

def native_T32():
    return [nr.DIRECTORY/'eval'/(nr.job_id(*job)+'.json') for job in nr.jobs('eval') if job[2]==32]
