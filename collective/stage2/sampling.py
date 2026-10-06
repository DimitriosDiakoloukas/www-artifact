"""Prospectively specified, size-matched sign stress tests; no conditional-law claim."""
import hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
RADII=(0,1,2,3)
FRACTIONS=(.02,.05,.1)
MECHANISMS=('fixed','exchange','cycles','cycle_matched_fixed','cycle_matched_exchange')
def source_hash():
    digest=hashlib.sha256()
    for name in ('sampling.py','matched.py'):
        digest.update(name.encode());digest.update((BASE/name).read_bytes())
    return digest.hexdigest()
def rng_for(checkpoint,query,radius,budget,mechanism,draw):
    return np.random.default_rng(np.random.SeedSequence([20261007,checkpoint,query,radius,budget,mechanism,draw]))
def fixed(signs,eligible,m,rng,exchange=False):
    ids=np.flatnonzero(eligible)
    assert 0<=m<=len(ids) and m%2==0
    if exchange:
        pos=ids[signs[ids]>0];neg=ids[signs[ids]<0]
        if min(len(pos),len(neg))<m//2:return None
        selected=np.concatenate([rng.choice(pos,m//2,replace=False),rng.choice(neg,m//2,replace=False)])
    else:selected=rng.choice(ids,m,replace=False)
    result=signs.copy();result[selected]*=-1
    assert np.array_equal(result[~eligible],signs[~eligible])
    if exchange:assert np.count_nonzero(result<0)==np.count_nonzero(signs<0)
    return result
class CycleSampler:
    def __init__(self,signs,edges,eligible):
        self.signs,self.edges,self.eligible=signs,edges,eligible
        ids=np.flatnonzero(eligible)
        self.positive=ids[signs[ids]>0]
        self.pos={};self.neg={}
        for i in ids:
            a,b=map(int,edges[i]);adj=self.pos if signs[i]>0 else self.neg
            adj.setdefault(a,{})[b]=int(i);adj.setdefault(b,{})[a]=int(i)
        self.neg_nodes={a:np.array(sorted(adj),dtype=np.int64) for a,adj in self.neg.items()}
    def draw(self,m,rng,max_proposals=20000):
        used=set();accepted=proposals=0;target=m//4
        while accepted<target and proposals<max_proposals and len(self.positive):
            proposals+=1
            i=int(self.positive[rng.integers(len(self.positive))])
            if i in used:continue
            a,b=map(int,self.edges[i])
            if rng.integers(2):a,b=b,a
            neighbors=self.neg_nodes.get(a)
            if neighbors is None:continue
            c=int(neighbors[rng.integers(len(neighbors))]);j=self.neg[a][c]
            if j in used or c==b:continue
            intersection=sorted(self.pos.get(c,{}).keys() & self.neg.get(b,{}).keys())
            if not intersection:continue
            d=intersection[rng.integers(len(intersection))]
            if len({a,b,c,d})<4:continue
            u,v=self.pos[c][d],self.neg[b][d]
            if u in used or v in used:continue
            used.update((i,j,u,v));accepted+=1
        result=self.signs.copy()
        if used:result[np.array(sorted(used))]*=-1
        assert np.array_equal(result[~self.eligible],self.signs[~self.eligible])
        # Equality of signed degree at every endpoint follows from two opposite signs/cycle.
        delta=np.bincount(self.edges.ravel(),weights=np.repeat(result-self.signs,2),minlength=int(self.edges.max())+1)
        assert not np.any(delta)
        return result,len(used),accepted,proposals,target
