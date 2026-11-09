"""Process-local shielding changes; elemental network parameters stay intact."""
from contextlib import contextmanager
from functools import wraps

import numpy as np

MODE=0
CURRENT=None
COUNTS=np.zeros(3,dtype='i8')  # factor calls, dynamic calls, elemental fallbacks
PATCHED=False


def reset(mode):
    global MODE,CURRENT,COUNTS
    assert mode in (0,1,2,3)
    MODE=int(mode);CURRENT=None;COUNTS=np.zeros(3,dtype='i8')


def select(elemental):
    COUNTS[0]+=1
    if MODE==1:return 35.
    if MODE==2:return 69.
    if MODE==3:
        if CURRENT is not None:COUNTS[1]+=1;return CURRENT
        COUNTS[2]+=1
    return elemental


@contextmanager
def state_context(xin,co_index,c13_index):
    global CURRENT
    previous=CURRENT
    co,c13=float(xin[co_index]),float(xin[c13_index])
    ratio=None
    if np.isfinite(co) and np.isfinite(c13) and co>0 and c13>0:
        with np.errstate(over='ignore',divide='ignore'):candidate=np.float64(co)/np.float64(c13)
        if np.isfinite(candidate) and candidate>0:ratio=float(candidate)
    CURRENT=ratio
    try:yield
    finally:CURRENT=previous


def install():
    global PATCHED
    assert not PATCHED
    from despotic.chemistry.visser_shielding import VisserCOShielding
    from despotic.chemistry.GOW import GOW,specList
    original=VisserCOShielding.factor
    @wraps(original)
    def factor(self,species,n12co,nh2,b_co=.3,tex=20.,c12_c13=60.):
        return original(self,species,n12co,nh2,b_co=b_co,tex=tex,c12_c13=select(c12_c13))
    VisserCOShielding.factor=factor
    co,c13=specList.index('CO'),specList.index('13CO')
    def wrap(previous):
        @wraps(previous)
        def call(self,xin,*args,**kwargs):
            # Never assign self.c12c13: it also sets elemental reservoirs.
            with state_context(xin,co,c13):return previous(self,xin,*args,**kwargs)
        return call
    GOW.dxdt=wrap(GOW.dxdt);GOW.dxdt_type=wrap(GOW.dxdt_type)
    PATCHED=True
