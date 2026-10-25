import numpy as np
from src.medifederate.ct import measure_mask
def test_measure():
    x=np.zeros((10,20),dtype=np.uint8); x[2:7,5:15]=1
    assert measure_mask(x)["area_mm2"]==50.0
