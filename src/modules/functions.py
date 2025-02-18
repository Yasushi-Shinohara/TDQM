# coding: UTF-8
# This is creaed 2025/02/19 by Y. Shinohara
# This is lastly modified 20xx/yy/zz by Y. Shinohara
import numpy as np
from scipy.sparse import dok_array

def grad1D_uniform_grid(N1, dx):
    grad1D = dok_array((N1, N1), dtype=np.float64)
    for i in range(1, N1 - 1):
        grad1D[i, i + 1] = 0.5/dx
        grad1D[i, i - 1] = -0.5/dx
    grad1D[0, 0 + 1] = 0.5/dx
    grad1D[N1 - 2, N1 - 1] = -0.5/dx
    grad1D = grad1D.tocsr()  # 計算時には CSR
    
    return grad1D
