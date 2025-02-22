# coding: UTF-8
# This is creaed 2025/02/19 by Y. Shinohara
# This is lastly modified 20xx/yy/zz by Y. Shinohara
import numpy as np
from scipy.sparse import dok_array

def grad1D_uniform_grid(N1, dx):
    grad1D = dok_array((N1, N1), dtype=np.float64)
#    for i in range(1, N1 - 1):
#        grad1D[i, i + 1] = 0.5/dx
#        grad1D[i, i - 1] = -0.5/dx
#    grad1D[0, 0 + 1] = 0.5/dx
#    grad1D[N1 - 1, N1 - 2] = -0.5/dx
#    for i in range(2, N1 - 2):
#        grad1D[i, i - 2] =  1.0/12.0/dx
#        grad1D[i, i - 1] = -2.0/3.0/dx
#        grad1D[i, i + 1] =  2.0/3.0/dx
#        grad1D[i, i + 2] = -1.0/12.0/dx
#    grad1D[0, 0 + 1] =  2.0/3.0/dx
#    grad1D[0, 0 + 2] = -1.0/12.0/dx
#    grad1D[1, 1 - 1] = -2.0/3.0/dx
#    grad1D[1, 1 + 1] =  2.0/3.0/dx
#    grad1D[1, 1 + 2] = -1.0/12.0/dx
#    grad1D[N1-2, N1-2 - 2] =  1.0/12.0/dx
#    grad1D[N1-2, N1-2 - 1] = -2.0/3.0/dx
#    grad1D[N1-2, N1-2 + 1] =  2.0/3.0/dx
#    grad1D[N1-1, N1-1 - 2] =  1.0/12.0/dx
#    grad1D[N1-1, N1-1 - 1] = -2.0/3.0/dx
    for i in range(N1):
        if (i - 4 > 0):
            grad1D[i, i - 4] =  1.0/280.0/dx
        if (i - 3 > 0):
            grad1D[i, i - 3] =  -4.0/105.0/dx
        if (i - 2 > 0):
            grad1D[i, i - 2] =  1.0/5.0/dx
        if (i - 1 > 0):
            grad1D[i, i - 1] = -4.0/5.0/dx
        if (i + 1 < N1):
            grad1D[i, i + 1] =  4.0/5.0/dx
        if (i + 2 < N1):
            grad1D[i, i + 2] = -1.0/5.0/dx
        if (i + 3 < N1):
            grad1D[i, i + 3] =  4.0/105.0/dx
        if (i + 4 < N1):
            grad1D[i, i + 4] = -1.0/280.0/dx
    grad1D = grad1D.tocsr()  # 計算時には CSR
    
    return grad1D
