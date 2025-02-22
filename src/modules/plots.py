# coding: UTF-8
# This is creaed 2025/02/17 by Y. Shinohara
# This is lastly modified 20xx/yy/zz by Y. Shinohara
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from modules.constants import *

#
def plotfielddata(tgrid, field):
    """Plotting the field data"""
    plt.figure()
    plt.title('Et')
    plt.xlabel('Time [fs]')
    plt.ylabel('E-field [V/nm]')
    plt.plot(tgrid.t*Atomtime2fs, field.E[0]*Atomfield2Vpnm, '-', label="Ex")
    plt.plot(tgrid.t*Atomtime2fs, field.E[1]*Atomfield2Vpnm, '--',label="Ey")
    plt.plot(tgrid.t*Atomtime2fs, field.E[2]*Atomfield2Vpnm, '-.',label="Ez")
    plt.grid()
    plt.legend()
    plt.show()
    #
    plt.figure()
    plt.title('At')
    plt.xlabel('Time [fs]')
    plt.ylabel('A-field [a.u.]')
    plt.plot(tgrid.t*Atomtime2fs, field.A[0], '-', label="Ax")
    plt.plot(tgrid.t*Atomtime2fs, field.A[1], '--',label="Ay")
    plt.plot(tgrid.t*Atomtime2fs, field.A[2], '-.',label="Az")
    plt.grid()
    plt.legend()
    plt.show()
    #
    return
#
def plotphidata(x, phi):
    plt.figure()
    plt.plot(x, np.real(phi), label='$\Re (\phi)$')
    plt.plot(x, np.imag(phi), label='$\Im (\phi)$')
    plt.fill_between(x, np.abs(phi), -np.abs(phi),facecolor='k',alpha=0.25,label='envelope')
    plt.legend()
    plt.show()
    #
    return
    
