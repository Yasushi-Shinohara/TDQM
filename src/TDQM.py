#!/usr/bin/python
# coding: UTF-8
import os
import time
ts = time.time()
from modules.print_funcs import print_header, print_footer, print_midtime, print_endtime
print_header()
from modules.constants import *
from modules.IO import *
from modules.DefineSystem import Unit, Model, TGrid, Field, TEvolution, Options
#from modules.DefineSystem import *
from modules.plots import plotfielddata, plotphidata
#from modules.functions import get_an_expectaion_value

#############################Prep. for the system########################
param = Input.read_input()
unit = Unit.construct_unit_namedtuple(param)
model = Model.construct_model_namedtuple(param)
tgrid = TGrid.construct_tgrid_namedtuple(param)
field = Field.construct_field_namedtuple(param, tgrid)
tevolution = TEvolution.construct_tevolution_namedtuple(param)
options = Options.construct_options_namedtuple(param)

# Getting the initial condition and relevant physical quantities

#############################Prep. for RT################################

if (options.plot_option):
    plotfielddata(tgrid, field)

if (np.amax(tgrid.t) < max([field.Tpulse1, field.Tpulse2, field.Tpulse3])):
    print('# WARNING: max(t) is shorter than Tpulse')

phi = model.set_GS_WF(model.x0, model.delx, model.p0, model)
if (options.plot_option):
    plotphidata(model.x, phi)
velocity = model.velocity(model)
t = 0.5*velocity@velocity
vpot = model.vpot(model)
h = t + vpot
print(h.dtype)
eig, vec = model.get_h_eigenpairs(h)
print(eig)
import matplotlib.pyplot as plt
plt.figure()
plt.ylim(0.0, 4.0)
plt.plot(model.x, vpot.diagonal())
for n in range(6):
    plt.plot(model.x, np.real(vec[:,n])*2.0+eig[n], label=str(n))
#plt.plot(model.x, np.real(phi), label='$\Re (\phi)$')
#plt.plot(model.x, np.imag(phi), label='$\Im (\phi)$')
#plt.fill_between(x, np.abs(phi), -np.abs(phi),facecolor='k',alpha=0.25,label='envelope')
plt.legend()
plt.show()

tt = time.time()
print_midtime(ts,tt)

#############################RT calculation##############################
#
psi = 1.0*phi
norm = np.linalg.norm(psi)

for it in range(tgrid.Nt):
    velocity = model.velocity(model)
    t = 0.5*velocity@velocity
    vpot = model.vpot(model)
    h = t + vpot
    psi = tevolution.get_psi_forward(psi, h, tgrid.dt)
    norm = np.linalg.norm(psi)
    if (it%200 == 0):
        print('# ',it, norm)
#
te = time.time()
print_endtime(ts,tt,te,tgrid.Nt)

print_footer() 
sys.exit()
