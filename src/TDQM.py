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
from modules.plots import plotfielddata, plotphidata, plotspectra
#from modules.functions import get_an_expectaion_value
from modules.functions import lap1D_uniform_grid, grad1D_uniform_grid

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

if (options.plot_option):
    #t = 0.5*velocity@velocity
    tkin = model.tkin(model)
    vpot = model.vpot(model)
    h = tkin + vpot
    print(h.dtype)
    eig, vec = model.get_h_eigenpairs(h)    
    plotspectra(eig, vec, model.x, vpot.diagonal())

tt = time.time()
print_midtime(ts,tt)

#############################RT calculation##############################
#
psi = 1.0*phi
norm = np.linalg.norm(psi)

for it in range(tgrid.Nt):
    velocity = model.velocity(model)
    #t = 0.5*velocity@velocity
    tkin = model.tkin(model)
    vpot = model.vpot(model)
    h = tkin + vpot
    psi = tevolution.get_psi_forward(psi, h, tgrid.dt)
    norm = np.linalg.norm(psi)
    if (it%200 == 0):
        print('# ',it, norm)
#
te = time.time()
print_endtime(ts,tt,te,tgrid.Nt)

print_footer() 
sys.exit()
