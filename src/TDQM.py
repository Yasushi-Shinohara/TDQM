#!/usr/bin/python
# coding: UTF-8
import os
import time
ts = time.time()
from modules.print_funcs import print_header, print_footer, print_midtime, print_endtime
print_header()
from modules.constants import *
from modules.IO import *
from modules.DefineSystem import Unit, Model, KGrid, TGrid, Field, TEvolution, Options
#from modules.DefineSystem import *
from modules.plots import plotGSdata, plotfielddata
from modules.functions import get_an_expectaion_value

#############################Prep. for the system########################
param = Input.read_input()
unit = Unit.construct_unit_namedtuple(param)
model = Model.construct_model_namedtuple(param)
kgrid = KGrid.construct_kgrid_namedtuple(param)
tgrid = TGrid.construct_tgrid_namedtuple(param)
field = Field.construct_field_namedtuple(param, tgrid)
tevolution = TEvolution.construct_tevolution_namedtuple(param)
options = Options.construct_options_namedtuple(param)

# Getting the initial condition and relevant physical quantities
phik = np.zeros([2, 2, kgrid.Nk], dtype="complex128")
ek = np.zeros([2, kgrid.Nk], dtype="float64")
if (model.spatialdim == 2) :
    for nk in range(kgrid.Nk):
        h = model.k2h(kgrid.kx[nk], kgrid.ky[nk], None, model)
        ek[:,nk],phik[:,:,nk] = np.linalg.eigh(h) #Getting the eigenvpairs
    if (options.plot_option):
        spink, Jk, Jspink = plotGSdata(kgrid, ek, phik, model)

print("# np.amin(ek), np.amax(ek), np.amax(ek)-np.amin(ek) =",np.amin(ek), np.amax(ek), np.amax(ek)-np.amin(ek))
print("# np.average(ek[0,:]), np.average(ek[1,:]) =",np.average(ek[0,:]), np.average(ek[1,:]))
#############################Prep. for RT################################

if (options.plot_option):
    plotfielddata(tgrid, field)

nv = np.zeros([tgrid.Nt],dtype=np.float64)
nc = np.zeros([tgrid.Nt],dtype=np.float64)
Ene = np.zeros([tgrid.Nt],dtype=np.float64)
Mag = np.zeros([3,tgrid.Nt],dtype=np.float64)
J = np.zeros([2,tgrid.Nt],dtype=np.float64)
Jspin = np.zeros([3,2,tgrid.Nt],dtype=np.float64)

if (np.amax(tgrid.t) < max([field.Tpulse1, field.Tpulse2, field.Tpulse3])):
    print('# WARNING: max(t) is shorter than Tpulse')

tt = time.time()
print_midtime(ts,tt)

#############################RT calculation##############################
#
psik = 1.0*phik[:,0,:]
print('# it, Ene[it], J[:,it], norm')
#
for it in range(tgrid.Nt):
    for nk in range(kgrid.Nk):
        kx = kgrid.kx[nk] + field.A[0,it]
        ky = kgrid.ky[nk] + field.A[1,it]
        h = model.k2h(kx, ky, None, model)
        psik[:, nk] = tevolution.get_psi_forward(psik[:, nk], h, tgrid.dt)
        _,phik[:,:,nk] = np.linalg.eigh(h)
        vx, vy, vz = model.k2v(kx, ky, None, model)
        w1x, w2x, w3x, w1y, w2y, w3y, w1z, w2z, w3z = model.k2w(kx, ky, None, model)
        Ene[it] += get_an_expectaion_value(psik[:, nk], h)
        Mag[0, it] = get_an_expectaion_value(psik[:, nk], sigma1)
        Mag[1, it] = get_an_expectaion_value(psik[:, nk], sigma2)
        Mag[2, it] = get_an_expectaion_value(psik[:, nk], sigma3)
        J[0, it] += get_an_expectaion_value(psik[:, nk], vx)
        J[1, it] += get_an_expectaion_value(psik[:, nk], vy)
        Jspin[0, 0, it] += get_an_expectaion_value(psik[:, nk], w1x)
        Jspin[1, 0, it] += get_an_expectaion_value(psik[:, nk], w2x)
        Jspin[2, 0, it] += get_an_expectaion_value(psik[:, nk], w3x)
        Jspin[0, 1, it] += get_an_expectaion_value(psik[:, nk], w1y)
        Jspin[1, 1, it] += get_an_expectaion_value(psik[:, nk], w2y)
        Jspin[2, 1, it] += get_an_expectaion_value(psik[:, nk], w3y)
    
    #psik = psikhk2psik(param,psik,hk)
    #_, phik, _ = hk2ekvkspink4GS(param, hk)

    Ene[it] = Ene[it]/kgrid.Nk
    Mag[:,it] = Mag[:,it]/kgrid.Nk
    J[:,it] = J[:,it]/kgrid.Nk
    Jspin[:,:,it] = Jspin[:,:,it]/kgrid.Nk
    #nv[it] = np.sum( (np.abs(np.conj(psik[:,0])*phik[:,0,0] + np.conj(psik[:,1])*phik[:,1,0]))**2 )/param.Nk
    #nc[it] = np.sum( (np.abs(np.conj(psik[:,0])*phik[:,0,1] + np.conj(psik[:,1])*phik[:,1,1]))**2 )/param.Nk
    nv[it] = np.sum( (np.abs(np.conj(psik[0,:])*phik[0,0,:] + np.conj(psik[1,:])*phik[1,0,:]))**2 )/kgrid.Nk
    nc[it] = np.sum( (np.abs(np.conj(psik[0,:])*phik[0,1,:] + np.conj(psik[1,:])*phik[1,1,:]))**2 )/kgrid.Nk
    #norm = np.sum(np.abs(psik[:,0])**2 + np.abs(psik[:,1])**2)/param.Nk
    norm = np.sum(np.abs(psik[0,:])**2 + np.abs(psik[1,:])**2)/kgrid.Nk
    #Ene[it] = psikhk_Ene(param,psik,hk)
    #J[it,:], _ = psikvk2JJk(param, psik, vk)
    #Jspin[it,:,:], _ = psikvk2Jspin(param,psik,vk)
    #if (it%1000 == 0):
    if (it%200 == 0):
        print('# ',it, Ene[it], J[:,it], norm)
#
te = time.time()
print_endtime(ts,tt,te,tgrid.Nt)

print_footer() 
sys.exit()
