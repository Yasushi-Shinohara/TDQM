# coding: UTF-8
# This is creaed 2023/12/23 by Y. Shinohara
# This is lastly modified 20xx/yy/zz by Y. Shinohara
import sys
from modules.constants import *
import logging
from modules.general_functions import dict_to_namedtuple
import numpy as np

class Unit:
    """A class to manage unit/dimension in TDkpModel. """
    @classmethod
    def construct_unit_namedtuple(cls, param):
        unit_d = param["unit"]
        unit = dict_to_namedtuple('unit_namedtuple', unit_d)

        return unit

class Model:
    """A class to manage model, Hamiltonian, velocity operator, in TDkpModel. """
    @classmethod
    def construct_model_namedtuple(cls, param):
        model_d = param["model"]
        if model_d['modelID'] == "LQZDFZ-surface":
            k2h = cls.k2h_LQZDFZ_surface
            k2hk = cls.k2hk_LQZDFZ_surface
            k2v = cls.k2v_LQZDFZ_surface
            k2vk = cls.k2vk_LQZDFZ_surface
            k2w = cls.k2w_LQZDFZ_surface
            k2wk = cls.k2wk_LQZDFZ_surface
            model_d.update(k2h = k2h, k2hk = k2hk, k2v = k2v, k2vk = k2vk, k2w = k2w, k2wk = k2wk)
        model = dict_to_namedtuple('model_namedtuple', model_d)

        return model

    @staticmethod
    def k2h_LQZDFZ_surface(kx,ky,kz,model):
        """Hamiltonian at a given k-point within LQZDFZ surface model."""
        k2 = kx**2 + ky**2
        h = (model.C0 + model.C2*k2)*sigma0
        h += model.A*(ky*sigma1 - kx*sigma2)
        h += 2.0*model.R*(kx**3 - 3.0*kx*ky**2)*sigma3
        return h

    @staticmethod
    def k2hk_LQZDFZ_surface(kx,ky,kz,model):
        """Hamiltonian for whole k-point set within LQZDFZ surface model."""
        k2 = kx**2 + ky**2
        hk = np.zeros([2,2, len(k)], dtype='complex128')
        for nk in range(len(k)):
            hk[:,:, nk] = (model.C0 + model.C2*k2[nk])*sigma0
            hk[:,:, nk] += model.A*(ky[nk]*sigma1 - kx[nk]*sigma2)
            hk[:,:, nk] += 2.0*model.R*(kx[nk]**3 - 3.0*kx[nk]*ky[nk]**2)*sigma3
        return h

    @staticmethod
    def k2v_LQZDFZ_surface(kx,ky,kz,model):
        """Velocity operator at a given k-point within LQZDFZ surface model."""
        vx = 2.0*model.C2*kx*sigma0
        vx += -model.A*sigma2
        vx += 6.0*model.R*(kx**2 - ky**2)*sigma3
        vy = 2.0*model.C2*ky*sigma0
        vy += model.A*sigma1
        vy += -12.0*model.R*kx*ky*sigma3
        vz = None
        return vx, vy, vz

    @staticmethod
    def k2vk_LQZDFZ_surface(kx,ky,kz,model):
        """Velocity operators  for whole k-point set within LQZDFZ surface model."""
        vkx = np.zeros([2,2, len(k)], dtype='complex128')
        vky = np.zeros([2,2, len(k)], dtype='complex128')
        vkz = None
        for nk in range(len(k)):
            vkx[:,:, nk] = 2.0*model.C2*kx[nk]*sigma0
            vkx[:,:, nk] += -model.A*sigma2
            vkx[:,:, nk] += 6.0*model.R*(kx[nk]**2 - ky[nk]**2)*sigma3
            vky[:,:, nk] = 2.0*model.C2*ky[nk]*sigma0
            vky[:,:, nk] += model.A*sigma1
            vky[:,:, nk] += -12.0*model.R*kx[nk]*ky[nk]*sigma3
        return vkx, vky, vkz

    @staticmethod
    def k2w_LQZDFZ_surface(kx,ky,kz,model):
        """Spin-velocity operator at a given k-point within LQZDFZ surface model."""
        w1x = 2.0*model.C2*kx*sigma1
        w2x = 2.0*model.C2*kx*sigma2 - model.A*sigma0
        w3x = 2.0*model.C2*kx*sigma3 + 6.0*model.R*(kx**2-ky**2)*sigma0
        w1y = 2.0*model.C2*ky*sigma1 + model.A*sigma0
        w2y = 2.0*model.C2*ky*sigma2
        w3y = 2.0*model.C2*ky*sigma3 -12.0*model.R*(kx**2-ky**2)*sigma0
        w1z = None
        w2z = None
        w3z = None
        return w1x, w2x, w3x, w1y, w2y, w3y, w1z, w2z, w3z

    @staticmethod
    def k2wk_LQZDFZ_surface(kx,ky,kz,model):
        """Spin-velocity operators for whole k-point set within LQZDFZ surface model."""
        wk1x = np.zeros([2,2, len(k)], dtype='complex128')
        wk2x = np.zeros([2,2, len(k)], dtype='complex128')
        wk3x = np.zeros([2,2, len(k)], dtype='complex128')
        wk1y = np.zeros([2,2, len(k)], dtype='complex128')
        wk2y = np.zeros([2,2, len(k)], dtype='complex128')
        wk3y = np.zeros([2,2, len(k)], dtype='complex128')
        wk1z = None
        wk2z = None
        wk3z = None
        for nk in range(len(k)):
            w1x[:,:, nk] = 2.0*model.C2*kx[nk]*sigma1
            w2x[:,:, nk] = 2.0*model.C2*kx[nk]*sigma2 - model.A*sigma0
            w3x[:,:, nk] = 2.0*model.C2*kx[nk]*sigma3 + 6.0*model.R*(kx**2-ky**2)*sigma0
            w1y[:,:, nk] = 2.0*model.C2*ky[nk]*sigma1 + model.A*sigma0
            w2y[:,:, nk] = 2.0*model.C2*ky[nk]*sigma2
            w3y[:,:, nk] = 2.0*model.C2*ky[nk]*sigma3 -12.0*model.R*(kx**2-ky**2)*sigma0
        return wk1x, wk2x, wk3x, wk1y, wk2y, wk3y, wk1z, wk2z, wk3z
#
class TGrid:
    """A class to manage tgrid in TDkpModel. """
    @classmethod
    def construct_tgrid_namedtuple(cls, param):
        tgrid_d = param["tgrid"]        
        t = cls.generate_tgrid(tgrid_d)
        tgrid_d.update(t = t)
        tgrid = dict_to_namedtuple('tgrid_namedtuple', tgrid_d)

        return tgrid

    @staticmethod
    def generate_tgrid(tgrid_d):
        tend = (tgrid_d["Nt"] - 1)*tgrid_d["dt"]
        t = np.linspace(0.0, tend, tgrid_d["Nt"])
        
        return t
#
class Field:
    """A class to manage E/A-fields in TDkpModel. """
    @classmethod
    def construct_field_namedtuple(cls, param, tgrid):
        field_d = param["field"]
        field_d.update(phiCEP1 = field_d["phiCEP1"]*tpi)
        field_d.update(phiCEP2 = field_d["phiCEP2"]*tpi)
        field_d.update(phiCEP3 = field_d["phiCEP3"]*tpi)
        A, E = cls.generate_field(field_d, tgrid)
        field_d.update(E = E, A = A)
        field = dict_to_namedtuple('field_namedtuple', field_d)

        return field

    @staticmethod
    def generate_field(field_d, tgrid):
        """(Multicolor) field construction"""
        t = tgrid.t
        Nt = tgrid.Nt
        dt = tgrid.dt
        A1amp = field_d["E1"]/field_d["omega1"]
        A2amp = field_d["E2"]/field_d["omega2"]
        A3amp = field_d["E3"]/field_d["omega3"]
        edir1 = np.array(field_d["edir1"])
        edir2 = np.array(field_d["edir2"])
        edir3 = np.array(field_d["edir3"])
        A1 = np.zeros([3, Nt], dtype="float64")
        A2 = np.zeros([3, Nt], dtype="float64")
        A3 = np.zeros([3, Nt], dtype="float64")
        #Making envelopes
        for nt in range(Nt):
            if (t[nt] < field_d["Tpulse1"]):
                A1[:,nt] = edir1*(np.sin(pi*t[nt]/field_d["Tpulse1"]))**field_d["nenvelope1"]
            if (t[nt] < field_d["Tpulse2"]):
                A2[:,nt] = edir2*(np.sin(pi*t[nt]/field_d["Tpulse2"]))**field_d["nenvelope2"]
            if (t[nt] < field_d["Tpulse3"]):
                A3[:,nt] = edir3*(np.sin(pi*t[nt]/field_d["Tpulse3"]))**field_d["nenvelope3"]
        for nxyz in range(3):
            A1[nxyz, :] = A1[nxyz, :]*A1amp*np.sin(field_d["omega1"]*(t - 0.5*field_d["Tpulse1"]) + field_d["phiCEP1"])
            A2[nxyz, :] = A2[nxyz, :]*A2amp*np.sin(field_d["omega2"]*(t - 0.5*field_d["Tpulse2"]) + field_d["phiCEP2"])
            A3[nxyz, :] = A3[nxyz, :]*A3amp*np.sin(field_d["omega3"]*(t - 0.5*field_d["Tpulse3"]) + field_d["phiCEP3"])
        A = A1 + A2 + A3
        E = np.zeros_like(A)
        for nt in range(1, Nt-1):
            E[:, nt] = (A[:, nt+1] - A[:, nt-1])/2.0/dt
        E[:, Nt - 1] = 2.0*E[:, nt -2] - E[:, nt -3] 
        E = -1.0*E
        return A, E
#
class TEvolution:
    """A class to specify time-evolution in TDkpModel. """
    @classmethod
    def construct_tevolution_namedtuple(cls, param):
        tevolution_d = param["tevolution"]
        if tevolution_d['propagator'] == "exp":
            get_psi_forward = cls.get_psi_forward_exp
            tevolution_d.update(get_psi_forward = get_psi_forward)
        elif tevolution_d['propagator'].upper() == "TE4":
            get_psi_forward = cls.get_psi_forward_TE4
            tevolution_d.update(get_psi_forward = get_psi_forward)
        tevolution = dict_to_namedtuple('tevolution_namedtuple', tevolution_d)

        return tevolution

    @staticmethod
    def get_psi_forward_exp(wf, h, dt):
        #U = clf.h2U(h)
        U = TEvolution.h2U(h, dt)
        wf = np.dot(U, wf)
        return wf

    @staticmethod
    def h2U(h, dt):
        w, v = np.linalg.eigh(h)
        #U = np.exp(-zI*w[0]*tgrid.dt)*np.outer(v[0,:],np.conj(v[0,:])) + np.exp(-zI*w[1]*tgrid.dt)*np.outer(v[1,:],np.conj(v[1,:]))
        U = np.exp(-zI*w[0]*dt)*np.outer(v[:,0],np.conj(v[:,0])) + np.exp(-zI*w[1]*dt)*np.outer(v[:,1],np.conj(v[:,1]))        
        return U

    @staticmethod
    def get_psi_forward_TE4(wf, h, dt, NTEorder = 4):
        temp = 1.0*wf
        for norder in range(1, NTEorder+1):
            temp = - zI*np.dot(h, temp)/float(norder)
            wf = wf + temp
        return wf

#
class Options:
    @classmethod
    def construct_options_namedtuple(cls, param):
        options_d = param["options"]
        options = dict_to_namedtuple('options_namedtuple', options_d)

        return options
    







