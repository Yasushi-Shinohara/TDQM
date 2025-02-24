# coding: UTF-8
# This is creaed 2023/12/23 by Y. Shinohara
# This is lastly modified 20xx/yy/zz by Y. Shinohara
import sys
from modules.constants import *
import logging
from modules.general_functions import dict_to_namedtuple
from modules.functions import grad1D_uniform_grid, lap1D_uniform_grid
import numpy as np
from scipy.sparse import dok_array
import scipy as sp

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
        if model_d['modelID'] == "1DHarmonic-oscillator":
            x = np.linspace(-model_d['Lbox1']/2.0, model_d['Lbox1']/2.0, model_d['N1'])
            dx = x[1] - x[0]
            model_d.update(x = x, dx = dx)
            velocity = cls.velocity1D
            tkin = cls.tkin1D
            vpot = cls.vpot_1DHarmonic_oscillator
            model_d.update(velocity = velocity, tkin = tkin, vpot = vpot)
            set_GS_WF = cls.set_GS_WF_1DHarmonic_oscillator
            model_d.update(set_GS_WF = set_GS_WF)
            model_d.update(get_h_eigenpairs = cls.get_h_eigenpairs)
        model = dict_to_namedtuple('model_namedtuple', model_d)

        return model

    @staticmethod
    def velocity1D(model):
        """Hamiltonian at a given k-point within LQZDFZ surface model."""
        velocity1D = grad1D_uniform_grid(model.N1, model.dx)/zI

        return velocity1D
    @staticmethod
    def tkin1D(model):
        """Hamiltonian at a given k-point within LQZDFZ surface model."""
        tkin1D = -0.5*lap1D_uniform_grid(model.N1, model.dx)

        return tkin1D
    @staticmethod
    def vpot_1DHarmonic_oscillator(model):
        """Hamiltonian at a given k-point within LQZDFZ surface model."""
        vpot = dok_array((model.N1, model.N1), dtype=np.float64)
        for i in range(model.N1):
            vpot[i, i] = 0.5*model.Omega**2*model.x[i]**2
        vpot = vpot.tocsr()  # 計算時には CSR
        return vpot
    @staticmethod
    def set_GS_WF_1DHarmonic_oscillator(x0, delx, p0, model):
        """Hamiltonian at a given k-point within LQZDFZ surface model."""
        phi = np.exp(-0.5*(model.x - x0)**2/delx**2)*np.exp(zI*p0*model.x)
        phi = phi/(np.sqrt(np.sum((np.abs(phi))**2)))
        return phi
    @staticmethod
    def get_h_eigenpairs(h):
        """Hamiltonian at a given k-point within LQZDFZ surface model."""
#        eig, vec =  np.linalg.eigh(h)
        eig =  sp.sparse.linalg.eigsh(h, k=6, return_eigenvectors = False, which='LA')
        print("The largest eigenvalues", eig)
        eig, vec =  sp.sparse.linalg.eigsh(h, k=6, which='SA')
        print("The smallest eigenvalues", eig)
        return eig, vec
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
#            temp = - zI*np.dot(h, temp)/float(norder)
            temp = - (zI*dt)*h@temp/float(norder)
            wf = wf + temp
        return wf

#
class Options:
    @classmethod
    def construct_options_namedtuple(cls, param):
        options_d = param["options"]
        options = dict_to_namedtuple('options_namedtuple', options_d)

        return options
    







