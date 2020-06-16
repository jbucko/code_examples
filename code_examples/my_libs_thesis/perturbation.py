import numpy as np
from scipy.linalg import null_space
from numpy.linalg import norm
from scipy import integrate
import matplotlib.pyplot as plt
from det_funs import*
from waveft_class import *

class pert_shift:

	def __init__(self,EinmeV_arr,BinT_arr,s,m,tau,Rinnm,tinmeV,UinmeV,VinmeV,p):

		#self.ksi = ksi
		self.EinmeV_arr = EinmeV_arr
		self.BinT_arr = BinT_arr
		self.s = s
		self.m = m
		self.tau = tau
		self.Rinnm = Rinnm
		self.tinmeV = tinmeV
		self.UinmeV = UinmeV
		self.VinmeV = VinmeV
		self.p = p

	def phi_pert(self,ksi,rho):
		return np.arctan(np.sqrt((1-self.p*ksi**2/rho**2) / (ksi**2/rho**2 - 1)))


	def E_shift_arr(self,):
		E_pert_arr = []
		for i in range(len(self.EinmeV_arr)):
			wfun = psi_complete(self.EinmeV_arr[i],self.BinT_arr[i],self.s,self.m,self.tau,self.Rinnm,self.tinmeV,self.UinmeV,self.VinmeV)
			d_norm = wfun.psi_sq_norm()
			rho = r_t(self.BinT_arr[i],self.Rinnm)
			integrand = lambda ksi: 4*self.UinmeV/2/np.pi/d_norm*self.phi_pert(ksi,rho)*wfun.psisq_out(ksi)

			E = integrate.quad(integrand,rho,rho/np.sqrt(self.p))
			E_pert_arr.append(E[0])
		return E_pert_arr



