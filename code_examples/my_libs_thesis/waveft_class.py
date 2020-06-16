import numpy as np
from scipy.linalg import null_space
from numpy.linalg import norm
from scipy import integrate
from det_funs import*

class psi_complete:

	def __init__(self,EinmeV,BinT,s,m,tau,Rinnm,tinmeV,UinmeV,VinmeV,norm_finite = False):

		#self.ksi = ksi
		self.EinmeV = EinmeV
		self.BinT = BinT
		self.s = s
		self.m = m
		self.tau = tau
		self.Rinnm = Rinnm
		self.tinmeV = tinmeV
		self.UinmeV = UinmeV
		self.VinmeV = VinmeV
		self.norm_finite = norm_finite
		self.r = r_t(self.BinT,self.Rinnm)


	def calc_null_space(self):
		rcond = 0.001
		while(1):
			null_sp,matrix = det(E_t(self.EinmeV,self.Rinnm),self.s,r_t(self.BinT,self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,True,False,rcond,True)
			if len(null_sp) == 0:
				rcond *=2
			else:
				break
		if rcond > 0.1:
			print('WARNING!!! singular values are quite large!')
		return null_sp,matrix

	def vec1(self,ksi):
		#r = r_t(self.BinT,self.Rinnm)
		v1 = psi1_in(self.r,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,1)
		return 1/norm(v1,ord = 2,axis = 0)*psi1_in(ksi,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,1)

	def vec2(self,ksi):
		#r = r_t(self.BinT,self.Rinnm)
		v2 = psi1_in(self.r,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,2)
		return 1/norm(v2,ord = 2,axis = 0)*psi1_in(ksi,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,2)

	def vec3(self,ksi):		
		#r = r_t(self.BinT,self.Rinnm)
		v3 = psi1_out(self.r,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,1)
		return 1/norm(v3,ord = 2,axis = 0)*psi1_out(ksi,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,1)

	def vec4(self,ksi):
		#r = r_t(self.BinT,self.Rinnm)
		v4 = psi1_out(self.r,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,2)
		return 1/norm(v4,ord = 2,axis = 0)*psi1_out(ksi,self.s,self.r,self.m,E_t(self.EinmeV,self.Rinnm),U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau,2)

	def psi_in(self,ksi):
		z_space = self.calc_null_space()[0]
		return z_space[0][0]*self.vec1(ksi) + z_space[1][0]*self.vec2(ksi)

	def psi_out(self,ksi):
		z_space = self.calc_null_space()[0]
		return -z_space[2][0]*self.vec3(ksi) - z_space[3][0]*self.vec4(ksi)

	def psisq_in(self,ksi):
		sq = self.psi_in(ksi).conj().T.dot(self.psi_in(ksi))
		#print(ksi,sq)
		if np.imag(sq)[0][0] > 1e-7:
			print('complex part in normalization of psi_in')
		return np.real(sq)[0][0]

	def psisq_out(self,ksi):
		sq = self.psi_out(ksi).conj().T.dot(self.psi_out(ksi))
		#print(ksi,sq)
		if np.imag(sq)[0][0] > 1e-7:
			print('complex part in normalization of psi_out')
		return np.real(sq)[0][0]

	def psi_sq_norm(self):
		print('\ncalculating normalization...')
		if self.norm_finite:
			upper_bound = 3
		else:
			upper_bound = np.inf
		int1 = integrate.quad(self.psisq_in,0,r_t(self.BinT,self.Rinnm),epsabs=1e-02,epsrel = 1e-2,limit = 1)
		int2 = integrate.quad(self.psisq_out,r_t(self.BinT,self.Rinnm),upper_bound,epsabs=1e-02,epsrel = 1e-2,limit = 1)
		# int1 = integrate.romberg(self.psisq_in,0,r_t(self.BinT,self.Rinnm))
		# int2 = integrate.romberg(self.psisq_out,r_t(self.BinT,self.Rinnm),upper_bound)
		print('integrals:',int1[0],int2[0])
		print('\ncalculation finished...\n')
		return int1[0]+int2[0]

	def psisq_joint(self,ksi):
		if ksi<self.r:
			return self.psisq_in(ksi)
		else:
			return self.psisq_out(ksi)

	def psisq_joint_elements(self,ksi):
		if ksi<self.r:
			return np.array([abs(self.psi_in(ksi)[0])**2,abs(self.psi_in(ksi)[1])**2,abs(self.psi_in(ksi)[2])**2,abs(self.psi_in(ksi)[3])**2])
		else:
			return np.array([abs(self.psi_out(ksi)[0])**2,abs(self.psi_out(ksi)[1])**2,abs(self.psi_out(ksi)[2])**2,abs(self.psi_out(ksi)[3])**2])


# EinmeV = 30
# BinT = 1.3
# s = 1
# m = 0
# tau = 1
# Rinnm = 20
# tinmeV = 400
# UinmeV = 40
# VinmeV = 45

# waveft = psi_complete(EinmeV,BinT,s,m,tau,Rinnm,tinmeV,UinmeV,VinmeV)
# print(waveft.calc_null_space())