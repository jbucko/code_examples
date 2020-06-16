import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.special import hyperu as hpu
from scipy.optimize import root,brentq
from mpmath import hyp1f1,hyperu,gamma 
import mpmath as mm
from mpmath import *
from numpy.linalg import norm
from matplotlib import cm
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
import time,sys
import time
from scipy.interpolate import interp2d
from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema
from matplotlib import rc
import multiprocessing
#rc('text', usetex=True)

sys.path.append('/home/jozef/Desktop/master_thesis/GitLab/cm-bilayerboundstates/HamiltonianModel/my_libs')
sys.path.append('/cluster/home/jbucko/master_thesis/my_libs')
from det_funs import*
from waveft_class import*

def sorting_parallel(j,dets_resolved,EinmeVs_resolved,BinTs_resolved,s,m,Rinnm,UinmeV,VinmeV,tinmeV,tau):
	minima_fixed_B = []
	dets_for_fixed_B = dets_resolved[:,j]
	#print(dets_for_fixed_B)
	minima = argrelextrema(dets_for_fixed_B,np.less)[0]
	black_contour_slice = zero_sqrt_out(EinmeVs_resolved,BinTs_resolved[j],s,Rinnm,UinmeV,VinmeV,tinmeV,tau)
	indices = [i for i in range(len(EinmeVs_resolved)-1) if np.sign(black_contour_slice[i]) == -np.sign(black_contour_slice[i+1])]
	#print('minima',minima)
	#print('indices:',indices)

	#print(indices)
	for i in range(len(minima)):
		if minima[i]>indices[0] and minima[i]<indices[1]:
			lb = E_t(EinmeVs_resolved[minima[i]-6],Rinnm)
			ub = E_t(EinmeVs_resolved[minima[i]+6],Rinnm)
			res = minimize_scalar(det,args = (s,r_t(BinTs_resolved[j],Rinnm),m,U0_t(UinmeV,Rinnm),V_t(VinmeV,Rinnm),t_t(tinmeV,Rinnm),tau),method = 'Bounded',bounds = (lb,ub),options={'maxiter': 5})
			Emin = res.x*(6582/10)/Rinnm
			minima_fixed_B.append(Emin)
			#print(Emin)
	return minima_fixed_B

class energy_minima():
	def __init__(self,m,UinmeV,VinmeV,tau,s,Rinnm,tinmeV,BinTmin,BinTmax,nB,nE):
		self.m = m
		self.UinmeV = UinmeV
		self.VinmeV = VinmeV
		self.tau = tau
		self.s = s
		self.Rinnm = Rinnm
		self.tinmeV = tinmeV
		self.BinTmin = BinTmin
		self.BinTmax = BinTmax
		self.nB = nB
		self.nE = nE

		self.BinTs = np.linspace(BinTmin, BinTmax ,nB)
		self.EinmeVs = np.linspace(self.UinmeV-self.VinmeV/2-1,self.UinmeV + self.VinmeV/2+1,nE)


# Bs,Es = np.meshgrid(BinTs,EinmeVs)
# sqrt_in = zero_sqrt_in(Es,Bs,s,self.Rinnm,self.UinmeV,self.VinmeV,self.tinmeV,tau)
# sqrt_out = zero_sqrt_out(Es,Bs,s,self.Rinnm,self.UinmeV,self.VinmeV,self.tinmeV,tau)
	
	def dets_calc(self):
		"""
		determinant calculations
		"""
		print('calculating determinant...')
		ts = time.time()
		i = 0
		dets = []
		# for E in self.EinmeVs:
		# 	for B in self.BinTs:
		# 		det_value = det(E_t(E,self.Rinnm),self.s,r_t(B,self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)
		# 		dets.append(det_value)
		# 	#print('row:',i)
		# 	i+=1

		num_cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=num_cores)
		for E in self.EinmeVs:
			results = [pool.apply_async(det, args=(E_t(E,self.Rinnm),self.s,r_t(B,self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)) for B in self.BinTs]
			along_B = [p.get() for p in results]
			dets.append(along_B)
		pool.close()

		"""
		data postprocessing and interpolation
		"""
		ti = time.time()
		dets = np.reshape(dets,(self.nE,self.nB))
		f = interp2d(self.BinTs,self.EinmeVs,dets, kind = 'linear')
		BinTs_resolved = np.linspace(self.BinTmin, self.BinTmax ,self.nB)
		EinmeVs_resolved = np.linspace(self.UinmeV-self.VinmeV/2-1,self.UinmeV + self.VinmeV/2+1,360)
		dets_resolved = f(BinTs_resolved,EinmeVs_resolved)
		te = time.time()
		print('determinant calculation finished. Ellapsed time: {:.4f},{:.4f}\n'.format(ti-ts,te - ti,'\n'))
		return BinTs_resolved, EinmeVs_resolved, dets_resolved, dets

	def search_minima(self):
		"""
		energy search from density map
		for each field value B we find local minima in the 1D array of interpolated energies
		and then also position of black line (dot edges).
		Within the found range we then do resolved search of the minima
		"""
		t1 = time.time()
		BinTs_resolved,EinmeVs_resolved,dets_resolved,_ = self.dets_calc()
		t2 = time.time()
		all_minima = []

		# for j in range(len(BinTs_resolved)):
		# 	minima_fixed_B = []
		# 	dets_for_fixed_B = dets_resolved[:,j]
		# 	minima = argrelextrema(dets_for_fixed_B,np.less)[0]
		# 	black_contour_slice = zero_sqrt_out(EinmeVs_resolved,BinTs_resolved[j],self.s,self.Rinnm,self.UinmeV,self.VinmeV,self.tinmeV,self.tau)
		# 	indices = [i for i in range(len(EinmeVs_resolved)-1) if np.sign(black_contour_slice[i]) == -np.sign(black_contour_slice[i+1])]
		# 	#print(indices)
		# 	for i in range(len(minima)):
		# 		if minima[i]>indices[0] and minima[i]<indices[1]:
		# 			lb = E_t(EinmeVs_resolved[minima[i]-6],self.Rinnm)
		# 			ub = E_t(EinmeVs_resolved[minima[i]+6],self.Rinnm)
		# 			res = minimize_scalar(det,args = (self.s,r_t(BinTs_resolved[j],self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau),method = 'Bounded',bounds = (lb,ub),options={'maxiter': 5})
		# 			Emin = res.x*(6582/10)/self.Rinnm
		# 			minima_fixed_B.append(Emin)

		# 	all_minima.append(minima_fixed_B)
		# print('\nall_minima',all_minima)
		num_cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=num_cores)
		results = [pool.apply_async(sorting_parallel, args=(j,dets_resolved,EinmeVs_resolved,BinTs_resolved,self.s,self.m,self.Rinnm,self.UinmeV,self.VinmeV,self.tinmeV,self.tau)) for j in range(len(BinTs_resolved))]
		minima_fixed_B = [p.get() for p in results]
		all_minima.extend(minima_fixed_B)
		pool.close()
		#print('\nall_minima',all_minima)

		t3 = time.time()
		print('search_minima times:',t2-t1,t3-t2)
		return BinTs_resolved,EinmeVs_resolved,dets_resolved, all_minima

	def calc_lines(self):
		"""
		here we devide found minima into separate curves
		"""

		placed = False
		all_minima_sorted = []

		t1 = time.time()
		BinTs_resolved,EinmeVs_resolved, dets_resolved, all_minima = self.search_minima()
		t2 = time.time()
		#print(all_minima,len(all_minima))
		l = 0
		while len(all_minima_sorted) == 0 and l<len(all_minima):
			for i in range(len(all_minima[l])):
				#print(all_minima[0])
				#print(det(E_t(all_minima[0][i],self.Rinnm),s,r_t(BinTs_resolved[0],self.Rinnm),m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),tau))
				if det(E_t(all_minima[l][i],self.Rinnm),self.s,r_t(BinTs_resolved[0],self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)<0.006:
					all_minima_sorted.append([all_minima[l][i]])
			l+=1
		#print('l:',l)
		if l<len(all_minima):
			for i in range(l,len(all_minima)):
				occupied = []
				for j in range(len(all_minima[i])):
					diff = np.array([abs(all_minima[i][j]-all_minima_sorted[k][-1]) for k in range(len(all_minima_sorted))])
					idx = np.where(diff == np.min(diff))[0][0]
					# print('diff, idx;',diff,idx)
					if diff[idx] < 0.5 and idx not in occupied:
						all_minima_sorted[idx].append(all_minima[i][j])
						occupied.extend([idx])
					else:
						if det(E_t(all_minima[i][j],self.Rinnm),self.s,r_t(BinTs_resolved[i],self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)<0.006:
							all_minima_sorted.append([all_minima[i][j]])
					# for k in range(len(all_minima_sorted)):
					# 	if abs(all_minima[i][j]-all_minima_sorted[k][-1])<0.5 and not placed:
					# 		all_minima_sorted[k].append(all_minima[i][j])
					# 		placed = True
					# if not placed:
					# 	#print('not placed:',all_minima[i][j],'from:',[all_minima_sorted[k][-1] for k in range(len(all_minima_sorted))])
					# 	#print(det(E_t(all_minima[i][j],self.Rinnm),s,r_t(BinTs_resolved[i],self.Rinnm),m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),tau))
					# 	if det(E_t(all_minima[i][j],self.Rinnm),self.s,r_t(BinTs_resolved[i],self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)<0.006:
					# 		all_minima_sorted.append([all_minima[i][j]])
					# placed = False

		#print('\n',all_minima_sorted,'\n')

		t3 = time.time()
		all_minima_sorted_valid = []
		for i in range(len(all_minima_sorted)):
			length = len(all_minima_sorted[i])
			print('length comparisons:',length,0.999*len(BinTs_resolved))
			if length >= 0.999*len(BinTs_resolved):
				#print('length:',length,'valid line:',all_minima_sorted[i])
				all_minima_sorted_valid.append(all_minima_sorted[i])


		#print('calc_lines times:',t3-t2,t2-t1)
		return BinTs_resolved,EinmeVs_resolved,dets_resolved, all_minima_sorted_valid





if __name__ == '__main__':

	################---main controls----################
	raw = 1 # if set to 1, also raw figure 50x50 px is generated
	frame = 1 # if set to 1, frame, axis labels and colorbar are displayed
	s = 1
	m = 0
	tau = 1
	Rinnm = 20
	tinmeV = 400
	UinmeV = 60
	VinmeV = 50
	nB = 50
	nE = 50

	BinTmin = 0.0535
	BinTmax = 2.5
	#################################

	t1 = time.time()

	energy_curve_class = energy_minima(m,UinmeV,VinmeV,tau,s,Rinnm,tinmeV,BinTmin,BinTmax,nB,nE)


	Bs,Es,lines = energy_curve_class.calc_lines()
	print(len(lines),len(lines[0]))
	# for i in range(len(lines)):
	# 	plt.plot(Bs,lines[i])
	# plt.show()

	t2 = time.time()
	print('----------total time {:.4f}------------'.format(t2-t1))
