"""
bound states energies
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
import gc


step = 0.001 # step size
inv = round(1/step)
rang = [-2,3] # range of x axis
n = int((rang[-1]-rang[0])/step) # number of integration intervals

# parabolic potential
def V(x):
	c = 400
	if (x < 0) or (x>1): 
		V = 0
	else:
		V = c*(x**2-x)
	return V


# psi'' + k*psi=0
def k(x,bb):
	return 2*(V(bb)-V(x))

# get index of array from coordiante
def arg(x):
	#return int(round(x*inv+n/2))
	return int(round(x*inv+n*(abs(rang[0])/(abs(rang[0])+abs(rang[-1])))+1)) 

# inverse - get coordinate from index
def inx(i):
	#return (i-n/2)/inv
	return (i-n*(abs(rang[0])/(abs(rang[0])+abs(rang[-1])))-1)/inv 

# calculate bounded states
def bound(b,plotit = False):
	"""
	integrates (Numerov) wavefunction from both sides of interval and tries to match it and its first derivatives
	Param b: choice of x coordinate (E=V(b))
	Param plotit: if True, wavefunction will be plotted
	Return: mismatch parameter
	"""


	# initialize wave functions
	PsiL  = np.zeros(n)
	PsiR  = np.zeros(n)	

	PsiL[arg(0.0)] = 1 #initial setup
	PsiR[arg(1.0)] = 1 #initial setup
	PsiL[arg(-step)] = np.exp(-step*np.sqrt(-2*V(b))) # analytical step
	PsiR[arg(1.00+step)] = np.exp(-step*np.sqrt(-2*V(b))) # analytical step

	# forward integration up to b
	for i in range(round(b*inv)):
		y = i*step
		PsiL[arg(y+step)] = (2*(1-5*step**2/12*k(y,b))*PsiL[arg(y)] - (1+step**2/12*k(y-step,b))*PsiL[arg(y-step)])/(1+step**2/12*k(y+step,b))

	PsiL_val = PsiL[arg(y+step)]
	PsiL_der = (PsiL[arg(y+step)]-PsiL[arg(y)])/step

	# backward integration up to b
	for j in range (round(inv*(1-b))):
		y = 1-j*step
		PsiR[arg(y-step)] = (2*(1-5*step**2/12*k(y,b))*PsiR[arg(y)] - (1+step**2/12*k(y+step,b))*PsiR[arg(y+step)])/(1+step**2/12*k(y-step,b))

	PsiR_val = PsiR[arg(y)]
	PsiR_der = (PsiR[arg(y)]-PsiR[arg(y-step)])/step


	alpha = PsiL_val/PsiR_val
	alpha_der = PsiL_der/PsiR_der


	if plotit == True:

		coord = np.linspace(rang[0],rang[-1],n)
		Psi_final = PsiL+alpha*PsiR
		Psi_final[arg(y-step)]/=2

		#add a tail
		for ii in range(arg(0.0)):
			#print(ii)
			Psi_final[ii] = np.exp(-np.sqrt(-2*V(b_correct))*(-inx(ii)))
		for ii in range(len(Psi_final)):
			if ii>arg(1.0):
				Psi_final[ii] = alpha*np.exp(-np.sqrt(-2*V(b_correct))*(inx(ii)-1.0 ))

		#normalization
		Psi_final = Psi_final/np.sqrt(Psi_final.dot(Psi_final))

		#final plot
		plt.plot(coord,Psi_final,label = 'E=%.3f'%V(b))
		plt.legend()
		plt.xlabel('x')
		plt.ylabel('wavefunction')
		plt.show()

		#plot the potential
		#plt.plot(coord,[V(coord[l]) for l in range(len(coord))])
		#plt.show()	

		del PsiL
		del PsiR
		del Psi_final	 

	return PsiL_der/PsiL_val-PsiR_der/PsiR_val


#find roots
roots = [] # roots array
print('------------root finding------------')
for rt in range(1000):
	try:
		root = scipy.optimize.brenth(bound,0.001*(rt+1),0.001+(1+rt)*0.001)
		#root = scipy.optimize.bisect(bound,0.82,0.84)
		if abs(bound(root)) < 0.001:
			print('b=%.15f, residual=%.5e, E=%.15f'%(root,bound(root),V(root)))
			roots.append(root)
	except Exception:
		pass


for i in range(len(roots)//2,len(roots)):
	b_correct = roots[i]
	bound(b_correct,True)

#to plot differences
differ = np.empty(0)
for z in range(inv):
	diff = bound((z+1)/inv)
	#diff = bound(0.01+z*0.001)
	# print('diff: ',diff)	
	# print((z+1)/inv,'\n')
	differ = np.append(differ,diff)

# mismatch parameter plot
plt.plot(differ,'x',label='root finder')
plt.legend()
plt.xlabel('b')
plt.ylabel('mismatch')
plt.show()













