"""
Calculates integral cross section of scattering of hydrogen atoms on krypton
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
plt.rcParams['text.usetex'] = True

eps = 5.9 # zadanie
mhbar = 6.12
sigma = 1.0 # angstrom
start = 0.5*sigma # r_min
stop = 5.0*sigma # r_max
r1 = 0.93*stop
r2 = 0.98*stop


def j(l,z):
	"""
	Spherical Bessel Function
	var l: order of Bessel
	var z: varialbe
	-------
	Return: j_l(z)
	"""
	return ss.spherical_jn(l,z)

def y(l,z):
	"""
	Neumann Function
	var l: order of Bessel
	var z: varialbe
	-------
	Return: n_l(z)
	"""
	return ss.spherical_yn(l,z)

def numerov_step(k0, k1, k2, psi0, psi1, dx):
	"""
	Perform one Numerov step given psi0 and psi1.
	---------
	Return psi2
	"""
	dd = dx**2/12
	c0 = (1+dd*k0)
	c1 = 2*(1-5*dd*k1)
	c2 = (1+dd*k2)
	return (c1*psi1-c0*psi0)/c2

def Numerov_integrate(k, n, r_min, r_max, l, E):
	"""
	Perform Numerov integration
	var k: function k(r)
	var n: number of steps
	var r_min, r_max: region of integration
	var l: partial wave order
	var E: energy of incoming particle
	--------------
	Return: [u0, u1, u2,..., un]
	"""
	dx = (r_max - r_min)/(n-1)
	k0, k1 = k(r_min,l,E), k(r_min + dx,l,E)

	C = np.sqrt(mhbar*eps/25)
	u0 = np.exp(-C/r_min**5)
	u1 = np.exp(-C/(r_min+dx)**5)

	u = np.zeros(n, dtype = np.float128)
	u[:2] = u0, u1

	for i in range(2,n):
		k2 = k(r_min + i*dx,l,E)
		u[i] = numerov_step(k0, k1, k2, u[i-2], u[i-1],dx)
		k0, k1 = k1, k2

	return u

def V(r):
	"""
	var r: radial distance
	-------
	Return: value of potential at r
	"""
	return eps*((sigma/r)**12-2*(sigma/r)**6)

	

k = lambda r,l,E:  mhbar*E - mhbar*V(r) - l*(l+1)/r**2


def phase(l,E):
	"""
	Calculates phase of scattering
	var l: partial wave order
	var E: energy of incoming particle
	--------
	Return: phase shift d_l, integrated radial part of wave function

	"""

	u = Numerov_integrate(k, 1000, start, stop,l,E)

	rs = np.linspace(start,stop,1000)
	#plt.plot(rs, u)
	#plt.show()

	r1 = rs[-10]
	r2 = rs[-3]
	u1 = u[-10]
	u2 = u[-3]
	K = r1*u2/(r2*u1)
	k_s = np.sqrt(mhbar*E)

	delta_l = np.arctan((K*j(l,k_s*r1)-j(l,k_s*r2))/(K*y(l,k_s*r1)-y(l,k_s*r2)))

	return delta_l, u

### Main ###


def cross(E, lmax):
	"""
	calculate integral cross section
	var E: total energy of incoming particle
	var lmax: truncating of the partial waves series
	-----------
	Return cross: cross section
	"""

	cross = 0
	for l in range(lmax):
		delta = phase(l,E)[0]
		cross+=4*np.pi/(mhbar*E)*(2*l+1)*np.sin(delta)**2
	return cross

Ee = np.linspace(0.3,4.0,70) # energy range

for i in range(1,11):
	print('up to l_max =',i)
	lmax = i*1
	crosses = []

	for e in Ee:
		#print(e)
		crosses.append(cross(e,lmax))
	crosses = np.array(crosses)
	plt.plot(Ee,crosses, label = '$l_{max} = %g$' %lmax)
	plt.ylabel('$\sigma_{tot} [\sigma^{-2}]$')
	plt.xlabel('E [meV]')
	plt.legend(loc = 'upper right')

plt.show()