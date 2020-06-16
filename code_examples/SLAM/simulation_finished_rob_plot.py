"""
this code simulates analytical SLAM with both radial and angluar information
"""

import numpy as np
import scipy as sc
from scipy import optimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib
import time
from circle_draw import draw_circle as circle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
plt.rc('text', usetex=True)
start_time = time.time()

glob=True 		# whether to plot situation in local or global coordinates
nl=3 			# number of landmarks
h = 0.01 		# time step
t_max=15 		# end of integration
#type of trajectory
circular = 1 
parabola = 0

# initial position of landmark in circle
if circular:
	x_0l1=-5.0
	y_0l1=2.
	#errors in initial landmark's position
	errx1=3 
	erry1=-1

	x_0l2=3.0
	y_0l2=1.
	#errors in initial landmark's position
	errx2=1 
	erry2=-2

	x_0l3=-5.0
	y_0l3=-4.
	#errors in initial landmark's position
	errx3=1 
	erry3=3


# initial positions in parabola
if parabola:
	x_0l1=15.0
	y_0l1=2.
	#errors in initial landmark's position
	errx1=3 
	erry1=-1

	x_0l2=4.0
	y_0l2=15.
	#errors in initial landmark's position
	errx2=1 
	erry2=-2

	x_0l3=12.0
	y_0l3=10.
	#errors in initial landmark's position
	errx3=1 
	erry3=3

# initial positions
x_0l=np.array([x_0l1,x_0l2,x_0l3,2])
y_0l=np.array([y_0l1,y_0l2,y_0l3,4])
# joint array of initial positions of landmarks and vehicle
xinit = np.array([x_0l1+errx1,y_0l1+erry1,x_0l2+errx2,y_0l2+erry2,x_0l3+errx3,y_0l3+erry3,0.,0.])

r_max=20. # maximal range

""" defining quantities """
ang_v=0.5 #angular velocity of robot's movement
wz_mean=.3 # rotation of robot
# deviations
sigma_wz=0.05
sigma_ux=.01
sigma_uy=.01 
sigma_th=0.05 
sigma_r=0.02 
# scales of model and measurement uncertainties
q_scale=1 
r_scale=1

def wz(t):
	"""
	returns normally distributed angular velocity
	"""
	return np.random.normal(wz_mean,sigma_wz)
	#return wz_mean
	#return np.random.uniform(wz_mean-sigma_wz,wz_mean+sigma_wz)

def Omega(t,wz):
	'''
	transforms angular velocity to 2D antisymmetric tesor
	param beta: rotation angle
	return: rotation matrix
	'''
	return np.array([[0,-wz],[wz,0]])


def m_Tr_2x2(beta):
	'''
	rotation in 2D by beta - used to tranform at the end to global frame
	param beta: rotation angle
	return: rotation matrix
	'''
	m_Beta22=np.array([[np.cos(beta),-np.sin(beta)],[np.sin(beta),np.cos(beta)]])
	return m_Beta22


# def H_glob(th,beta):
# 	m_Beta22=np.array([[np.cos(beta),-np.sin(beta)],[np.sin(beta),np.cos(beta)]])
# 	#print(m_Beta22)
# 	m_Beta = np.vstack([np.column_stack([m_Beta22,np.zeros((2,2))]),np.column_stack([np.zeros((2,2)),m_Beta22])])
# 	m_Hh=np.array([[np.cos(th), -np.sin(th),-np.cos(th), np.sin(th)],[np.sin(th), np.cos(th),-np.sin(th), -np.cos(th)]  ])#,\
# 		#[0,0,1,0],[0,0,0,1]])
# 	#print(m_Beta)
# 	#print(m_Hh)
# 	return np.dot(m_Hh,m_Beta) 

def H(th):
	'''
	H-matrix y = H*x
	param theta: angle to landmark in local frame
	return: H-matrix after considering splitting to virtual vehicles
	'''
	m_Hh=np.array([[np.cos(th), -np.sin(th),-np.cos(th), np.sin(th)],[np.sin(th), np.cos(th),-np.sin(th), -np.cos(th)],\
		[0,0,1,0],[0,0,0,1]])
	return m_Hh

def cov_R(sig_th,rr,x_vc,y_vc,x_vj,y_vj): 
	'''
	covariance matrix R of y-Hx
	param sig_th: variance of theta measurement
	param rr: maximal range
	params x_vc, y_vc: (x,y) local coordinates of previous step consensus about vehocle's position
	params x_vc, y_vc: (x,y) local coordinates of j-th landmark
	return: R-matrix
	'''
	return r_scale*np.array([[sig_th**2*(rr**2+sigma_r**2),0,0,0],[0,sig_th**4/4*(rr**2+sigma_r**2),0,0],[0,0,sig_th**4/4*(rr**2+sigma_r**2)+(x_vc-x_vj)**2,0],[0,0,0,sig_th**4/4*(rr**2+sigma_r**2)+(y_vc-y_vj)**2]])

def u_x(t):
	if circular:
		return np.random.normal(-ang_v*np.sin(ang_v*t),sigma_ux)
	if parabola:
		return 0.1*t

def u_y(t):
	if circular:
		return np.random.normal(ang_v*np.cos(ang_v*t),sigma_uy)
	if parabola:
		return 1

def x_G(t):
	if circular:
		return np.cos(ang_v*t)-1
	if parabola:
		return 0.1*t**2/2

def y_G(t):
	if circular:
		return np.sin(ang_v*t)
	if parabola:
		return t

def ux_tran(t,wz): #only simulation of velocity estimate --wz is exact as u has its own error
	m_Tr=np.array([[np.cos(wz*t),np.sin(wz*t)],[-np.sin(wz*t),np.cos(wz*t)]])
	return np.dot(m_Tr,np.array([u_x(t),u_y(t)]))[0]

def uy_tran(t,wz):
	m_Tr=np.array([[np.cos(wz*t),np.sin(wz*t)],[-np.sin(wz*t),np.cos(wz*t)]])
	return np.dot(m_Tr,np.array([u_x(t),u_y(t)]))[1]

def r_measurement(t,x_l,y_l):
	return np.random.normal(np.sqrt((x_l-x_G(t))**2+(y_l-y_G(t))**2),sigma_r)

def th_measurement(t,x_l,y_l):

	th_mean = (np.arctan2((x_l-x_G(t)),(y_l-y_G(t)))+wz_mean*t)
	#th_mean = wz*t
	return np.random.normal(th_mean,sigma_th)
	#return th_mean

def feval(funcName, *args):
	'''
	evalution of function
	param funcName: name of a function
	param *args: arguments of function
	return: funcName(args)
	'''
	return eval(funcName)(*args)

#-----------------------------update functions---------------------------------------#
def myFunc(t, x, P, theta,r_range,omega,x_vj,y_vj,x_vc,y_vc):
	"""
	calculates coordinates differential
	Param t: integration time
	Param x: coordinate vector in form [x_land,y_land,x_veh,y_veh]
	Param P: covariance matrix of position
	Param theta: angle in local coord
	Param r_range: distance to landmark
	Param omega: ang. velocity
	Param x_vj: x coord. of j-th landmark
	Param y_vj: y -||-
	Param x_vc: x of vehicle
	Param y_vc: y of vehicle
	"""
	dx = np.zeros((len(x)))
	y_vir=np.array([0,r_range,x_vc,y_vc])
	m_H=H(theta)
	m_R=cov_R(sigma_th,r_range+3*sigma_r,x_vc,y_vc,x_vj,y_vj)
	m_K=P.dot(np.transpose(m_H)).dot(np.linalg.inv(\
		H(theta).dot(P).dot(np.transpose(H(theta)))+m_R))
	ym=np.dot(m_H,x)

	OMEGA=np.vstack([np.column_stack([omega,np.zeros((2,2))]),np.column_stack([np.zeros((2,2)),omega])])
	dx = np.array([0,0,ux_tran(t,wz_mean),uy_tran(t,wz_mean)])-np.dot(OMEGA,x)+np.dot(m_K,(y_vir-ym))

	return dx

def myFunc1(t, P, theta, r_range, xx,omega,x_vj,y_vj,x_vc,y_vc):
	"""
	calculates differential of covariance matrix
	Params: see previous function
	"""
   
	m_Q = q_scale*sigma_wz**2*r_max**2*np.array([[1.1,-1,1,-1],[-1,1.1,-1,1],\
		[1,-1,1+sigma_ux**2/(sigma_wz**2*r_max**2),-1],[-1,1,-1,1+sigma_uy**2/(sigma_wz**2*r_max**2)]])
	m_H=H(theta)
	m_R=cov_R(sigma_th, r_range+3*sigma_r,x_vc,y_vc,x_vj,y_vj)
	m_K=P.dot(np.transpose(m_H)).dot(np.linalg.inv(\
		H(theta).dot(P).dot(np.transpose(H(theta)))+m_R))

	OMEGA=np.vstack([np.column_stack([omega,np.zeros((2,2))]),np.column_stack([np.zeros((2,2)),omega])])
	dP = -m_K.dot(m_H).dot(P)+np.dot(OMEGA,P)-np.dot(P,OMEGA)+m_Q
	return dP

def RK4thOrder(funcX, funcP, xinit, Pinit, t_range, h):
	"""
	performs numerical integration: Runge-Kutta of 4 order
	"""
	m = 4
	mm = int(len(xinit))
	#consensus coordinates
	xcons = 0
	ycons = 0
	n = int((t_range[-1] - t_range[0])/h)
	th_last = 0

	
	t = t_range[0]
	x = xinit
	P=np.zeros(shape=(nl,4,4))
	# do integration for each landmark
	for iii in range(nl):
		P[iii] = Pinit
	
	# Containers for solutions, add initial values as first elements
	tsol = np.empty(0)
	tsol = np.append(tsol, t)

	xsol = np.empty(0)
	xsol = np.append(xsol, x)
	Psol = [[] for i in range(nl)]# every virtual vehicle will have its own P history
	for i in range(nl):
		for k in range(m):
			for l in range(m):
				Psol[i]=np.append(Psol[i],Pinit[k,l])
	#print(Psol[0])
	BetaArr = []
	Beta_corrArr = []
	Beta=0
	Beta_corr=0
	# initial value for each virtual vehicle
	xx=np.zeros(shape=(nl,4))
	for ii in range(nl):
			xx[ii]=np.array([x[2*ii],x[2*ii+1],x[-2],x[-1]])


	for i in range(n): # time stes loop
		Wz=wz(t)
		omega=Omega(t,Wz)
		theta=np.zeros(shape=(nl))
		r=np.zeros(shape=(nl))
		for ii in range(nl): # landmark loop
			xx[ii][-2]=xcons #[xl,yl,xc,yc]
			xx[ii][-1]=ycons


			print('\ntime:',t)
			theta[ii] = th_measurement(t,x_0l[ii],y_0l[ii])
			r[ii] = r_measurement(t,x_0l[ii],y_0l[ii])
			
			"""
			RK4 derivatives
			"""

			k1 = feval(funcX, t, xx[ii], P[ii], theta[ii], r[ii],omega,xx[ii][-2],xx[ii][-1],xcons,ycons)
			k1P = feval(funcP, t, P[ii], theta[ii], r[ii],xx,omega,xx[ii][-2],xx[ii][-1],xcons,ycons)

			xp2 = xx[ii] + k1*(h/2)
			p2P = P[ii] + k1P*(h/2)

			k2 = feval(funcX, t+h/2, xp2, p2P, theta[ii], r[ii],omega,xp2[-2],xp2[-1],xcons,ycons)
			k2P = feval(funcP, t+h/2, p2P, theta[ii], r[ii],xp2,omega,xp2[-2],xp2[-1],xcons,10)

			xp3 = xx[ii] + k2*(h/2)
			p3P = P[ii] + k2P*(h/2)

			k3 = feval(funcX, t+h/2, xp3, p3P, theta[ii],r[ii],omega,xp3[-2],xp3[-1],xcons,ycons)
			k3P = feval(funcP, t+h/2, p3P, theta[ii], r[ii],xp3,omega,xp3[-2],xp3[-1],xcons,ycons)

			xp4 = xx[ii] + k3*h
			p4P = P[ii] + k3P*h


			k4 = feval(funcX, t+h, xp4, p4P, theta[ii], r[ii],omega,xp4[-2],xp4[-1],xcons,ycons)
			k4P = feval(funcP, t+h, p4P, theta[ii], r[ii],xp4,omega,xp4[-2],xp4[-1],xcons,ycons)

			"""
			rearrange coordinates and P-matrix
			"""
			for j in range(m):
				xx[ii][j] = xx[ii][j] + (h/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])

			for l in range(m):
				for j in range(m):
					P[ii][l,j] = P[ii][l,j] + (h/6)*(k1P[l,j] + 2*k2P[l,j] + 2*k3P[l,j] + k4P[l,j])
		

		BetaArr.append(Beta)
		Beta=Beta+Wz*h

		th_last = theta

		t = t + h
		tsol = np.append(tsol, t)
		for q in range(nl):
			xsol = np.append(xsol, [xx[q][0],xx[q][1]]) 

		#least squares consensus
		Pr1 = np.zeros(shape=(2,2))
		Pr2 = [0,0]
		Pr3 = []
		for l in range(nl):
			#print(Pr1,' ',Pr2)
			Pr1 = Pr1 + (np.linalg.inv(P[l][2:4,2:4]))
			Pr2 = Pr2 + (np.linalg.inv(P[l][2:4,2:4])).dot(xx[l][2:4])

		Pr3 = np.linalg.inv(Pr1).dot(Pr2)

		xcons,ycons = Pr3

		xsol = np.append(xsol,[xcons,ycons])
		for q in range(nl):
			for i in range(m):
				for j in range(m):
					Psol[q] = np.append(Psol[q],P[q][i,j])

	BetaArr=np.array(BetaArr)
	
	xsol=np.reshape(xsol,(int(len(xsol)/mm),mm))
	for q in range(nl):
		Psol[q]=np.reshape(Psol[q],(int(len(Psol[q])/m**2),m**2))

	return [tsol, xsol, Psol[2], BetaArr]


# -----------------------------MAIN------------------------------------#


t = np.array([0.0, t_max])
#Pinit=.00*np.ones((4,4))
Pinit=np.diag((10.,10.,1.,1.))
#Pinit=np.array([[1.,0,1.,0.],[0,1.,0,1.],[1.,0,1.,0],[0.,1.,0,1.]])

[ts, xs, Ps, beta_gl] = RK4thOrder('myFunc','myFunc1', xinit, Pinit, t, h)

# final plots in local coordinates
if glob==False:
	fig=plt.figure(2)
	for i in range (2*nl):
		ax=fig.add_subplot(int((2*nl+4)/2),2,i+1)
		ax.plot(ts,xs[:,i],'.', ms=0.5)
		ax.set_xlabel(r'$t$',fontsize=16)
		if (i%2==1):
			ax.set_ylabel(r'$x_l$',fontsize=16)
		if (i%2==0):
			ax.set_ylabel(r'$y_l$',fontsize=16)


	ax=fig.add_subplot(int((2*nl+4)/2),2,2*nl+1)
	ax.plot(ts,xs[:,-2],'.', ms=0.8)
	ax.set_ylabel(r'$x_v$',fontsize=16)
	ax.set_xlabel(r'$t$',fontsize=16)


	ax=fig.add_subplot(int((2*nl+4)/2),2,2*nl+2)
	ax.plot(ts,xs[:,-1],'.', ms=0.8)
	ax.set_ylabel(r'$y_v$',fontsize=16)
	ax.set_xlabel(r'$t$',fontsize=16)

	ax=fig.add_subplot(int((2*nl+4)/2),2,2*nl+3)
	ax.plot(xs[:,0],xs[:,1],'.', ms=0.5)
	# ax.plot(xs[:,2],xs[:,3],'.', ms=0.5)
	# ax.plot(xs[:,4],xs[:,5],'.', ms=0.5)
	ax.set_xlabel(r'$x_l$',fontsize=16)
	ax.set_ylabel(r'$y_l$',fontsize=16)
	ax=fig.add_subplot(int((2*nl+4)/2),2,2*nl+4)
	ax.plot(xs[:,-2],xs[:,-1],'.', ms=0.8)
	for i in range(nl):
		ax.plot(xs[:,2*i],xs[:,2*i+1])
	#ax.plot(xs[:,0],xs[:,1],'.', ms=0.8)
	ax.set_xlabel(r'$x_{loc}$',fontsize=16)
	ax.set_ylabel(r'$y_{loc}$',fontsize=16)
	plt.subplots_adjust(hspace=0.8,wspace=0.8)

	fig = plt.figure(5)
	ax = fig.add_subplot(111)
	ax.plot(xs[:,-2],xs[:,-1],'.', ms=0.8) 
	for i in range(nl):
		ax.plot(xs[:,2*i],xs[:,2*i+1])
	#ax.plot(xs[:,0],xs[:,1],'.', ms=0.8)
	ax.set_xlabel(r'$x_{loc}$',fontsize=16)
	ax.set_ylabel(r'$y_{loc}$',fontsize=16)
	plt.subplots_adjust(hspace=0.8,wspace=0.8)

'''global coordinates'''
if glob==True:

	"""
	covariance matrix P
	"""
	fig=plt.figure(11,dpi=200)
	for i in range (16):

		#ax.set_xlabel('t')
		if (i<4):
			first=1
		elif (i<8):
			first=2
		elif (i<12):
			first=3
		else:
			first=4
		second=(i)%4+1
		ax=fig.add_subplot(4,4,i+1)
		ax.plot(ts,Ps[:,i],'.', ms=0.5,label=r'$P_{%s%s}$'%(str(first),str(second)))
		ax.legend(frameon=False,markerscale=0.,loc=1,bbox_to_anchor=(1.1,1.05))
		#ax.set_title(r'$P_{%s%s}$'%(str(first),str(second)))
		plt.subplots_adjust(hspace=0.7,wspace=0.7)
	plt.savefig('P3.png')

	"""
	transformation into global coordinates
	"""
	for i in range(len(beta_gl)):
		for j in range(int(len(xinit)/2)):
			xs[i][2*j:2*j+2]=m_Tr_2x2(beta_gl[i]).dot(xs[i][2*j:2*j+2])
	xs=xs[0:-1]
	ts=ts[0:-1]

	"""
	plot of convergence of coordinates
	"""
	colors = ['red','orange','green']
	fig=plt.figure(22,dpi = 200)
	for i in range (2*nl):
		ax=fig.add_subplot(int((2*nl+4)/2),2,i+1)
		ax.plot(ts,xs[:,i], ms=0.5,label = r'landmark %i'%(i//2+1),color = colors[i//2])
		ax.set_xlabel(r'$t [s]$',fontsize=10)
		ax.legend(fontsize=8)
		fig.set_figheight(15)
		fig.set_figwidth(10)
		if (i%2==0):
			ax.set_ylabel(r'$x_G [m]$',fontsize=10)
		if (i%2==1):
			ax.set_ylabel(r'$y_G [m]$',fontsize=10)

	ax=fig.add_subplot(int((2*nl+4)/2),2,2*nl+1)
	ax.plot(ts,xs[:,-2], ms=0.8,label = r'vehicle')
	ax.set_ylabel(r'$x_v [m]$',fontsize=10)
	ax.set_xlabel(r'$t [s]$',fontsize=10)
	ax.legend(fontsize=8)

	ax=fig.add_subplot(int((2*nl+4)/2),2,2*nl+2)
	ax.plot(ts,xs[:,-1],'.', ms=0.8,label = r'vehice')
	ax.set_ylabel(r'$y_v [m]$',fontsize=10)
	ax.set_xlabel(r'$t [s]$',fontsize=10)
	ax.legend(fontsize=8)
	plt.subplots_adjust(hspace=0.8,wspace=0.8)
	plt.savefig("convergence.png")

	"""
	map in global coordinates
	"""
	fig = plt.figure(55,dpi=200)
	ax = fig.add_subplot(111)
	ax.plot(xs[:,-2],xs[:,-1], ms=0.8,label = r'vehicle') 
	for i in range(nl):
		ax.plot(xs[:,2*i],xs[:,2*i+1],label=r'landmark %i'%(i+1),color = colors[i])
		if i==nl-1:
			ax.plot(x_0l[i],y_0l[i],'.',ms = 8,color='black',label=r'true positions')
		else:
			ax.plot(x_0l[i],y_0l[i],'.',ms = 8,color='black')

	plt.arrow(0,0,0,0.7,width = 0.05,color = 'black')
	ax.set_xlabel(r'$x_{glob} [m]$',fontsize=16)
	ax.set_ylabel(r'$y_{glob} [m]$',fontsize=16)
	ax.legend(fontsize=10)
	plt.subplots_adjust(hspace=0.8,wspace=0.8)
	plt.grid(b=1)
	plt.savefig("map.png")

#----------------------------------------------------------------------#
	"""zoomed map plot"""
	fig,ax = plt.subplots(dpi = 200)
	ax.plot(xs[:,-2],xs[:,-1], ms=0.8,label = r'vehicle') 
	for i in range(nl):
		ax.plot(xs[:,2*i],xs[:,2*i+1],label=r'landmark %i'%(i+1),color = colors[i])
		if i==nl-1:
			ax.plot(x_0l[i],y_0l[i],'.',ms = 4,color='black',label=r'true positions')
		else:
			ax.plot(x_0l[i],y_0l[i],'.',ms = 4,color='black')
		
	ax.plot(circle(x_0l1,y_0l1,0.5)[0],circle(x_0l1,y_0l1,0.5)[1],'.',color = 'black',ms = 0.5)
	ax.plot(circle(x_0l2,y_0l2,0.5)[0],circle(x_0l2,y_0l2,0.5)[1],'.',color = 'black',ms = 0.5)
	ax.plot(circle(x_0l3,y_0l3,0.5)[0],circle(x_0l3,y_0l3,0.5)[1],'.',color = 'black',ms = 0.5)
	plt.arrow(0,0,0,0.7,width = 0.05,color = 'black')

	axins = ax.inset_axes([0.27,0.07,0.4,0.4])
	axins.plot(xs[:,-2],xs[:,-1], ms=0.8,label = r'vehicle') 
	for i in range(nl):
		axins.plot(xs[:,2*i],xs[:,2*i+1],label=r'landmark %i'%(i+1),color = colors[i])
		if i==nl-1:
			axins.plot(x_0l[i],y_0l[i],'.',ms = 4,color='black',label=r'true positions')
		else:
			axins.plot(x_0l[i],y_0l[i],'.',ms = 4,color='black')
		
	axins.plot(circle(x_0l1,y_0l1,0.5)[0],circle(x_0l1,y_0l1,0.5)[1],'.',color = 'black',ms = 0.1)
	axins.plot(circle(x_0l2,y_0l2,0.5)[0],circle(x_0l2,y_0l2,0.5)[1],'.',color = 'black',ms = 0.5)
	axins.plot(circle(x_0l3,y_0l3,0.5)[0],circle(x_0l3,y_0l3,0.5)[1],'.',color = 'black',ms = 0.8)


	x1, x2, y1, y2 = -5.6, -4.4, -4.6, -3.4
	axins.set_xlim(x1, x2)
	axins.set_ylim(y1, y2)
	ax.set_aspect(aspect = 1.)
	mark_inset(ax,axins,loc1 = 2,loc2 = 4,ec = "0.5")
	
	ax.set_xlabel(r'$x_{glob} [m]$',fontsize=16)
	ax.set_ylabel(r'$y_{glob} [m]$',fontsize=16)
	ax.legend(fontsize=10)
	plt.subplots_adjust(hspace=0.8,wspace=0.8)
	plt.savefig("map_with_circles.png")
	#plt.show()
#print("--- %s seconds ---" % (time.time() - start_time))