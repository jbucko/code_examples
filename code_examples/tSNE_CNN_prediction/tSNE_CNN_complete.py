import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import numpy as np
#import folder_label_from_name_separated as read_data
import sys,os
from sklearn.decomposition import PCA,KernelPCA 
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import h5py, pickle

n_components = 2 # no_tSNE components
cluster = False # if true - running on cluster, if not - on local
load_from_folder = 1
load_from_hdf5 = 1
channels = ['red','green','blue']
"""
0: red
1: green
2: blue
"""
comp = 0

print('\n----------------tSNE code launched------------------\n')

"""different options for data loading"""

###########---- loading from stored kPCA data ----################
"""load hdf5 file"""
if load_from_hdf5:
	file = h5py.File('kPCA_90var_data.hdf5','r')
	dsets = list(file.keys())
	print('\n.hdf5 contains following datasets:',dsets,'\n')

	data_kpca = file['kPCA_90_data_'+channels[comp]][:]
	#print('first_data:',data_kpca[0])
	labels = file['kPCA_90_labels_'+channels[comp]][:]
	scaler_kpca = file['scaler_'+channels[comp]][:]
	print('data, labels, scaler:',data_kpca.shape,labels.shape,scaler_kpca.shape)
	print('data from hdf5 loaded...\n')
	#sys.exit()


###########---- loading from directory under PATH_DATA----###########
if load_from_folder:
	#local
	if not cluster:
		PATH_DATA = "/home/jozef/Desktop/master_thesis/GitLab/cm-bilayerboundstates/dim_reduction/tSNE_kernel_combined/to_predict/data/"

	#cluster
	if cluster:
		PATH_DATA = "/cluster/home/jbucko/master_thesis/unperturbed_dataset/"

	from PIL import Image
	import glob
	image_list = []
	mt_list = []
	data = []
	
	for filename in glob.glob(PATH_DATA +'*.jpg'): #assuming gif
		im=Image.open(filename)
		im1 = im.copy()	
		im1 = np.array(im1)
		mt_list.append([np.nan,np.nan])
		data.append(im1[:,:,comp])
		im.close()

	print("data for prediction loaded...\n")
	channel = np.zeros(shape = (len(data),360*360))
	for i in range(len(data)):
		channel[i] = np.reshape(data[i],(1,360*360))


	#scale from 0 to 1
	channel/=255
	#print(channel.shape,len(data),scaler_kpca.shape)
	channel = (channel-scaler_kpca[0])/np.sqrt(scaler_kpca[1])
	#print(scaler_kpca[0],scaler_kpca[1])
	print('data for prediction scaled...\n')
	#print('scaled data',channel[0])
	model_trained = np.genfromtxt('weights_'+channels[comp]+'.csv')
	print('trained PCA loaded, shape of matrix:',model_trained.shape,'\n')
	#sys.exit()
	channel = channel.dot(model_trained.T)
	print('transform finished, shape of resulting prediction data:',channel.shape,'\n')


#######--concatenate hdf5 and folder data types if any--##########
if load_from_hdf5 and load_from_folder:
	channel = np.vstack((data_kpca,channel))
	mt_list = np.vstack((labels,mt_list))
if load_from_hdf5 and not load_from_folder:
	channel = data_kpca.copy()
	mt_list = labels.copy()
print('merging finished, data and label dimensions:',channel.shape,mt_list.shape,'\n')

###########---tSNE---#############
print('starting tSNE...')
tsne = TSNE(n_components = n_components)
channel = tsne.fit_transform(channel)
#print(channel)
print('tSNE finished!\n')
np.savetxt('tsne_two_principal_'+channels[comp]+'.txt',channel)
np.savetxt('tsne_labels_'+channels[comp]+'.txt',mt_list)
print('number of iterations to converge: ',tsne.n_iter_)
print('KL_divergence value: ',tsne.kl_divergence_)

###########---classification---###########
print('\nfitting KNN...\n')
if load_from_folder:
	from_folder = channel[-len(data):]
	from_kpca = channel[:-len(data)]

	classes = [[0,1] ,[0,-1],[1,1] ,[1,-1] ,[2,1] ,[2,-1],[-1,1] ,[-1,-1] ,[-2,1], [-2,-1]]
	cl_attr = np.zeros((len(data),10))
	from sklearn.neighbors import NearestNeighbors
	knn = NearestNeighbors(n_neighbors = 11)
	knn.fit(channel)
	for i in range(len(data)):
		nn = knn.kneighbors([channel[-len(data)+i]])[-1][0]
		#print(nn)
		for l in nn:
			if l<len(from_kpca):
				#print(l,mt_list[l])
				idx = [j for j in range(len(classes)) if np.array_equal(classes[j],mt_list[l])][0]
				cl_attr[i,idx] +=1
	# print(cl_attr)

	pred_classes = np.argmax(cl_attr,axis = 1)
	i=0
	for c in pred_classes:
		print('predicted class:',classes[c],'confidence: {:.2f}'.format(cl_attr[i,c]/np.sum(cl_attr[i,:])))
		i+=1

# classes = [[0,1] ,[0,-1],[1,1] ,[1,-1] ,[2,1] ,[2,-1],[-1,1] ,[-1,-1] ,[-2,1], [-2,-1]]
# pred_classes = [0,0]
###############---predict U,V---############
print('\npredicting U and V...\n')
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet,self).__init__()
		self.layer1 = nn.Sequential(
				nn.Conv2d(3,16,kernel_size = (5,5),stride = 1,padding = (2,2)),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = (5,5),stride = (5,5))
				)
		self.layer2 = nn.Sequential(
				nn.Conv2d(16,32,kernel_size = (3,3),stride = 1,padding = (1,1)),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = (3,3),stride = (3,3))
				)
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(360*360//3//3//5//5*32,250)
		self.fc2 = nn.Linear(250,50)
		self.fc3 = nn.Linear(50,2)
		self.relu =  torch.nn.ReLU(inplace=True)
		self.softmax = torch.nn.Softmax()

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0),-1)
		out = self.drop_out(out)
		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		#out = self.softmax(out)
		return out

# local
PATH_DATA = "./to_predict/"
# cluster
#PATH_DATA = "./../separated_dataset/"+which_data+'/'
STORE_MODEL_PATH = "./trained_CNNs/"
data = torchvision.datasets.ImageFolder_label_from_name_separated(root = PATH_DATA,transform = transforms.ToTensor())


j=0
for c in pred_classes:
	m,tau = classes[c]
	#print(m,tau)

	if m<0:
		str_m = 'm'+str(abs(m))
	else:
		str_m = str(m)
	if tau<0:
		str_tau = 'm'+str(abs(tau))
	else:
		str_tau = str(tau)
	which_data = 'm_'+str_m+'_tau_'+str_tau
	#print(which_data)


	data_loader = DataLoader(dataset = data, batch_size = 1)


	model = ConvNet()
	model.load_state_dict(torch.load(STORE_MODEL_PATH+'supervised_CNN_'+which_data+'.ckpt'))
	model.eval()

	# loss and optimizer
	criterion = nn.MSELoss()

	for i,(images,labels) in enumerate(data_loader):
		if i==j:
			outputs = model(images)
			loss = criterion(outputs,labels.float())

			labels = labels.float()
			print('predicted item no.',j)
			print('predicted U and V: {:.4f} {:.4f}'.format(outputs.data[0][0].item(),outputs.data[0][1].item()))
			print('true U and V: {:.4f} {:.4f} \n'.format(labels.data[0][0].item(),labels.data[0][1].item()))
			
	j +=1