import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import h5py
import sys
import numpy as np

myFile = h5py.File(sys.argv[1], 'r')

data = myFile['prediction'][()]
labels = myFile['target'][()]
num_labels = np.max(labels)

if data.shape[1] != 2:
  data = PCA(n_components=2).fit_transform(data)

f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(1,num_labels):
    plt.plot(data[labels==i,0].flatten(), data[labels==i,1].flatten(), '.')#, c=c[i-1])
    
plt.legend([i for i in range(num_labels) ])
plt.grid()
plt.savefig("visualize.png")
#plt.show()

