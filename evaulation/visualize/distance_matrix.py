from sklearn.metrics.pairwise import pairwise_distances
import argparse,os
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import h5py
import numpy as np


parser = argparse.ArgumentParser(description='Calculate distance matrix for given data')
parser.add_argument('data', help='path to cPickle file with saved numpy array. Each row is a sample')

def load_H5PY(name_file):
    '''Read data from H5PY format'''
    with h5py.File(name_file,  "r") as f:
      return f['clusters'].value

if __name__ == '__main__':
  args        = parser.parse_args()
  data        = load_H5PY(args.data)
  distance_matrix = pairwise_distances(data)
  idx_close = np.argsort(distance_matrix[0])
  print distance_matrix[0][idx_close[1]]
  print distance_matrix[0]
  print "Matrix", data.shape
  ax = plt.subplot(1, 1, 1)
  plt.imshow(np.abs(distance_matrix))#, interpolation='nearest')
  ax.set_aspect('equal')
  plt.colorbar(orientation='vertical')
  plt.savefig("clusters.png")

  
  
