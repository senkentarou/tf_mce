# coding: utf-8

# import packages
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

# output_folder = os.path.abspath(os.path.join(yourpath, os.pardir))
output_folder = '.'
output_name = 'GMM'

def gen_samples(sample, mean, covar, class_label):

    mean_shape = np.unique(np.array(mean).shape)
    covar_shape = np.unique(np.array(covar).shape)
    if not (len(mean_shape) == 1 and len(covar_shape) == 1 and mean_shape[0] == covar_shape[0]):
        print("Error; input lists have different shape each other.")
        print("mean: %s (shape: %s)" % (mean, mean_shape))
        print("covariance: %s (shape: %s)" % (covar, covar_shape))
        exit(1)

    # 3 = 2 features + 1 category label
    dataset =  np.ndarray(shape=(sample, mean_shape[0]+1))
    for i in range(sample):
        sample = np.random.multivariate_normal(mean, covar, 1)[0]
        dataset[i] = np.append(sample, [class_label])  # add category label
    return dataset

all_sample_num = 2000
cls_lbl = 0
mean_vec = [-2, 1]
covar_vec = [[1, 0.6],
             [0.6, 2]]
D1 = gen_samples(all_sample_num, mean_vec, covar_vec, cls_lbl)

cls_lbl = 1
mean_vec = [3, 4]
covar_vec = [[5, -1],
             [-1, 1]]
D2 = gen_samples(all_sample_num, mean_vec, covar_vec, cls_lbl)

with open( os.path.join(output_folder,output_name + '.dat'), 'wt') as f:
    writer = csv.writer(f,delimiter=' ')
    writer.writerows(D1)
    writer.writerows(D2)

# see the scatter plot
plt.plot(D1[:,0],D1[:,1],'.',color='blue')
plt.plot(D2[:,0],D2[:,1],'.',color='red')
plt.savefig(os.path.join(output_folder, output_name + '.png'))

#plt.show()
plt.pause(5)
#"""
