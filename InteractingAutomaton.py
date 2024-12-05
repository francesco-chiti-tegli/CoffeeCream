# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:47:04 2024

@author: Francesco Chiti Tegli
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

plt.rcParams['figure.dpi'] = 1000
vmin, vmax = 0, 1

# Initializing grid, cream above (1) and coffee below (0)
# The size n of the automaton must be a multiple of 10
n = 100
array = np.zeros((n, n))
array[:n//2, :] = 1

plt.figure(1)
plt.imshow(array, cmap='gray', vmin=vmin, vmax=vmax)
plt.xticks([])
plt.yticks([])
plt.show()

# The grain size (g) should be quite larger than
print('The grain size (g) should be quite larger than ' + str(np.sqrt(3*np.log(2*n**2))))
grain_size = 10
print('Grain size chosen: ' + str(grain_size))

#%% (1) Non-interacting & Interactig Stochastic step, Coarse Graining and Compression measure functions

def I_Step(matrix):
    n = matrix.shape[0]
    indx = np.argwhere(matrix == 1)
    indx = np.random.permutation(indx)
    for i in np.arange(indx.shape[0]):
        displ = random.choice([[-1,0],[1,0],[0,1],[0,-1]])

        if ((indx[i][0]+displ[0] < n) & (indx[i][1]+displ[1] < n) & (indx[i][0]+displ[0] >= 0) & (indx[i][1]+displ[1] >= 0)):
            if (matrix[indx[i][0]+displ[0],indx[i][1]+displ[1]] != matrix[indx[i][0],indx[i][1]]):
                temp = matrix[indx[i][0]+displ[0],indx[i][1]+displ[1]]
                matrix[indx[i][0]+displ[0],indx[i][1]+displ[1]] = matrix[indx[i][0],indx[i][1]]
                matrix[indx[i][0],indx[i][1]] = temp
    return matrix

def I_Step_PBC(matrix):
    n = matrix.shape[0]
    indx = np.argwhere(matrix == 1)
    indx = np.random.permutation(indx)
    for i in np.arange(indx.shape[0]):
        displ = random.choice([[-1, 0], [1, 0], [0, 1], [0, -1]])
        
        new_x = (indx[i][0] + displ[0]) % n
        new_y = (indx[i][1] + displ[1]) % n

        if (matrix[new_x, new_y] != matrix[indx[i][0], indx[i][1]]):
            temp = matrix[new_x, new_y]
            matrix[new_x, new_y] = matrix[indx[i][0], indx[i][1]]
            matrix[indx[i][0], indx[i][1]] = temp
    return matrix

def lempel_ziv_complexity(matrix):
    s = matrix.flatten()
    s = ''.join(map(str, s))
    n = len(s)
    complexity = 0
    i = 0
    
    while i < n:
        complexity += 1
        j = i + 1
        while j < n and s[i:j+1] in s[:i]:
            j += 1
        i = j
    return complexity

def normalize_array(arr):
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def spectral(matrix):
    cond_number = np.linalg.cond(matrix)
    return cond_number

def CoarseGraining(matrix, g):
    output = np.zeros_like(matrix)
    for i in np.arange(matrix.shape[0]):
        for j in np.arange(matrix.shape[1]):

            i_min = max(i - g // 2, 0)
            i_max = min(i + g // 2 + 1, matrix.shape[0])
            j_min = max(j - g // 2, 0)
            j_max = min(j + g // 2 + 1, matrix.shape[1])

            region = matrix[i_min:i_max, j_min:j_max]

            output[i, j] = np.sum(region) / region.size
    return output

def CoarseGraining_PBC(matrix, g):
    output = np.zeros_like(matrix)
    rows, cols = matrix.shape
    
    for i in range(rows):
        for j in range(cols):
            i_indices = [(i + x - g // 2) % rows for x in range(g)]
            j_indices = [(j + y - g // 2) % cols for y in range(g)]
            
            region = matrix[np.ix_(i_indices, j_indices)]
            
            output[i, j] = np.sum(region) / region.size
    return output

def THREE_Buckets(matrix):
    mask0 = (matrix < 1/3)
    mask1 = (matrix > 2/3)
    mask05 = (matrix >= 1/3).astype('uint8') * (matrix <= 2/3).astype('uint8')
    mask05 = mask05.astype('bool')
    
    matrix[mask0] = 0
    matrix[mask1] = 1
    matrix[mask05] = 0.5
    return matrix


#%% (3) Automaton run: Entropy,Complexity VS Time

plt.ion()

time = 10000
sampled_times = []

path_compl = 'C:/Users/fchit/OneDrive/Documenti/Università/LabComp/CoffeeMilkSnapshots/Complexity'
path_entro = 'C:/Users/fchit/OneDrive/Documenti/Università/LabComp/CoffeeMilkSnapshots/Entropy'

entropy_LZ = []
complexity_LZ= []
spectral_complexity = []

for p in np.arange(time):
    array = I_Step_PBC(array)
    coarse_grained = THREE_Buckets(CoarseGraining_PBC(array, g=grain_size))
    
    if np.mod(p,50)==0:
      sampled_times.append(p)
      '''
      fig, (ax1, ax2) = plt.subplots(1, 2)
      ax1.imshow(array, cmap='gray', vmin=vmin, vmax=vmax)
      ax2.imshow(coarse_grained, cmap='gray', vmin=vmin, vmax=vmax)
      plt.grid(True)
      plt.pause(1)
      plt.clf()
      plt.close()
      '''
      image = Image.fromarray((coarse_grained * 255).astype(np.uint8))
      image.save(os.path.join(path_compl, 'CoffeMilk_CG_' + str(p) + '.jpeg'), "JPEG", quality=100)
      
      
      image = Image.fromarray((array * 255).astype(np.uint8))
      image.save(os.path.join(path_entro, 'CoffeMilk_FG_' + str(p) + '.jpeg'), "JPEG", quality=100)
      
      complexity_LZ.append(lempel_ziv_complexity(coarse_grained))
      entropy_LZ.append(lempel_ziv_complexity(array))      
      spectral_complexity.append(spectral(array))
      
    if (p == 0):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('t = 1')
        fig.tight_layout()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.imshow(array, cmap='gray', vmin=vmin, vmax=vmax)
        ax2.imshow(coarse_grained, cmap='gray', vmin=vmin, vmax=vmax)
        plt.pause(1)
        plt.close()
    if (p == 1099):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('t = 1100')
        fig.tight_layout()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.imshow(array, cmap='gray', vmin=vmin, vmax=vmax)
        ax2.imshow(coarse_grained, cmap='gray', vmin=vmin, vmax=vmax)
        plt.pause(1)
        plt.close()
    if (p == 2499):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('t = 2500')
        fig.tight_layout()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.imshow(array, cmap='gray', vmin=vmin, vmax=vmax)
        ax2.imshow(coarse_grained, cmap='gray', vmin=vmin, vmax=vmax)
        plt.pause(1)
        plt.close()
sampled_times = np.array(sampled_times)
plt.ioff()
plt.show()

complexity_LZ = np.array(complexity_LZ)
entropy_LZ = np.array(entropy_LZ)
spectral_complexity = np.array(spectral_complexity)

'''
czip_path_comp = 'C:/Users/fchit/OneDrive/Documenti/Università/LabComp/CoffeeMilkSnapshots/ComplexityCompressed'
with zipfile.ZipFile(czip_path_comp, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for filename in os.listdir(path_compl):
        file_path = os.path.join(path_compl, filename)
        zipf.write(file_path, arcname=filename)
with zipfile.ZipFile(czip_path_comp, 'r') as zipf:
    complexity_czip = np.array([file_info.file_size for file_info in zipf.infolist()])

czip_path_entro = 'C:/Users/fchit/OneDrive/Documenti/Università/LabComp/CoffeeMilkSnapshots/EntropyCompressed'
with zipfile.ZipFile(czip_path_entro, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for filename in os.listdir(path_entro):
        file_path = os.path.join(path_entro, filename)
        zipf.write(file_path, arcname=filename)
with zipfile.ZipFile(czip_path_entro, 'r') as zipf:
    entropy_czip = np.array([file_info.file_size for file_info in zipf.infolist()])
'''
complexity_jpeg = []
entropy_jpeg = []
for i in np.arange(sampled_times.shape[0]):
    complexity_jpeg.append(os.path.getsize(os.path.join(path_compl, 'CoffeMilk_CG_' + str(sampled_times[i]) + '.jpeg')))
    entropy_jpeg.append(os.path.getsize(os.path.join(path_entro, 'CoffeMilk_FG_' + str(sampled_times[i]) + '.jpeg')))
complexity_jpeg = np.array(complexity_jpeg)
entropy_jpeg = np.array(entropy_jpeg)

'''
plt.figure(3)
plt.suptitle('Non-interacting, size=' + str(n) + ', .czip images')
plt.plot(sampled_times,complexity_czip,label='Complexity')
plt.plot(sampled_times,entropy_czip,label='Entropy')
plt.legend()
plt.show()
'''

plt.figure(4)
plt.suptitle('Non-interacting, size=' + str(n) + ', JPEG images')
plt.plot(sampled_times,complexity_jpeg,label='Complexity')
plt.plot(sampled_times,entropy_jpeg,label='Entropy')
plt.legend()
plt.show()

plt.figure(5)
plt.suptitle('Non-interacting, size=' + str(n) + ', Lempel-Ziv')
plt.plot(sampled_times,complexity_LZ,label='Complexity')
plt.plot(sampled_times,entropy_LZ,label='Entropy')
plt.legend()
plt.show()

plt.figure(6)
plt.suptitle('Non-interacting, size=' + str(n) + ', Complexity')
plt.plot(sampled_times,normalize_array(complexity_LZ),label='LZ')
plt.plot(sampled_times,normalize_array(complexity_jpeg),label='JPEG')
plt.legend()
plt.show()

plt.figure(7)
plt.suptitle('Non-interacting, size=' + str(n) + ', Entropy')
plt.plot(sampled_times,normalize_array(entropy_LZ),label='LZ')
plt.plot(sampled_times,normalize_array(entropy_jpeg),label='JPEG')
plt.legend()
plt.show()

for filename in os.listdir(path_compl):
    os.remove(os.path.join(path_compl, filename))
for filename in os.listdir(path_entro):
    os.remove(os.path.join(path_entro, filename))
    
# Next steps:
# Try .gzip
# Periodic bounday conditions (DONE)
# Max Complexity VS box size
# Max entropy VS box size
# Max Complexity VS time
# Experimental curve
# Introducing a different formula for complexity (DONE - Lempel-Ziv VS JPEG)