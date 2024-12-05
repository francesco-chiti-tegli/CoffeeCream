# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:58:34 2024

@author: Francesco Chiti Tegli
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

plt.rcParams['figure.dpi'] = 1000

def LoadVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)

video = LoadVideo(video_path="C:/Users/fchit/Downloads/Video_Caffe2.mp4")

#%% (1) Cropping the cup region

def crop_frames(frames):
    return frames[:,720:1150,280:750,:]

#For Video_Acqua is [:, 1025:1500, 300:750, :]
#For Video_Caffe1 is [210:470, 1010:1200, 340:780, :]
#For Video_Caffe2 is [:,720:1150,280:750,:]


def NormalizeVideo(array):
    if array.dtype != np.uint8:
        raise ValueError("L'array deve essere di tipo uint8.")
    return array.astype(np.float64) / 255.0

video_cropped = crop_frames(video)

w = np.random.randint(video_cropped.shape[0])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.tight_layout()
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.imshow(video[w])
ax2.imshow(video_cropped[w])
plt.show()
del fig, ax1, ax2, w

#%% (2) Visualize the cropped video - %matplotlib qt

def VisualizeVideo(frames, frame_rate=10):
    plt.ion()
    fig, ax = plt.subplots()
    
    img_display = ax.imshow(frames[0], cmap='gray' if len(frames[0].shape) == 2 else None)
    ax.axis('off')

    delay = 1 / frame_rate

    for frame in frames:
        img_display.set_data(frame)
        plt.pause(delay)
    plt.ioff()
    plt.show()
    
def DifferencesVideo(frames, smooth=40):
    output = [frames[0]]
    for j in range(1, frames.shape[0]):
        output.append(np.abs(frames[j] - frames[j - 1]))
    output = np.array(output)
    output = np.mean(output, axis=-1)
    for j in range(1, frames.shape[0]):
        output[j] = cv2.blur(output[j], (smooth,smooth))
    return output

VisualizeVideo(video_cropped)

#%% (3) Measure complexity evolution - %matplotlib inline

path_compl = 'C:/Users/fchit/OneDrive/Documenti/Università/LabComp/CoffeeMilkSnapshots/Complexity'

for p in np.arange(video_cropped.shape[0]):
    image = Image.fromarray(video_cropped[p].astype(np.uint8))
    image.save(os.path.join(path_compl, 'CoffeMilk_CG_' + str(p) + '.jpeg'), "JPEG", quality=50)

complexity_experimental = []
for i in np.arange(video_cropped.shape[0]):
    complexity_experimental.append(os.path.getsize(os.path.join(path_compl, 'CoffeMilk_CG_' + str(i) + '.jpeg')))
complexity_experimental = np.array(complexity_experimental)

for filename in os.listdir(path_compl):
    os.remove(os.path.join(path_compl, filename))

plt.rcParams['figure.dpi'] = 1000

plt.figure(2)
#plt.suptitle('Experimental, JPEG images')
plt.plot(np.arange(video_cropped.shape[0]),complexity_experimental,label='Complexity')
plt.legend()
plt.show()

print('The maximum is in: ' + str(np.argmax(complexity_experimental)))

#%% (4) Three snapshots

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.tight_layout(pad=0.005)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax1.imshow(video_cropped[100])
ax2.imshow(video_cropped[248])
ax3.imshow(video_cropped[600])
ax1.set_title("t = 100")
ax2.set_title("t = 248")
ax3.set_title("t = 600")
plt.show()

#%% (5) Confront them

from scipy.optimize import minimize

def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def cost_function(params, m, n):
    sx, sy = params
    n_scaled_x = sx * np.linspace(0, 1, len(n))
    n_scaled_y = sy * n
    n_interp = np.interp(np.linspace(0, 1, len(m)), n_scaled_x, n_scaled_y)
    
    return np.sum((m - n_interp) ** 2)

def fit_and_plot(m, n):
    m_cropped = m[200:325]
    m_norm = normalize_array(m_cropped)
    n_norm = normalize_array(n)
    

    initial_guess = [1, 1]
    
    result = minimize(cost_function, initial_guess, args=(m_norm, n_norm), bounds=[(0, 5), (0, 5)])
    sx_opt, sy_opt = result.x
    
    n_scaled_x = sx_opt * np.linspace(0, 1, len(n))
    n_scaled_y = sy_opt * n_norm
    n_interp = np.interp(np.linspace(0, 1, len(m_norm)), n_scaled_x, n_scaled_y)
    
    chi_square = np.sum((m_norm - n_interp) ** 2)

    dof = len(m_norm) - 2
    chi_square_reduced = chi_square / dof if dof > 0 else np.nan
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, 1, len(m_norm)), m_norm, label="Experimental Peak (Cropped)", color="blue")
    plt.plot(np.linspace(0, 1, len(n_norm)), n_norm, label="PBC Automaton Peak", color="orange", linestyle="--")
    plt.plot(np.linspace(0, 1, len(m_norm)), n_interp, label="Fitted Experimental Peak", color="green")
    plt.legend()
    plt.xlabel("Normalized Time")
    plt.ylabel("Normalized Complexity")
    plt.grid()
    plt.show()
    
    return sx_opt, sy_opt, chi_square, chi_square_reduced