# -*- coding: utf-8 -*-
"""neuroGramData.py

"""

import requests
import os
import zipfile
import tempfile
import shutil # For moving files/directories
# pip install neurom
import neurom as nm
from neurom import view
import matplotlib.pyplot as plt


neuro_vlm_folder = 'SWCs/'

print(f"Contents of '{neuro_vlm_folder}':")
if os.path.exists(neuro_vlm_folder):
    for item in os.listdir(neuro_vlm_folder):
        print(f"- {item}")
else:
    print(f"Error: Folder '{neuro_vlm_folder}' not found in Google Drive.")

swc_files = []
for root, dirs, files in os.walk(neuro_vlm_folder):
    for file in files:
        if file.endswith('.swc'):
            swc_files.append(os.path.join(root, file))

print(f"Found {len(swc_files)} SWC files:")
for swc_file in swc_files[:10]: # Print first 10 for brevity
    print(f"- {swc_file}")

if swc_files:
    example_swc_file = swc_files[0]
    print(f"\nUsing the first found SWC file for demonstration: {example_swc_file}")
else:
    print("No SWC files found. Please ensure the downloaded folders contain .swc files.")




"""Let's load the first SWC file using `neurom` and extract some basic information. `neurom` treats SWC files as `Neuron` objects, allowing easy access to its morphological properties."""

if 'example_swc_file' in locals() and example_swc_file:
    try:
        # Load the morphology
        morph = nm.load_morphology(example_swc_file)
        print(f"Successfully loaded SWC file: {example_swc_file}")

        # Print basic properties
        print(f"\nNumber of neurites: {len(morph.neurites)}")

        # Get total length statistic
        total_length = nm.get('total_length', morph)
        print(f"Total length of morphology: {total_length:.2f} units")

        # Improved Visualization
        fig, ax = plt.subplots(figsize=(8, 8))

        # plot_morph uses the XY plane by default; we'll force it to show the full extent
        view.plot_morph(morph, ax=ax)

        ax.set_title(f"Visualization of {os.path.basename(example_swc_file)}")
        ax.set_xlabel('X (units)')
        ax.set_ylabel('Y (units)')
        ax.axis('equal') # Ensure 1:1 aspect ratio
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    except Exception as e:
        print(f"Error loading or processing {example_swc_file}: {e}")
else:
    print("Cannot load example SWC file as no files were found or selected.")

"""### Voxelizing SWC to 3D Volumes
To compare vector data with raw microscopy, we need to convert the SWC into a voxel grid (rasterization). The following code creates a small 3D volume around the neuron's bounding box.
"""

import numpy as np

def get_neuron_mask(morph, voxel_size=10.0):
    """Generates a binary 3D mask from morphology nodes."""
    # Get all point coordinates (X, Y, Z)
    points = morph.points[:, :3]

    # Calculate bounding box
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Determine grid dimensions
    shape = ((max_coords - min_coords) / voxel_size).astype(int) + 1
    mask = np.zeros(shape, dtype=np.uint8)

    # Map points to grid indices
    indices = ((points - min_coords) / voxel_size).astype(int)

    # Fill mask (simple point-based voxelization)
    for idx in indices:
        mask[idx[0], idx[1], idx[2]] = 1

    return mask, min_coords

# Generate a mask for the example morphology
voxel_res = 20.0 # Adjust resolution as needed (units per voxel)
mask, origin = get_neuron_mask(morph, voxel_size=voxel_res)

print(f"Created 3D mask with shape: {mask.shape}")
print(f"Mask origin in CCFv3 space: {origin}")
print(f"Number of occupied voxels: {np.sum(mask)}")

"""### Visualizing the Voxelized 3D Mask
Since the mask is a 3D volume, we can visualize it by projecting the maximum values along each axis (X, Y, and Z).
"""

import matplotlib.pyplot as plt

# Calculate Maximum Intensity Projections (MIPs)
mip_z = np.max(mask, axis=2) # Projection onto XY plane
mip_y = np.max(mask, axis=1) # Projection onto XZ plane
mip_x = np.max(mask, axis=0) # Projection onto YZ plane

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mip_z.T, origin='lower', cmap='gray_r')
axes[0].set_title('XY Projection (MIP Z)')
axes[0].set_xlabel('X index')
axes[0].set_ylabel('Y index')

axes[1].imshow(mip_y.T, origin='lower', cmap='gray_r')
axes[1].set_title('XZ Projection (MIP Y)')
axes[1].set_xlabel('X index')
axes[1].set_ylabel('Z index')

axes[2].imshow(mip_x.T, origin='lower', cmap='gray_r')
axes[2].set_title('YZ Projection (MIP X)')
axes[2].set_xlabel('Y index')
axes[2].set_ylabel('Z index')

plt.tight_layout()
plt.show()

"""This code snippet first defines the URL of the zip file and the desired destination folder within your Google Drive. It then downloads the file using the `requests` library and saves it to the specified path in your Drive."""