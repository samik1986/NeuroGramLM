"""
Author: Samik Banerjee
Date: 2026-08-30
Usage: Functions to compute invariant geometric and topological features from neuronal structures.
"""

import numpy as np

def compute_tortuosity(points, window_size=5):
    """
    Computes local tortuosity for a sequence of points.
    Tortuosity = L_arc / L_chord
    """
    if len(points) < 2:
        return np.ones(len(points))
    
    tortuosity = np.ones(len(points))
    half_win = window_size // 2
    
    for i in range(len(points)):
        start_idx = max(0, i - half_win)
        end_idx = min(len(points) - 1, i + half_win)
        
        if end_idx - start_idx < 1:
            continue
            
        segment = points[start_idx:end_idx+1]
        
        # Arc length: sum of distances between consecutive points
        diffs = np.diff(segment, axis=0)
        l_arc = np.sum(np.linalg.norm(diffs, axis=1))
        
        # Chord length: distance between start and end of segment
        l_chord = np.linalg.norm(segment[-1] - segment[0])
        
        if l_chord > 1e-6:
            tortuosity[i] = l_arc / l_chord
            
    return tortuosity

def compute_curvature_energy(points, sigma=2.0):
    """
    Computes local curvature energy using finite differences.
    sigma is used if we apply smoothing (not implemented in this raw version for brevity, 
    but can be added via scipy.ndimage.gaussian_filter1d).
    """
    if len(points) < 3:
        return np.zeros(len(points))
        
    # First derivative (velocity)
    v = np.gradient(points, axis=0)
    # Second derivative (acceleration)
    a = np.gradient(v, axis=0)
    
    v_norm = np.linalg.norm(v, axis=1)
    
    curvature = np.zeros(len(points))
    for i in range(len(points)):
        if v_norm[i] > 1e-6:
            cross_prod = np.cross(v[i], a[i])
            curvature[i] = np.linalg.norm(cross_prod) / (v_norm[i]**3)
            
    # Energy is kappa^2
    energy = curvature ** 2
    return energy

def compute_local_inertia_tensor(points, query_point, radius=10.0):
    """
    Computes the inertia tensor for points within a radius of the query_point.
    Returns the sorted eigenvalues (lambda_1 >= lambda_2 >= lambda_3).
    """
    # Find neighbors
    dists = np.linalg.norm(points - query_point, axis=1)
    neighbors = points[dists <= radius]
    
    if len(neighbors) < 3:
        return np.zeros(3)
        
    # Center the neighbors relative to query_point
    centered = neighbors - query_point
    
    # Compute tensor
    # I = sum( |r|^2 * Identity - r * r^T )
    tensor = np.zeros((3, 3))
    I3 = np.eye(3)
    
    for r in centered:
        r_sq = np.dot(r, r)
        tensor += r_sq * I3 - np.outer(r, r)
        
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(tensor)
    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    return eigenvalues

def compute_branching_angle(v_in, v_out):
    """
    Computes the angle between incoming and outgoing direction vectors.
    """
    n_in = np.linalg.norm(v_in)
    n_out = np.linalg.norm(v_out)
    
    if n_in < 1e-6 or n_out < 1e-6:
        return 0.0
        
    cos_theta = np.dot(v_in, v_out) / (n_in * n_out)
    # Clip for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
