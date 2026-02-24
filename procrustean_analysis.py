# import necessary modules
import numpy as np


def rotation_matrix(theta):
    """2x2 rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def procrustes_2d(template, observed):
    """
    Find R, t such that observed ≈ R @ template_i + t for each point i.
    
    Parameters
    ----------
    template : (N, 2) array — reference landmark positions
    observed : (N, 2) array — measured landmark positions (may contain NaNs)
    
    Returns
    -------
    theta : float — rotation angle in radians
    t     : (2,) — translation vector
    residuals : (N, 2) — per-landmark residuals after alignment
    """
    # Handle missing points (NaN) — only use visible landmarks
    valid = ~np.any(np.isnan(observed), axis=1)
    if valid.sum() < 3:
        return np.nan, np.full(2, np.nan), np.full_like(observed, np.nan)
    
    T = template[valid]
    O = observed[valid]
    
    # Step 1: Compute centroids and center the point sets
    centroid_T = T.mean(axis=0)
    centroid_O = O.mean(axis=0)
    Tc = T - centroid_T
    Oc = O - centroid_O
    
    # Step 2: Cross-covariance matrix
    H = Tc.T @ Oc  # shape (2, 2)
    
    # Step 3: SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Step 4: Rotation (handle reflection — det should be +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, d])  # correct for reflection if needed
    R = Vt.T @ D @ U.T  # shape (2, 2)
    
    # Step 5: Translation
    t = centroid_O - R @ centroid_T
    
    # Extract angle
    theta = np.arctan2(R[1, 0], R[0, 0])
    
    # Residuals for all points (including NaN ones)
    residuals = np.full_like(observed, np.nan)
    residuals[valid] = observed[valid] - (R @ template[valid].T).T - t
    
    return theta, t, residuals


def generalized_procrustes(all_observed, n_iter=10, tol=1e-6):
    """
    Generalized Procrustes Analysis to find mean template shape.
    
    Parameters
    ----------
    all_observed : (T, N, 2) array of landmark positions, may contain NaNs
    n_iter       : maximum iterations
    tol          : convergence tolerance (max change in template)
    
    Returns
    -------
    template : (N, 2) mean shape, centered at origin
    """
    T, N, _ = all_observed.shape
    
    # ── Initialization ──────────────────────────────────────────
    # Pick frame with most valid landmarks as starting template
    n_valid = [(~np.any(np.isnan(f), axis=1)).sum() for f in all_observed]
    init_idx = np.argmax(n_valid)
    
    template = all_observed[init_idx].copy()
    # Center it
    valid = ~np.any(np.isnan(template), axis=1)
    template -= template[valid].mean(axis=0)
    
    for iteration in range(n_iter):
        # ── Step 1: Register all frames to current template ─────
        aligned = np.full_like(all_observed, np.nan)
        
        for i in range(T):
            obs = all_observed[i]
            valid_obs = ~np.any(np.isnan(obs), axis=1)
            valid_tmp = ~np.any(np.isnan(template), axis=1)
            valid = valid_obs & valid_tmp
            
            if valid.sum() < 3:
                continue
            
            theta, t, _ = procrustes_2d(template, obs)
            if np.isnan(theta):
                continue
            
            # Rotate observed points into template frame
            # i.e., undo the rotation: R^T @ (obs - t)
            R = rotation_matrix(theta)
            aligned[i, valid_obs] = (R.T @ (obs[valid_obs] - t).T).T
        
        # ── Step 2: Average aligned frames ──────────────────────
        new_template = np.nanmean(aligned, axis=0)  # (N, 2)
        
        # ── Step 3: Re-center ───────────────────────────────────
        valid_tmp = ~np.any(np.isnan(new_template), axis=1)
        new_template -= new_template[valid_tmp].mean(axis=0)
        
        # ── Step 4: Check convergence ───────────────────────────
        valid_both = valid_tmp & ~np.any(np.isnan(template), axis=1)
        change = np.max(np.abs(new_template[valid_both] - template[valid_both]))
        
        template = new_template
        
        if change < tol:
            print(f"Converged after {iteration + 1} iterations (change={change:.2e})")
            break
    
    return template

parts = [
    'neck', 'tail_joint', 'tail_end', 'antenna_left', 'antenna_right',
    'head_left', 'head_right', 'shoulder_left', 'shoulder_right'
]
body_parts = [
    'neck', 'tail_joint', 'tail_end', 'shoulder_left', 'shoulder_right'
]
head_parts = [
    'antenna_left', 'antenna_right', 'head_left', 'head_right'
]

# define a function that does all of this succinctly, starting from raw data
def get_angles_procrustes(data, parts=body_parts):
    """
    Get the rotation angles of body parts using Procrustes analysis.

    This function needs the following other functions:
    - `generalized_procrustes`
    - `procrustes_2d`
    """
    body_coords = []
    for part in parts:
        coords = data[part][['x', 'y']].values
        body_coords.append(coords)
    body_coords = np.array(body_coords).transpose((1, 0, 2))  # shape (T, N_parts, 2)
    # measure the mean jitter, per part, over time
    mean_jitter = np.linalg.norm(np.mean(np.diff(body_coords, axis=0), axis=0), axis=1)
    # now get the weighted center of mass
    weighted_com = np.average(body_coords, axis=1, weights=1/mean_jitter)
    # subtract from body_coords
    body_coords -= weighted_com[:, np.newaxis, :]
    # use generalized procrustes analysis to find the mean shape
    template = generalized_procrustes(body_coords)
    # align the template so that the vector from the neck to the tail base is along the x-axis
    neck_to_tail = template[0, :] - template[1, :]
    neck_to_tail /= np.linalg.norm(neck_to_tail)
    perp_vector = np.array([-neck_to_tail[1], neck_to_tail[0]])
    rotation_matrix = np.column_stack((neck_to_tail, perp_vector))
    template = (np.linalg.inv(rotation_matrix) @ template.T).T
    # now, let's use this template and procrustes_2d to find the optimal rotation for all frames
    thetas = []
    for coords in body_coords:
        # register the coordinates to the template
        theta, t, resids = procrustes_2d(template, coords)
        thetas.append(theta)
    thetas = np.array(thetas)
    # return the centers of mass and the rotation angles
    return weighted_com, thetas