# import necessary modules
import pandas as pd
import numpy as np
from scipy.optimize import minimize


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
    'neck', 'tail_joint', 'shoulder_left', 'shoulder_right'
]
head_parts = [
    'neck', 'antenna_left', 'antenna_right', 'head_left', 'head_right'
]

# define a function that does all of this succinctly, starting from raw data
def get_angles_procrustes(data, parts=body_parts, top_anchor='neck', bottom_anchor='tail_joint'):
    """
    Get the rotation angles of body parts using Procrustes analysis.

    This function needs the following other functions:
    - `generalized_procrustes`
    - `procrustes_2d`

    Parameters
    ----------
    data : dict
        Dictionary containing the coordinates of body parts.
    parts : list of str, optional
        List of body parts to consider. Default is `body_parts`.
    top_anchor, bottom_anchor : str, array-like, optional
        Name of the top or bottom anchor body part or list of body parts. Default is 'neck'
        for top and 'tail_joint' for bottom. If a list is provided, it uses the center of 
        mass of the two points.
    """
    body_coords = []
    for part in parts:
        coords = data[part][['x', 'y']].values
        body_coords.append(coords)
    body_coords = np.array(body_coords).transpose((1, 0, 2))  # shape (T, N_parts, 2)
    # measure the mean jitter, per part, over time
    # breakpoint()
    # mean_jitter = np.linalg.norm(np.mean(np.diff(body_coords, axis=0), axis=0), axis=1)
    jitter = np.linalg.norm(np.diff(body_coords, axis=0), axis=2)
    mean_jitter = jitter.mean(axis=0)
    # now get the weighted center of mass
    weighted_com = np.average(body_coords, axis=1, weights=1/mean_jitter)
    # subtract from body_coords
    body_coords -= weighted_com[:, np.newaxis, :]
    # use generalized procrustes analysis to find the mean shape
    template = generalized_procrustes(body_coords)
    # align the template so that the vector from the neck to the tail base is along the x-axis
    coords = {}
    # allow for using the mean of multiple points for the anchors
    for key, anchor in zip(['top', 'bottom'], [top_anchor, bottom_anchor]):
        if isinstance(anchor, list):
            coords[key] = np.mean([template[parts.index(part), :] for part in anchor], axis=0)
        else:
            coords[key] = template[parts.index(anchor), :]
    neck_to_tail = coords['top'] - coords['bottom']
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

def get_body_angle(*args, **kwargs):
    return get_angles_procrustes(*args, **kwargs)

def get_head_angle(data, parts=head_parts, top_anchor=['antenna_left', 'antenna_right'], bottom_anchor='neck'):
    return get_angles_procrustes(data, parts=parts, top_anchor=top_anchor, bottom_anchor=bottom_anchor)


class AngularKalmanFilter():
    '''
    1D Kalman filter for angular data with optional viscous damping physics.
    
    For flies with active control, use low or zero damping (0-5 Hz).
    For passive coasting, use higher damping (10-50 Hz).
    '''
    def __init__(self, damping_coefficient=0.0, fps=60.0,
                 measurement_noise_deg=5.0, process_noise_scale=1.0):
        self.damping_coefficient = damping_coefficient
        self.dt = 1.0 / fps
        self.measurement_noise = np.deg2rad(measurement_noise_deg)
        self.process_noise_scale = process_noise_scale
        
        self.state = np.zeros(2)  # [angle, angular_velocity]
        
        if self.damping_coefficient > 0:
            self.decay_factor = np.exp(-self.damping_coefficient * self.dt)
        else:
            self.decay_factor = 1.0  # Constant velocity model
        
        self.A = np.array([[1.0, self.dt], [0.0, self.decay_factor]])
        self.C = np.array([[1.0, 0.0]])
        
        angular_accel_std = np.deg2rad(200.0 * self.process_noise_scale)
        self.Q = np.array([
            [angular_accel_std**2 * self.dt**4 / 4,  angular_accel_std**2 * self.dt**3 / 2],
            [angular_accel_std**2 * self.dt**3 / 2,  angular_accel_std**2 * self.dt**2]
        ])
        
        self.R = np.array([[self.measurement_noise**2]])
        self.P = np.array([[self.measurement_noise**2, 0.0], [0.0, np.deg2rad(360.0)**2]])
        
        self.filtered_angles = []
        self.filtered_velocities = []
    
    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def predict(self):
        self.state = self.A @ self.state
        self.state[0] = self.normalize_angle(self.state[0])
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.state[0]
    
    def update(self, measurement_deg):
        measurement = np.deg2rad(measurement_deg)
        innovation = self.normalize_angle(measurement - self.state[0])
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T / S
        self.state = self.state + K.flatten() * innovation
        self.state[0] = self.normalize_angle(self.state[0])
        self.P = (np.eye(2) - np.outer(K, self.C)) @ self.P
        self.filtered_angles.append(np.rad2deg(self.state[0]))
        self.filtered_velocities.append(np.rad2deg(self.state[1]))
        return self.state[0]
    
    def add_initial_state(self, angle_deg, angular_velocity_deg_per_s=0.0):
        self.state[0] = np.deg2rad(angle_deg)
        self.state[1] = np.deg2rad(angular_velocity_deg_per_s)
        self.filtered_angles = [angle_deg]
        self.filtered_velocities = [angular_velocity_deg_per_s]
    
    def filter_sequence(self, angles_deg):
        self.add_initial_state(angles_deg[0])
        for angle in angles_deg[1:]:
            self.predict()
            self.update(angle)
        return np.array(self.filtered_angles), np.array(self.filtered_velocities)
    
    def compute_log_likelihood(self, angles_deg):
        self.add_initial_state(angles_deg[0])
        log_likelihood = 0.0
        for angle_deg in angles_deg[1:]:
            self.predict()
            measurement = np.deg2rad(angle_deg)
            innovation = self.normalize_angle(measurement - self.state[0])
            S = self.C @ self.P @ self.C.T + self.R
            S_scalar = S[0, 0]
            log_lik_contrib = -0.5 * (np.log(2 * np.pi) + np.log(S_scalar) + innovation**2 / S_scalar)
            log_likelihood += log_lik_contrib
            self.update(angle_deg)
        return log_likelihood
    
    @staticmethod
    def optimize_parameters(angles_deg, fps, 
                           damping_bounds=(0.0, 50.0),
                           measurement_noise_bounds=(0.5, 20.0),
                           process_noise_bounds=(0.1, 10.0),
                           verbose=False):
        def negative_log_likelihood(params):
            damping, meas_noise, proc_noise = params
            kf = AngularKalmanFilter(damping, fps, meas_noise, proc_noise)
            return -kf.compute_log_likelihood(angles_deg)
        
        x0 = [(damping_bounds[0] + damping_bounds[1]) / 2,
              (measurement_noise_bounds[0] + measurement_noise_bounds[1]) / 2,
              (process_noise_bounds[0] + process_noise_bounds[1]) / 2]
        
        bounds = [damping_bounds, measurement_noise_bounds, process_noise_bounds]
        
        if verbose:
            print(f"Optimizing parameters... initial: damping={x0[0]:.2f} Hz, noise={x0[1]:.2f}°, proc={x0[2]:.2f}")
        
        result = minimize(negative_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'disp': verbose})
        
        optimal_damping, optimal_meas_noise, optimal_proc_noise = result.x
        
        if verbose:
            print(f"Optimal: damping={optimal_damping:.2f} Hz, noise={optimal_meas_noise:.2f}°, proc={optimal_proc_noise:.2f}")
        
        return {
            'damping_coefficient': optimal_damping,
            'measurement_noise_deg': optimal_meas_noise,
            'process_noise_scale': optimal_proc_noise,
            'log_likelihood': -result.fun,
            'success': result.success
        }


def get_filtered_angles(data, parts=body_parts, top_anchor='neck', bottom_anchor='tail_joint',
                       fps=60.0, optimize_params=True,
                       damping_coefficient=None, measurement_noise_deg=None, process_noise_scale=None,
                       max_time_constant_ms=100.0, verbose=False):
    """
    Get body or head angles with optional Kalman filtering.
    
    Parameters
    ----------
    data : DataFrame
        DLC tracking data
    parts : list
        Body parts to use for angle calculation
    top_anchor, bottom_anchor : str or list
        Anchor points for orientation
    fps : float
        Video framerate in frames per second (default: 60.0)
    optimize_params : bool
        If True, automatically optimize Kalman filter parameters
    damping_coefficient : float or None
        Manual damping coefficient (Hz), or None to optimize
    measurement_noise_deg : float or None
        Manual measurement noise (deg), or None to optimize
    process_noise_scale : float or None
        Manual process noise scale, or None to optimize
    max_time_constant_ms : float
        Maximum time constant for response (sets min damping)
    verbose : bool
        Print optimization details
    
    Returns
    -------
    centers : ndarray
        Body/head center positions (N, 2)
    angles_raw : ndarray
        Raw angles in radians (N,)
    angles_filtered : ndarray or None
        Filtered angles in radians (N,) if Kalman enabled, else None
    velocities : ndarray or None
        Angular velocities in rad/s (N,) if Kalman enabled, else None
    filter_params : dict or None
        Optimized parameters if Kalman enabled, else None
    """
    # Get raw angles using Procrustean analysis
    centers, angles_raw = get_angles_procrustes(data, parts=parts, 
                                                top_anchor=top_anchor, 
                                                bottom_anchor=bottom_anchor)
    
    # Convert to degrees for filtering
    angles_raw_deg = np.rad2deg(angles_raw)
    
    # Compute sampling interval from fps
    sampling_interval = 1.0 / fps
    
    # Apply Kalman filter if requested
    if optimize_params or damping_coefficient is not None:
        min_damping = 1000.0 / max_time_constant_ms
        max_damping = 50.0
        
        # Optimize if any parameter is None
        if optimize_params and (damping_coefficient is None or measurement_noise_deg is None or 
                               process_noise_scale is None):
            if verbose:
                print(f"Optimizing Kalman parameters (damping bounds: {min_damping:.1f}-{max_damping:.1f} Hz)...")
            
            filter_params = AngularKalmanFilter.optimize_parameters(
                angles_raw_deg,
                fps,
                damping_bounds=(min_damping, max_damping),
                measurement_noise_bounds=(0.5, 20.0),
                process_noise_bounds=(0.1, 10.0),
                verbose=verbose
            )
            
            damping = filter_params['damping_coefficient']
            meas_noise = filter_params['measurement_noise_deg']
            proc_noise = filter_params['process_noise_scale']
        else:
            # Use provided parameters
            damping = damping_coefficient if damping_coefficient is not None else 1.0
            meas_noise = measurement_noise_deg if measurement_noise_deg is not None else 5.0
            proc_noise = process_noise_scale if process_noise_scale is not None else 1.0
            filter_params = {'damping_coefficient': damping, 
                           'measurement_noise_deg': meas_noise,
                           'process_noise_scale': proc_noise}
        
        # Apply filter
        kf = AngularKalmanFilter(damping, fps, meas_noise, proc_noise)
        angles_filtered_deg, velocities_deg = kf.filter_sequence(angles_raw_deg)
        
        # Convert back to radians
        angles_filtered = np.deg2rad(angles_filtered_deg)
        velocities = np.deg2rad(velocities_deg)
        
        if verbose:
            print(f"Filtered with: damping={damping:.2f} Hz, noise={meas_noise:.2f}°, proc={proc_noise:.2f}")
        
        return centers, angles_raw, angles_filtered, velocities, filter_params
    else:
        return centers, angles_raw, None, None, None

def process_tracking_file(h5_path, fps=60.0, optimize=True, 
                         body_damping=None, head_damping=None,
                         measurement_noise=None, process_noise=None,
                         body_max_tc_ms=100.0, head_max_tc_ms=50.0,
                         output_path=None, verbose=True):
    """
    Process DLC tracking file to add filtered body and head angles.
    
    Parameters
    ----------
    h5_path : str
        Path to DLC HDF5 tracking file
    fps : float
        Video frame rate
    optimize : bool
        If True, optimize Kalman parameters
    body_damping, head_damping : float or None
        Manual damping coefficients (Hz)
    measurement_noise : float or None
        Manual measurement noise (degrees)
    process_noise : float or None
        Manual process noise scale
    body_max_tc_ms : float
        Maximum time constant for body (ms)
    head_max_tc_ms : float
        Maximum time constant for head (ms)
    output_path : str or None
        Output file path (if None, overwrites input)
    verbose : bool
        Print progress
    
    Returns
    -------
    df : DataFrame
        Updated dataframe with angle columns
    """
    if verbose:
        print(f"Loading tracking data from: {h5_path}")
    
    df = pd.read_hdf(h5_path)
    first_key = df.keys()[0][0]
    sampling_interval = 1.0 / fps
    
    if verbose:
        print(f"Processing {len(df)} frames at {fps:.1f} fps")
        print("=" * 60)
    
    # Process body angles
    if verbose:
        print("\n=== Computing Body Angles ===")
    
    body_centers, body_raw, body_filt, body_vel, body_params = get_filtered_angles(
        df[first_key],
        fps=fps,
        optimize_params=optimize,
        damping_coefficient=body_damping,
        measurement_noise_deg=measurement_noise,
        process_noise_scale=process_noise,
        max_time_constant_ms=body_max_tc_ms,
        verbose=verbose
    )
    
    # Process head angles
    if verbose:
        print("\n=== Computing Head Angles ===")
    
    head_centers, head_raw, head_filt, head_vel, head_params = get_filtered_angles(
        df[first_key],
        parts=['neck', 'antenna_left', 'antenna_right', 'head_left', 'head_right'],
        top_anchor=['antenna_left', 'antenna_right'],
        bottom_anchor='neck',
        fps=fps,
        optimize_params=optimize,
        damping_coefficient=head_damping,
        measurement_noise_deg=measurement_noise,
        process_noise_scale=process_noise,
        max_time_constant_ms=head_max_tc_ms,
        verbose=verbose
    )
    
    # Add columns to dataframe
    df['body_angle_raw'] = body_raw
    df['body_x'] = body_centers[:, 0]
    df['body_y'] = body_centers[:, 1]
    
    if body_filt is not None:
        df['body_angle_filtered'] = body_filt
        df['body_angular_velocity'] = body_vel
    
    df['head_angle_raw'] = head_raw
    df['head_x'] = head_centers[:, 0]
    df['head_y'] = head_centers[:, 1]
    
    if head_filt is not None:
        df['head_angle_filtered'] = head_filt
        df['head_angular_velocity'] = head_vel
    
    # Save
    output = output_path if output_path else h5_path
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Saving to: {output}")
    
    df.to_hdf(output, key='df', mode='w')
    
    if verbose:
        print("\nColumns added:")
        print("  - body_angle_raw, body_x, body_y")
        print("  - head_angle_raw, head_x, head_y")
        if body_filt is not None:
            print("  - body_angle_filtered, body_angular_velocity")
            print("  - head_angle_filtered, head_angular_velocity")
        
        if body_params:
            print(f"\nBody filter parameters:")
            print(f"  Damping: {body_params['damping_coefficient']:.2f} Hz (τ={1000/body_params['damping_coefficient']:.1f} ms)")
            print(f"  Measurement noise: {body_params['measurement_noise_deg']:.2f}°")
            print(f"  Process noise scale: {body_params['process_noise_scale']:.2f}")
        
        if head_params:
            print(f"\nHead filter parameters:")
            print(f"  Damping: {head_params['damping_coefficient']:.2f} Hz (τ={1000/head_params['damping_coefficient']:.1f} ms)")
            print(f"  Measurement noise: {head_params['measurement_noise_deg']:.2f}°")
            print(f"  Process noise scale: {head_params['process_noise_scale']:.2f}")
        
        print("\nDone!")
    
    return df
