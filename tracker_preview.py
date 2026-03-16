from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from procrustean_analysis import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph import Qt
from PyQt6 import QtGui
import skvideo.io

np.float = float
np.int = int


class KalmanFilter():
    '''
    2D Kalman filter, assuming constant acceleration.

    Use a 2D Kalman filter to return the estimated position of points given 
    linear prediction of position assuming (1) fixed jerk, (2) gaussian
    jerk noise, and (3) gaussian measurement noise.

    Parameters
    ----------
    num_objects : int
        Number of objects to model as well to expect from detector.
    sampling_interval : float
        Sampling interval in seconds. Should equal (frame rate) ** -1.
    jerk : float
        Jerk is modeled as normally distributed. This is the mean.
    jerk_std : float
        Jerk distribution standard deviation.
    measurement_noise_x : float
        Variance of the x component of measurement noise.
    measurement_noise_y : float
        Variance of the y component of measurement noise.

    '''
    def __init__(self, num_objects, num_frames=None, sampling_interval=60**-1,
                 jerk=0, jerk_std=125,
                 measurement_noise_x=5, measurement_noise_y=5,
                 width=None, height=None, distance_threshold=20,
                 error_threshold=5):
        self.record = []
        self.distance_threshold = distance_threshold
        self.error_threshold = error_threshold
        self.error_frames = 0
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.num_frames = num_frames
        self.sampling_interval = sampling_interval
        self.dt = self.sampling_interval
        self.jerk = jerk
        self.jerk_std = jerk_std
        self.measurement_noise_x = measurement_noise_x
        self.measurement_noise_y = measurement_noise_y
        self.tkn_x, self.tkn_y = self.measurement_noise_x, self.measurement_noise_y
        # process error covariance matrix
        self.Ez = np.array(
            [[self.tkn_x, 0         ],
             [0,          self.tkn_y]])
        # measurement error covariance matrix (constant jerk)
        self.Ex = np.array(
            [[self.dt**6/36, 0,             self.dt**5/12, 0,             self.dt**4/6, 0           ],
             [0,             self.dt**6/36, 0,             self.dt**5/12, 0,            self.dt**4/6],
             [self.dt**5/12, 0,             self.dt**4/4,  0,             self.dt**3/2, 0           ],
             [0,             self.dt**5/12, 0,             self.dt**4/4,  0,            self.dt**3/2],
             [self.dt**4/6,  0,             self.dt**3/2,  0,             self.dt**2,   0           ],
             [0,             self.dt**4/6,  0,             self.dt**3/2,  0,            self.dt**2  ]])
        self.Ex *= self.jerk_std**2
        # set initial position variance
        self.P = np.copy(self.Ex)
        ## define update equations in 2D as matrices - a physics based model for predicting
        # object motion
        ## we expect objects to be at:
        # [state update matrix (position + velocity)] + [input control (acceleration)]
        self.state_update_matrix = np.array(
            [[1, 0, self.dt, 0,       self.dt**2/2, 0           ],
             [0, 1, 0,       self.dt, 0,            self.dt**2/2],
             [0, 0, 1,       0,       self.dt,      0           ],
             [0, 0, 0,       1,       0,            self.dt     ],
             [0, 0, 0,       0,       1,            0           ],
             [0, 0, 0,       0,       0,            1           ]])
        self.control_matrix = np.array(
            [self.dt**3/6, self.dt**3/6, self.dt**2/2, self.dt**2/2, self.dt, self.dt])
        # measurement function to predict next measurement
        self.measurement_function = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]])
        self.A = self.state_update_matrix
        self.B = self.control_matrix
        self.C = self.measurement_function
        ## initialize result variables
        self.Q_local_measurement = []  # point detections
        ## initialize estimateion variables for two dimensions
        self.max_tracks = self.num_objects
        dimension = self.state_update_matrix.shape[0]
        self.Q_estimate = np.empty((dimension, self.max_tracks))
        self.Q_estimate.fill(np.nan)
        if self.num_frames is not None:
            self.Q_loc_estimateX = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateX.fill(np.nan)
            self.Q_loc_estimateY = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateY.fill(np.nan)
        else:
            self.Q_loc_estimateX = []
            self.Q_loc_estimateY = []
        self.num_tracks = self.num_objects
        self.num_detections = self.num_objects
        self.frame_num = 0

    def get_prediction(self):
        '''
        Get next predicted coordinates using current state and measurement information.

        Returns
        -------
        estimated points : ndarray
            approximated positions with shape.
        '''
        ## kalman filter
        # predict next state with last state and predicted motion
        self.Q_estimate = self.A @ self.Q_estimate + (self.B * self.jerk)[:, None]
        # predict next covariance
        self.P = self.A @ self.P @ self.A.T + self.Ex
        # Kalman Gain
        try:
            self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Ez)
            ## now assign the detections to estimated track positions
            # make the distance (cost) matrix between all pairs; rows = tracks and
            # cols = detections
            self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
            # np.clip(self.estimate_points[0], -self.height/2, self.height/2, out=self.estimate_points[0])
            # np.clip(self.estimate_points[1], -self.width/2, self.width/2, out=self.estimate_points[1])
            return self.estimate_points.T  # shape should be (num_objects, 2)
        except:
            return np.array([np.nan, np.nan])
    

    def add_starting_points(self, points):
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_estimate.fill(0)
        self.Q_estimate[:2] = points.T
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def add_measurement(self, points):
        ## detections matrix
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        # check if any points are too far from the prediction
        if self.distance_threshold is not None:
            distances = np.linalg.norm(points - self.Q_estimate[:2].T, axis=1)
            too_far = distances > self.distance_threshold
            if np.any(too_far) and self.error_frames < self.error_threshold:
                # replace the measurement with the prediction
                points[too_far] = self.Q_estimate[:2, too_far].T
                self.error_frames += 1
            elif np.any(too_far) and self.error_frames >= self.error_threshold:
                self.error_frames = 0
        self.Q_loc_meas = points
        self.Q_estimate = self.Q_estimate + self.K @ (self.Q_loc_meas.T - self.C @ self.Q_estimate)
        # update covariance estimation
        self.P = (np.eye((self.K @ self.C).shape[0]) - self.K @ self.C) @ self.P
        ## store data
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def update_vals(self, **kwargs):
        """Allow for replacing parameters like jerk_std and noise estimates."""
        for key, val in kwargs.items():
            self.__setattr__(key, val)


class AngularKalmanFilter():
    '''
    1D Kalman filter for angular data with optional viscous damping physics.
    
    Uses either constant velocity model (good for active tracking) or viscous 
    damping model (good for passive coasting). For flies with active control,
    use low or zero damping.
    
    Parameters
    ----------
    damping_coefficient : float
        Damping rate in Hz (1/time_constant). Set to 0 for constant velocity model.
        For active tracking (oscillations, maneuvers): use 0-2 Hz
        For passive coasting: use 10-50 Hz
    sampling_interval : float
        Sampling interval in seconds (1/framerate).
    measurement_noise_deg : float
        Standard deviation of angle measurement noise in degrees.
    process_noise_scale : float
        Scale factor for process noise (accounts for active control, external perturbations).
        Higher values = more responsive to changes, less smooth.
    '''
    def __init__(self, damping_coefficient=0.0, sampling_interval=1/30.0,
                 measurement_noise_deg=5.0, process_noise_scale=1.0):
        self.damping_coefficient = damping_coefficient  # Hz
        self.dt = sampling_interval
        self.measurement_noise = np.deg2rad(measurement_noise_deg)
        self.process_noise_scale = process_noise_scale
        
        # State: [angle (rad), angular_velocity (rad/s)]
        self.state = np.zeros(2)
        
        # Viscous damping: velocity decays exponentially (or constant if damping=0)
        if self.damping_coefficient > 0:
            self.decay_factor = np.exp(-self.damping_coefficient * self.dt)
        else:
            self.decay_factor = 1.0  # Constant velocity model
        
        # State transition matrix
        self.A = np.array([
            [1.0, self.dt],
            [0.0, self.decay_factor]
        ])
        
        # Measurement matrix (we only observe angle, not velocity)
        self.C = np.array([[1.0, 0.0]])
        
        # Process noise covariance (uncertainty in dynamics)
        # Model active control as random angular accelerations
        # Higher values = filter tracks changes better but noisier
        angular_accel_std = np.deg2rad(200.0 * self.process_noise_scale)  # rad/s^2
        self.Q = np.array([
            [angular_accel_std**2 * self.dt**4 / 4,  angular_accel_std**2 * self.dt**3 / 2],
            [angular_accel_std**2 * self.dt**3 / 2,  angular_accel_std**2 * self.dt**2]
        ])
        
        # Measurement noise covariance
        self.R = np.array([[self.measurement_noise**2]])
        
        # State covariance matrix - start with high uncertainty
        self.P = np.array([
            [self.measurement_noise**2, 0.0],
            [0.0, np.deg2rad(360.0)**2]  # Large initial velocity uncertainty
        ])
        
        # Storage for filtered angles
        self.filtered_angles = []
        self.filtered_velocities = []
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def predict(self):
        """Predict next state using viscous damping model."""
        # State prediction
        self.state = self.A @ self.state
        self.state[0] = self.normalize_angle(self.state[0])
        
        # Covariance prediction
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.state[0]
    
    def update(self, measurement_deg):
        """Update state estimate with new angle measurement."""
        measurement = np.deg2rad(measurement_deg)
        
        # Innovation (measurement residual)
        innovation = self.normalize_angle(measurement - self.state[0])
        
        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R
        
        # Kalman gain
        K = self.P @ self.C.T / S
        
        # State update
        self.state = self.state + K.flatten() * innovation
        self.state[0] = self.normalize_angle(self.state[0])
        
        # Covariance update
        self.P = (np.eye(2) - np.outer(K, self.C)) @ self.P
        
        # Store results
        self.filtered_angles.append(np.rad2deg(self.state[0]))
        self.filtered_velocities.append(np.rad2deg(self.state[1]))
        
        return self.state[0]
    
    def add_initial_state(self, angle_deg, angular_velocity_deg_per_s=0.0):
        """Initialize filter with starting angle and velocity."""
        self.state[0] = np.deg2rad(angle_deg)
        self.state[1] = np.deg2rad(angular_velocity_deg_per_s)
        self.filtered_angles = [angle_deg]
        self.filtered_velocities = [angular_velocity_deg_per_s]
    
    def filter_sequence(self, angles_deg):
        """
        Filter an entire sequence of angles.
        
        Parameters
        ----------
        angles_deg : array-like
            Sequence of angle measurements in degrees
            
        Returns
        -------
        filtered_angles : ndarray
            Filtered angles in degrees
        filtered_velocities : ndarray
            Estimated angular velocities in deg/s
        """
        self.add_initial_state(angles_deg[0])
        
        for angle in angles_deg[1:]:
            self.predict()
            self.update(angle)
        
        return np.array(self.filtered_angles), np.array(self.filtered_velocities)
    
    def compute_log_likelihood(self, angles_deg):
        """
        Compute log-likelihood of angle sequence given current parameters.
        
        Used for parameter optimization via maximum likelihood estimation.
        
        Parameters
        ----------
        angles_deg : array-like
            Sequence of angle measurements in degrees
            
        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data
        """
        # Reset filter state
        self.add_initial_state(angles_deg[0])
        
        log_likelihood = 0.0
        
        for angle_deg in angles_deg[1:]:
            # Predict
            self.predict()
            
            # Compute innovation (prediction error)
            measurement = np.deg2rad(angle_deg)
            innovation = self.normalize_angle(measurement - self.state[0])
            
            # Innovation covariance
            S = self.C @ self.P @ self.C.T + self.R
            S_scalar = S[0, 0]
            
            # Log-likelihood contribution (Gaussian)
            # log p(z|x) = -0.5 * [log(2π) + log(|S|) + innovation^T S^-1 innovation]
            log_lik_contrib = -0.5 * (np.log(2 * np.pi) + np.log(S_scalar) + 
                                     innovation**2 / S_scalar)
            log_likelihood += log_lik_contrib
            
            # Update filter
            self.update(angle_deg)
        
        return log_likelihood
    
    @staticmethod
    def optimize_parameters(angles_deg, sampling_interval, 
                           damping_bounds=(0.0, 50.0),
                           measurement_noise_bounds=(0.5, 20.0),
                           process_noise_bounds=(0.1, 5.0),
                           verbose=True):
        """
        Find optimal Kalman filter parameters using maximum likelihood estimation.
        
        Parameters
        ----------
        angles_deg : array-like
            Sequence of angle measurements in degrees
        sampling_interval : float
            Time between measurements in seconds
        damping_bounds : tuple
            (min, max) bounds for damping coefficient in Hz
        measurement_noise_bounds : tuple
            (min, max) bounds for measurement noise std in degrees
        process_noise_bounds : tuple
            (min, max) bounds for process noise scale factor
        verbose : bool
            Print optimization progress
            
        Returns
        -------
        optimal_params : dict
            Dictionary with optimal 'damping_coefficient', 'measurement_noise_deg', 
            'process_noise_scale', and 'log_likelihood'
        """
        from scipy.optimize import minimize
        
        def negative_log_likelihood(params):
            damping, meas_noise, proc_noise = params
            
            # Create filter with these parameters
            kf = AngularKalmanFilter(
                damping_coefficient=damping,
                sampling_interval=sampling_interval,
                measurement_noise_deg=meas_noise,
                process_noise_scale=proc_noise
            )
            
            # Compute log-likelihood
            log_lik = kf.compute_log_likelihood(angles_deg)
            
            # Return negative (for minimization)
            return -log_lik
        
        # Initial guess (middle of bounds)
        x0 = [
            (damping_bounds[0] + damping_bounds[1]) / 2,
            (measurement_noise_bounds[0] + measurement_noise_bounds[1]) / 2,
            (process_noise_bounds[0] + process_noise_bounds[1]) / 2
        ]
        
        # Bounds for optimization
        bounds = [damping_bounds, measurement_noise_bounds, process_noise_bounds]
        
        if verbose:
            print(f"\nOptimizing Kalman filter parameters...")
            print(f"Initial guess: damping={x0[0]:.2f} Hz, meas_noise={x0[1]:.2f}°, proc_noise={x0[2]:.2f}")
        
        # Optimize using L-BFGS-B (handles bounds well)
        result = minimize(
            negative_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': verbose}
        )
        
        optimal_damping, optimal_meas_noise, optimal_proc_noise = result.x
        optimal_log_lik = -result.fun
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Optimal parameters:")
            print(f"  Damping coefficient: {optimal_damping:.3f} Hz")
            print(f"  Measurement noise:   {optimal_meas_noise:.3f}°")
            print(f"  Process noise scale: {optimal_proc_noise:.3f}")
            print(f"  Log-likelihood:      {optimal_log_lik:.2f}")
            if optimal_damping < 1.0:
                print(f"  → Low damping suggests active control dominates")
            elif optimal_damping > 20.0:
                print(f"  → High damping suggests passive dynamics")
        
        return {
            'damping_coefficient': optimal_damping,
            'measurement_noise_deg': optimal_meas_noise,
            'process_noise_scale': optimal_proc_noise,
            'log_likelihood': optimal_log_lik,
            'success': result.success
        }


class ImageViewWithSignal(pg.ImageView):
    frameChanged = QtCore.pyqtSignal(int)

    def setCurrentIndex(self, ind):
        super().setCurrentIndex(ind)
        self.frameChanged.emit(ind)



class TrackerPreview(QtWidgets.QWidget):
    def __init__(self, video_path, h5_path=None, parent=None, cmap='rainbow', kalman=True, procrustean=True, parts=None):
        super().__init__(parent)
        self.setWindowTitle("Tracker Preview")
        self.kalman = kalman
        self.procrustean = procrustean
        self.video_path = video_path
        self.h5_path = h5_path
        self.points = None
        self.cmap = cmap
        self.parts = parts
        self.subjective_mode = False
        # Load video frames using skvideo.io
        self.frames = skvideo.io.vread(self.video_path)
        self.init_ui()
        self.load_video()
        self.play_video()

    def init_ui(self):
        pg.setConfigOptions(antialias=True)

        layout = QtWidgets.QVBoxLayout(self)
        self.image_view = ImageViewWithSignal()
        # Call update_points whenever the frame index changes
        self.image_view.frameChanged.connect(self.update_points)
        self.image_view.getView().invertY(True)
        self.image_view.getView().invertX(False)  # or True if you want to flip X as well
        self.view_box = self.image_view.getView()
        # ViewBox configuration will be done after loading the image
        # to prevent setImage() from resetting it
        # add the layout
        layout.addWidget(self.image_view)

        # Controls layout
        controls_layout = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.fps_dropdown = QtWidgets.QComboBox()
        self.fps_options = [5, 15, 30, 60, 120, 240, 480]
        for fps in self.fps_options:
            self.fps_dropdown.addItem(f"{fps} FPS", fps)
        self.fps_dropdown.setCurrentIndex(self.fps_options.index(30))
        self.fps_label = QtWidgets.QLabel("Framerate: 30 FPS")
        self.subjective_checkbox = QtWidgets.QCheckBox("Egocentric View")

        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.fps_label)
        controls_layout.addWidget(self.fps_dropdown)
        controls_layout.addWidget(self.subjective_checkbox)
        layout.addLayout(controls_layout)
        self.setLayout(layout)

        # Connect signals
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.fps_dropdown.currentIndexChanged.connect(self.update_fps_label)
        self.subjective_checkbox.stateChanged.connect(self.toggle_subjective_mode)

        # Timer for playback
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.current_frame = 0
        self.fps = 30

    def load_video(self):
        # transpose to swap the x- and y-axes
        self.frames = self.frames.transpose((0, 2, 1, 3))
        self.num_frames = self.frames.shape[0]
        self.current_frame = 0
        self.border_pad = 0
        
        # Get dimensions before setImage
        num_frames, self.img_height, self.img_width, num_channels = self.frames.shape
        
        # Load the image data
        self.image_view.setImage(self.frames, xvals=np.arange(self.num_frames))
        
        # NOW configure the ViewBox AFTER setImage() so it doesn't get reset
        self.view_box.setAspectLocked(True)
        self.view_box.disableAutoRange()
        max_length = np.sqrt(self.img_height**2 + self.img_width**2)
        self.view_box.setRange(
            xRange=(-max_length/2, max_length/2),
            yRange=(-max_length/2, max_length/2),
            padding=0
        )
        
        # Center the image at (0, 0) in objective mode
        self.apply_transform(subjective=self.subjective_mode)
        self.image_view.setCurrentIndex(0)
        # Optionally, overlay points if provided
        if self.h5_path is not None:
            self.load_points()


    def load_points(self):
        # use pandas to load the h5 dataset
        self.data = pd.read_hdf(self.h5_path)
        first_key = self.data.keys()[0]
        self.points = self.data[first_key[0]]
        keys = self.points.keys()
        if self.parts is None:
            self.parts = list(set([key[0] for key in keys]))
        cmap = plt.get_cmap(self.cmap)
        self.colors = (255 * cmap(np.linspace(0, 1, len(self.parts)))).astype('uint8')

        # Add a legend to the top left in the image view
        self.legend = pg.LegendItem(offset=(10, 10))
        self.legend.setParentItem(self.image_view.getView())
        self.legend.anchor((0, 0), (0, 0))
        # Remove previous legend items if reloading
        if hasattr(self, 'legend_items'):
            for item in self.legend_items:
                self.legend.removeItem(item)
        self.legend_items = []
        for part, color in zip(self.parts, self.colors):
            dummy = pg.ScatterPlotItem([0], [0], pen=pg.mkPen(color), brush=pg.mkBrush(color), size=10)
            self.legend.addItem(dummy, str(part))
            self.legend_items.append(part)
    
    def compute_filtered_angles(self, damping_coefficient=None, measurement_noise_deg=None, 
                                process_noise_scale=None, optimize_params=True,
                                max_time_constant_ms=100.0, head_max_time_constant_ms=None):
        """
        Pre-compute filtered body and head angles using Kalman filter.
        
        This runs the Procrustean analysis to get raw angles, then applies angular Kalman
        filtering to smooth them. Can automatically optimize parameters via MLE.
        
        Parameters
        ----------
        damping_coefficient : float or None
            Viscous damping rate in Hz. If None and optimize_params=True, will be optimized.
            Set to 0 for constant velocity model (good for active tracking).
        measurement_noise_deg : float or None
            Expected measurement noise in degrees. If None and optimize_params=True, will be optimized.
        process_noise_scale : float or None
            Scale factor for process noise. If None and optimize_params=True, will be optimized.
        optimize_params : bool
            If True and any parameter is None, optimize parameters using MLE.
        max_time_constant_ms : float
            Maximum allowed time constant for BODY in milliseconds. Sets lower bound on damping.
            Lower values = faster response. Typical: 50-150 ms for body tracking.
        head_max_time_constant_ms : float or None
            Maximum allowed time constant for HEAD in milliseconds. If None, uses 50% of body value.
            Head typically needs faster response (20-50 ms) due to faster movements.
        """
        if not self.procrustean or not hasattr(self, 'points'):
            print("Procrustean analysis disabled or points not loaded. Skipping angle filtering.")
            return
        
        print("Computing filtered angles...")
        
        # First, compute raw angles using Procrustean analysis if not already done
        if not hasattr(self, 'body_angles'):
            self.body_centers, self.body_angles = get_angles_procrustes(self.points)
            self.body_angles = (self.body_angles + np.pi) % (2 * np.pi)
            self.body_angles = np.rad2deg(self.body_angles)
        
        if not hasattr(self, 'head_angles'):
            self.head_centers, self.head_angles = get_head_angle(self.points)
            self.head_angles = (self.head_angles + np.pi) % (2 * np.pi)
            self.head_angles = np.rad2deg(self.head_angles)
        
        sampling_interval = 1.0 / self.fps if hasattr(self, 'fps') else 1.0 / 30.0
        
        # Calculate damping bounds from max time constant
        # Head typically needs faster response than body
        if head_max_time_constant_ms is None:
            head_max_time_constant_ms = max_time_constant_ms * 0.5  # Head responds 2x faster
        
        body_min_damping = 1000.0 / max_time_constant_ms  # Hz
        head_min_damping = 1000.0 / head_max_time_constant_ms  # Hz
        max_damping = 50.0  # Hz (upper limit for numerical stability)
        
        print(f"Body damping bounds: {body_min_damping:.2f} - {max_damping:.2f} Hz (τ: {1000/max_damping:.1f} - {max_time_constant_ms:.1f} ms)")
        print(f"Head damping bounds: {head_min_damping:.2f} - {max_damping:.2f} Hz (τ: {1000/max_damping:.1f} - {head_max_time_constant_ms:.1f} ms)")
        
        # Optimize parameters if requested and any are None
        if optimize_params and (damping_coefficient is None or measurement_noise_deg is None or 
                               process_noise_scale is None):
            print("\n=== Optimizing Body Angle Parameters ===")
            body_params = AngularKalmanFilter.optimize_parameters(
                self.body_angles,
                sampling_interval,
                damping_bounds=(body_min_damping, max_damping),
                measurement_noise_bounds=(0.5, 20.0),
                process_noise_bounds=(0.1, 10.0),  # Increased upper bound
                verbose=True
            )
            
            print("\n=== Optimizing Head Angle Parameters ===")
            head_params = AngularKalmanFilter.optimize_parameters(
                self.head_angles,
                sampling_interval,
                damping_bounds=(head_min_damping, max_damping),
                measurement_noise_bounds=(0.5, 20.0),
                process_noise_bounds=(0.1, 10.0),  # Increased upper bound
                verbose=True
            )
            
            # Use optimized parameters (override any that were None)
            body_damping = damping_coefficient if damping_coefficient is not None else body_params['damping_coefficient']
            body_meas_noise = measurement_noise_deg if measurement_noise_deg is not None else body_params['measurement_noise_deg']
            body_proc_noise = process_noise_scale if process_noise_scale is not None else body_params['process_noise_scale']
            
            head_damping = damping_coefficient if damping_coefficient is not None else head_params['damping_coefficient']
            head_meas_noise = measurement_noise_deg if measurement_noise_deg is not None else head_params['measurement_noise_deg']
            head_proc_noise = process_noise_scale if process_noise_scale is not None else head_params['process_noise_scale']
            
            # Store optimized parameters
            self.body_filter_params = body_params
            self.head_filter_params = head_params
        else:
            # Use provided parameters or defaults
            body_damping = damping_coefficient if damping_coefficient is not None else 1.0
            body_meas_noise = measurement_noise_deg if measurement_noise_deg is not None else 5.0
            body_proc_noise = process_noise_scale if process_noise_scale is not None else 1.0
            
            head_damping = body_damping
            head_meas_noise = body_meas_noise
            head_proc_noise = body_proc_noise
        
        # Create Kalman filters with optimized or provided parameters
        print(f"\nFiltering with parameters:")
        print(f"  Body: damping={body_damping:.2f} Hz, meas_noise={body_meas_noise:.2f}°, proc_noise={body_proc_noise:.2f}")
        print(f"  Head: damping={head_damping:.2f} Hz, meas_noise={head_meas_noise:.2f}°, proc_noise={head_proc_noise:.2f}")
        
        body_kf = AngularKalmanFilter(
            damping_coefficient=body_damping,
            sampling_interval=sampling_interval,
            measurement_noise_deg=body_meas_noise,
            process_noise_scale=body_proc_noise
        )
        
        head_kf = AngularKalmanFilter(
            damping_coefficient=head_damping,
            sampling_interval=sampling_interval,
            measurement_noise_deg=head_meas_noise,
            process_noise_scale=head_proc_noise
        )
        
        # Filter the angle sequences
        self.body_angles_filtered, self.body_velocities = body_kf.filter_sequence(self.body_angles)
        self.head_angles_filtered, self.head_velocities = head_kf.filter_sequence(self.head_angles)
        
        print(f"\nFiltered {len(self.body_angles)} frames successfully!")
            
    def play_video(self):
        self.current_frame = self.image_view.currentIndex
        self.fps = int(self.fps_dropdown.currentData())
        
        # If Kalman filtering enabled and not already computed, pre-compute filtered angles
        if self.kalman and self.procrustean and hasattr(self, 'points'):
            if not hasattr(self, 'body_angles_filtered'):
                self.compute_filtered_angles()
        
        # Setup procrustean arrows if enabled
        if self.procrustean:
            if 'body_centers' not in dir(self):
                # use get_angles_procrustes on the full dataset
                self.body_centers, self.body_angles = get_angles_procrustes(self.points)
                # rotate by pi
                self.body_angles = (self.body_angles + np.pi) % (2 * np.pi)
                # convert body angles to degrees
                self.body_angles = np.rad2deg(self.body_angles)
            else:
                if hasattr(self, 'body_arrow'):
                    # if this has been run before, remove the previous arrow
                    self.image_view.removeItem(self.body_arrow)
            # and add an arrow that will point to centers from tail_joint
            # use the mean distance from the tail joint to the body centers as the length of the arrow
            arrow_length = np.linalg.norm(self.points['tail_joint'][['x', 'y']].values - self.body_centers, axis=1).mean()
            x, y = self.body_centers[0]
            # Use filtered angles if available
            angle_to_use = self.body_angles_filtered[0] if hasattr(self, 'body_angles_filtered') else self.body_angles[0]
            self.body_arrow = pg.ArrowItem(angle=angle_to_use, tailLen=arrow_length, pen='r', brush='r', pxMode=False)
            self.body_arrow.setPos(x, y)
            self.image_view.addItem(self.body_arrow)
            
            # do the same for the head
            if 'head_centers' not in dir(self):
                # use get_angles_procrustes on the full dataset
                self.head_centers, self.head_angles = get_head_angle(self.points)
                # rotate by pi
                self.head_angles = (self.head_angles + np.pi) % (2 * np.pi)
                # convert head angles to degrees
                self.head_angles = np.rad2deg(self.head_angles)
            else:
                # if this has been run before, remove the previous arrow
                if hasattr(self, 'head_arrow'):
                    self.image_view.removeItem(self.head_arrow)
            # and add an arrow that will point to centers from tail_joint
            # use the mean distance from the tail joint to the body centers as the length of the arrow
            bottom = self.points['neck'][['x', 'y']].values
            # top = self.points[['antenna_left', 'antenna_right']][['x', 'y']].values.mean(axis=1)
            top = .5*(self.points['antenna_left'][['x', 'y']].values + self.points['antenna_right'][['x', 'y']].values)
            arrow_length = np.linalg.norm(bottom - top, axis=1).mean()
            # let's make a new array of points closer to the antennae by adding the offset at the right angle
            # Use filtered or raw angles for positioning
            angles_for_centers = self.head_angles_filtered if hasattr(self, 'head_angles_filtered') else self.head_angles
            self.head_arrow_centers = self.head_centers - arrow_length * np.array([np.cos(np.deg2rad(angles_for_centers)), np.sin(np.deg2rad(angles_for_centers))]).T
            x, y = self.head_arrow_centers[0]
            angle_to_use = self.head_angles_filtered[0] if hasattr(self, 'head_angles_filtered') else self.head_angles[0]
            self.head_arrow = pg.ArrowItem(angle=angle_to_use, tailLen=arrow_length, pen='r', brush='r', pxMode=False)
            self.head_arrow.setPos(x, y)
            self.image_view.addItem(self.head_arrow)
            
        self.timer.start(int(1000 / self.fps))
        self.is_playing = True

    def pause_video(self):
        self.timer.stop()
        self.is_playing = False

    def next_frame(self):
        if self.frames is not None:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.border_pad = int(round(abs(self.img_height - self.frames[self.current_frame].shape[0])/2))
            self.image_view.setCurrentIndex(self.current_frame)
            # update the points overlay
            self.update_points()
            # update the body arrow if procrustean analysis is enabled
            if self.procrustean:
                self.update_arrows()
                self.apply_transform(subjective=self.subjective_mode)

    def update_fps_label(self, index):
        fps = int(self.fps_dropdown.itemData(index))
        self.fps_label.setText(f"Framerate: {fps} FPS")
        if self.timer.isActive():
            self.timer.start(int(1000 / fps))

    def update_points(self):
        # Overlay tracked body part points on each frame
        # Remove previous scatter plot items if they exist
        if self.points is not None:
            if hasattr(self, 'scatter_items'):
                for item in self.scatter_items:
                    self.image_view.removeItem(item)
            self.scatter_items = []
            
            # Get body center and angle for transformation if in subjective mode
            if hasattr(self, 'body_centers'):
                # Use filtered angles if available for subjective transform
                body_angles_to_use = self.body_angles_filtered if hasattr(self, 'body_angles_filtered') else self.body_angles
                center_x, center_y = self.body_centers[self.current_frame]
                angle = body_angles_to_use[self.current_frame]
            else:
                center_x, center_y, angle = 0, 0, 0

            for num, (part, color) in enumerate(zip(self.parts, self.colors)):
                x, y, alpha = self.points.loc[self.current_frame, part][['x', 'y', 'likelihood']].values
                
                # Transform coordinates if in subjective mode
                if self.procrustean and hasattr(self, 'body_centers'):
                    x, y = self.transform_coordinates(x, y, center_x, center_y, angle, subjective=self.subjective_mode)
                
                new_color = np.copy(color)
                new_color[-1] = 255*alpha  # Set alpha based on likelihood
                pen = pg.mkPen(new_color)
                brush = pg.mkBrush(new_color)
                scatter = pg.ScatterPlotItem([x], [y], pen=pen, brush=brush)
                self.image_view.addItem(scatter)
                self.scatter_items.append(scatter)

    def update_arrows(self):
        if self.procrustean:
            # Get transformation parameters if in subjective mode
            if hasattr(self, 'body_centers'):
                # Use filtered angles if available, otherwise use raw angles
                body_angles_to_use = self.body_angles_filtered if hasattr(self, 'body_angles_filtered') else self.body_angles
                head_angles_to_use = self.head_angles_filtered if hasattr(self, 'head_angles_filtered') else self.head_angles
                
                body_center_x, body_center_y = self.body_centers[self.current_frame]
                body_angle = body_angles_to_use[self.current_frame]
                
                # update the body arrow
                if hasattr(self, 'body_arrow'):
                    x, y = self.body_centers[self.current_frame]
                    arrow_angle = body_angles_to_use[self.current_frame]
                    # Transform position and angle if in subjective mode
                    x, y = self.transform_coordinates(x, y, body_center_x, body_center_y, body_angle, subjective=self.subjective_mode)
                    if self.subjective_mode:
                        # Adjust arrow angle to account for view rotation
                        arrow_angle = arrow_angle - body_angle
                        arrow_angle += 90
                    self.body_arrow.setPos(x, y)
                    self.body_arrow.setStyle(angle=arrow_angle)
                    
                if hasattr(self, 'head_arrow'):
                    x, y = self.head_arrow_centers[self.current_frame]
                    arrow_angle = head_angles_to_use[self.current_frame]
                    # Transform position and angle if in subjective mode
                    x, y = self.transform_coordinates(x, y, body_center_x, body_center_y, body_angle, subjective=self.subjective_mode)
                    if self.subjective_mode:
                        # Adjust arrow angle to account for view rotation
                        arrow_angle = arrow_angle - body_angle
                        arrow_angle += 90
                    self.head_arrow.setPos(x, y)
                    self.head_arrow.setStyle(angle=arrow_angle)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space:
            print("space")
            self.toggle_play_pause()
        else:
            super().keyPressEvent(event)

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    # def center_image_objective(self):
    #     """Center the image at (0, 0) for objective view."""
    #     img_item = self.image_view.getImageItem()
    #     transform = QtGui.QTransform()
    #     dx = -self.img_width/2 - self.border_pad
    #     dy = -self.img_height/2 - self.border_pad
    #     transform.translate(dx, dy)
    #     img_item.setTransform(transform)

    def transform_coordinates(self, x, y, center_x, center_y, angle, subjective=True):
        """Transform coordinates to subjective reference frame.
        
        Parameters
        ----------
        x, y : float or array
            Original coordinates
        center_x, center_y : float
            Body center coordinates
        angle : float
            Body angle in degrees
            
        Returns
        -------
        x_trans, y_trans : transformed coordinates
        """
        # Apply the same transform as applied to the image
        # 1. Translate to center the body at origin
        xs = x - center_x
        ys = y - center_y
        if subjective:
            # 2. Rotate by -angle + 90 degrees
            angle_rad = np.deg2rad(-angle + 90)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            # rotate x and y
            x_trans = cos_a * xs - sin_a * ys
            y_trans = sin_a * xs + cos_a * ys
            # reset variables
            xs, ys = x_trans, y_trans
        return xs, ys

    def toggle_subjective_mode(self, state):
        """Toggle between objective and subjective (egocentric) reference frames."""
        self.subjective_mode = bool(state)
        if hasattr(self, 'body_centers'):
            self.apply_transform(subjective=self.subjective_mode)
        # if self.subjective_mode:
        #         if self.procrustean and hasattr(self, 'body_centers'):
        #             self.apply_transform(subjective=True)
        #     else:
        #         self.center_image_objective()
    
    def apply_transform(self, subjective=False):
        """Apply translation and rotation to center and align view with body."""
        if not hasattr(self, 'body_centers') or not hasattr(self, 'body_angles'):
            return
        # get the body center and angle
        center_x, center_y = self.body_centers[self.current_frame]
        angle = self.body_angles[self.current_frame]
        # Create transform - order matters: rotate first, then translate
        transform = QtGui.QTransform()
        # 1. Rotate to align body with fixed direction
        if subjective:
            transform.rotate(-angle + 90)  # Negative to counter-rotate the world
        # 2. Adjust for body center position (to keep body centered in view)
        dx = -center_x
        dy = -center_y
        # 3. Apply translation
        transform.translate(dx, dy)
        # Apply transform to the ImageItem, NOT the ViewBox
        img_item = self.image_view.getImageItem()
        img_item.setTransform(transform)

    def reset_view(self):
        """Reset view to objective reference frame."""
        # Reset to centered objective view
        self.center_image_objective()


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


# Allow running as a standalone application
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # You can change these paths or make them command-line arguments
    # video_path = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\PCF_SH\\vid\\fly_1_trial_1_cond1_vidData.mp4"
    # h5_fn = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\PCF_SH\\vid\\fly_1_trial_1_cond1_vidDataDLC_Resnet50_SuperFlyJul29shuffle1_snapshot_best-30.h5"
    # video_path = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\BAJA_FH\\vid\\fly_1_trial_1_cond1_vidData.mp4"
    # h5_fn = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\BAJA_FH\\vid\\fly_1_trial_1_cond1_vidDataDLC_Resnet50_SuperFlyJul29shuffle1_snapshot_best-30.h5"
    video_path = "C:\\Users\\johnp\\OneDrive\\Desktop\\smellovision\\old\\healthy_fly_exp\\demo\\2025_09_11_16_12_21_01.mp4"
    h5_fn = "C:\\Users\\johnp\\OneDrive\\Desktop\\smellovision\\old\\healthy_fly_exp\\demo\\2025_09_11_16_12_21_01DLC_Resnet50_SuperFlyJul29shuffle1_snapshot_best-180.h5"

    preview = TrackerPreview(video_path, h5_path=h5_fn, kalman=True)
    preview.show()
    sys.exit(app.exec())
