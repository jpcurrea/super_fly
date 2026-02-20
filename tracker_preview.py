from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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

class ImageViewWithSignal(pg.ImageView):
    frameChanged = QtCore.pyqtSignal(int)

    def setCurrentIndex(self, ind):
        super().setCurrentIndex(ind)
        self.frameChanged.emit(ind)



class TrackerPreview(QtWidgets.QWidget):
    def __init__(self, video_path, h5_path=None, parent=None, cmap='rainbow', kalman=False, procrustean=True, parts=None):
        super().__init__(parent)
        self.setWindowTitle("Tracker Preview")
        self.kalman = kalman
        self.procrustean = procrustean
        self.video_path = video_path
        self.h5_path = h5_path
        self.points = None
        self.cmap = cmap
        self.parts = parts
        self.init_ui()
        self.load_video()
        self.play_video()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.image_view = ImageViewWithSignal()
        # Call update_points whenever the frame index changes
        self.image_view.frameChanged.connect(self.update_points)
        self.image_view.getView().invertY(True)
        self.image_view.getView().invertX(False)  # or True if you want to flip X as well
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

        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.fps_label)
        controls_layout.addWidget(self.fps_dropdown)
        layout.addLayout(controls_layout)
        self.setLayout(layout)

        # Connect signals
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.fps_dropdown.currentIndexChanged.connect(self.update_fps_label)

        # Timer for playback
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.current_frame = 0
        self.frames = None
        self.fps = 30

    def load_video(self):
        # Load video frames using skvideo.io
        self.frames = skvideo.io.vread(self.video_path)
        # transpose to swap the x- and y-axes
        self.frames = self.frames.transpose((0, 2, 1, 3))
        self.frames
        self.num_frames = self.frames.shape[0]
        self.current_frame = 0
        self.image_view.setImage(self.frames, xvals=np.arange(self.num_frames))
        img_item = self.image_view.getImageItem()
        # img_item.setTransform(QtGui.QTransform().scale(1, -1).rotate(90))
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
            
    def play_video(self):
        self.current_frame = self.image_view.currentIndex
        self.fps = int(self.fps_dropdown.currentData())
        # for x in range(10):
        #     # for troubleshooting
        #     self.next_frame()
        # Optionally, start a Kalman filter
        if self.kalman:
            self.kalman_filter = KalmanFilter(num_objects=len(self.parts), measurement_noise_x=5, measurement_noise_y=5,
                                              jerk_std=10000, sampling_interval=1/self.fps,
                                              width=self.frames.shape[2], height=self.frames.shape[1])
            starting_points = []
            for part in self.parts:
                x, y = self.points.loc[0, part][['x', 'y']].values
                starting_points.append(np.array([[x, y]]))
            starting_points = np.concatenate(starting_points)
            self.kalman_filter.add_starting_points(starting_points)
            prediction = self.kalman_filter.get_prediction()
        # TODO: add procrustean analysis to get the longitudinal body axis 
        # and superimpose an arrow from the weighted center of mass along the body axis
        if self.procrustean:
            if 'body_centers' not in dir(self):
                # use get_angles_procrustes on the full dataset
                self.body_centers, self.body_angles = get_angles_procrustes(self.points)
                # rotate by pi
                self.body_angles = (self.body_angles + np.pi) % (2 * np.pi)
                # convert body angles to degrees
                self.body_angles = np.rad2deg(self.body_angles)
            else:
                # if this has been run before, remove the previous arrow
                self.image_view.removeItem(self.body_arrow)
            # and add an arrow that will point to centers from tail_joint
            # use the mean distance from the tail joint to the body centers as the length of the arrow
            arrow_length = np.linalg.norm(self.points['tail_joint'][['x', 'y']].values - self.body_centers, axis=1).mean()
            x, y = self.body_centers[0]
            self.body_arrow = pg.ArrowItem(pos=(x, y), angle=self.body_angles[0], tailLen=arrow_length, pen='r', brush='r')
            self.image_view.addItem(self.body_arrow)
        self.timer.start(int(1000 / self.fps))
        self.is_playing = True

    def pause_video(self):
        self.timer.stop()
        self.is_playing = False

    def next_frame(self):
        if self.frames is not None:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.image_view.setCurrentIndex(self.current_frame)
            # update the points overlay
            self.update_points()
            # update the body arrow if procrustean analysis is enabled
            if self.procrustean:
                self.update_arrow()

    def update_fps_label(self, index):
        fps = int(self.fps_dropdown.itemData(index))
        self.fps_label.setText(f"Framerate: {fps} FPS")
        if self.timer.isActive():
            self.timer.start(int(1000 / fps))

    def update_points(self):
        # This is a placeholder for overlaying points on each frame
        # You can use pg.ScatterPlotItem or draw on the frames directly
        # Remove previous scatter plot items if they exist
        if self.points is not None:
            if hasattr(self, 'scatter_items'):
                for item in self.scatter_items:
                    self.image_view.removeItem(item)
            self.scatter_items = []
            coords = self.points.iloc[self.current_frame]
            # if using a kalman filter, add all of the current measurements
            if self.kalman:
                measurements = []
                for part in self.parts:
                    if part in coords:
                        x, y = coords[part][['x', 'y']].values
                        measurements.append(np.array([[x, y]]))
                measurements = np.concatenate(measurements)
                self.kalman_filter.add_measurement(measurements)
                predictions = self.kalman_filter.get_prediction()
            for num, (part, color) in enumerate(zip(self.parts, self.colors)):
                x, y, alpha = self.points.loc[self.current_frame, part][['x', 'y', 'likelihood']].values
                # if Kalman present, add points and use the predicted point
                if self.kalman:
                    x, y = predictions[num]
                new_color = np.copy(color)
                new_color[-1] = 255*alpha  # Set alpha based on likelihood
                pen = pg.mkPen(new_color)
                brush = pg.mkBrush(new_color)
                scatter = pg.ScatterPlotItem([x], [y], pen=pen, brush=brush)
                self.image_view.addItem(scatter)
                self.scatter_items.append(scatter)

    def update_arrow(self):
        if self.procrustean and hasattr(self, 'body_arrow'):
            x, y = self.body_centers[self.current_frame]
            self.body_arrow.setPos(x, y)
            self.body_arrow.setStyle(angle=self.body_angles[self.current_frame])

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


# Allow running as a standalone application
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # You can change these paths or make them command-line arguments
    # video_path = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\PCF_SH\\vid\\fly_1_trial_1_cond1_vidData.mp4"
    # h5_fn = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\PCF_SH\\vid\\fly_1_trial_1_cond1_vidDataDLC_Resnet50_SuperFlyJul29shuffle1_snapshot_best-30.h5"
    video_path = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\BAJA_FH\\vid\\fly_1_trial_1_cond1_vidData.mp4"
    h5_fn = "Z:\\smw\\Daniela\\Polarization_Paper\\Filters_Data\\BAJA_FH\\vid\\fly_1_trial_1_cond1_vidDataDLC_Resnet50_SuperFlyJul29shuffle1_snapshot_best-30.h5"
    preview = TrackerPreview(video_path, h5_path=h5_fn, kalman=True)
    preview.show()
    sys.exit(app.exec())
