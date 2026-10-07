import copy
import deeplabcut
import subprocess
import sys
import os
from wandb import config
import yaml
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSizePolicy, QProgressBar, 
    QFileDialog, QDialog, QComboBox, QLineEdit, QMessageBox, QGridLayout, QCheckBox
)
from PyQt6.QtCore import Qt
from video import Video
import glob
import pprint
import napari

class ExportModelDialog(QDialog):
    """Dialog for exporting a DLC model with various options."""
    
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export DLC Model")
        self.config_path = config_path
        self._init_ui()
        self.resize(550, 500)
    
    def _init_ui(self):
        layout = QGridLayout()
        row = 0
        
        # Shuffle
        layout.addWidget(QLabel("Shuffle:"), row, 0)
        self.shuffle_input = QLineEdit()
        self.shuffle_input.setText("1")
        self.shuffle_input.setToolTip("The shuffle of the model to export (default: 1)")
        layout.addWidget(self.shuffle_input, row, 1)
        row += 1
        
        # Training Set Index
        layout.addWidget(QLabel("Training Set Index:"), row, 0)
        self.trainingsetindex_input = QLineEdit()
        self.trainingsetindex_input.setText("0")
        self.trainingsetindex_input.setToolTip("The index of the training fraction (default: 0)")
        layout.addWidget(self.trainingsetindex_input, row, 1)
        row += 1
        
        # Snapshot Index
        layout.addWidget(QLabel("Snapshot Index:"), row, 0)
        self.snapshotindex_input = QLineEdit()
        self.snapshotindex_input.setPlaceholderText("Leave empty for config.yaml default, or use -1 for latest")
        self.snapshotindex_input.setToolTip("The snapshot index for the weights. Use -1 for latest snapshot.")
        layout.addWidget(self.snapshotindex_input, row, 1)
        row += 1
        
        # Iteration
        layout.addWidget(QLabel("Iteration:"), row, 0)
        self.iteration_input = QLineEdit()
        self.iteration_input.setPlaceholderText("Leave empty to use config.yaml default")
        self.iteration_input.setToolTip("The model iteration (active learning loop) to export")
        layout.addWidget(self.iteration_input, row, 1)
        row += 1
        
        # TFGPUinference checkbox
        self.tfgpu_checkbox = QCheckBox("Use TensorFlow GPU Inference")
        self.tfgpu_checkbox.setChecked(True)
        self.tfgpu_checkbox.setToolTip("Use the TensorFlow inference model. For DeepLabCut-live, set to False.")
        layout.addWidget(self.tfgpu_checkbox, row, 0, 1, 2)
        row += 1
        
        # Overwrite checkbox
        self.overwrite_checkbox = QCheckBox("Overwrite Existing Export")
        self.overwrite_checkbox.setChecked(False)
        layout.addWidget(self.overwrite_checkbox, row, 0, 1, 2)
        row += 1
        
        # Make tar checkbox
        self.make_tar_checkbox = QCheckBox("Compress to TAR File")
        self.make_tar_checkbox.setChecked(True)
        self.make_tar_checkbox.setToolTip("Compress the exported directory to a tar file (required for model zoo)")
        layout.addWidget(self.make_tar_checkbox, row, 0, 1, 2)
        row += 1
        
        # Wipe paths checkbox
        self.wipepaths_checkbox = QCheckBox("Remove Project Paths from Config")
        self.wipepaths_checkbox.setChecked(False)
        self.wipepaths_checkbox.setToolTip("Removes the actual path of your project and init_weights from pose_cfg")
        layout.addWidget(self.wipepaths_checkbox, row, 0, 1, 2)
        row += 1
        
        # Without detector checkbox
        self.without_detector_checkbox = QCheckBox("Export Without Detector (PyTorch only)")
        self.without_detector_checkbox.setChecked(False)
        self.without_detector_checkbox.setToolTip("PyTorch engine only. Exports top-down models without the detector.")
        layout.addWidget(self.without_detector_checkbox, row, 0, 1, 2)
        row += 1
        
        # Buttons
        button_layout = QHBoxLayout()
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(export_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout, row, 0, 1, 2)
        
        self.setLayout(layout)
    
    def get_options(self):
        """Return the selected export options."""
        options = {
            'shuffle': int(self.shuffle_input.text()) if self.shuffle_input.text() else 1,
            'trainingsetindex': int(self.trainingsetindex_input.text()) if self.trainingsetindex_input.text() else 0,
            'snapshotindex': int(self.snapshotindex_input.text()) if self.snapshotindex_input.text() else None,
            'iteration': int(self.iteration_input.text()) if self.iteration_input.text() else None,
            'TFGPUinference': self.tfgpu_checkbox.isChecked(),
            'overwrite': self.overwrite_checkbox.isChecked(),
            'make_tar': self.make_tar_checkbox.isChecked(),
            'wipepaths': self.wipepaths_checkbox.isChecked(),
            'without_detector': self.without_detector_checkbox.isChecked()
        }
        return options

class TrainDLCGui(QWidget):
    def __init__(self, config_path):
        super().__init__()
        self.setWindowTitle("Train DLC Model")
        self.config_path = config_path
        self.videos = []
        self._load_config()
        self._init_ui()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.video_paths = list(self.config.get('video_sets', {}).keys())
        self.videos = [Video(path) for path in self.video_paths]
        self.iteration = self.config['iteration']
        if 'table' in dir(self):
            self._populate_table()

    def _init_ui(self):
        outer_layout = QVBoxLayout(self)

        # Config display at the top
        self.config_label = QLabel()
        self.config_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._update_config_label()
        outer_layout.addWidget(self.config_label)

        main_layout = QHBoxLayout()

        # Left column: buttons/options
        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        left_col.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.add_video_btn = QPushButton("Add Video")
        self.add_video_btn.clicked.connect(self.add_videos)
        left_col.addWidget(self.add_video_btn)

        self.extract_frames_btn = QPushButton("Extract Frames")
        self.extract_frames_btn.clicked.connect(self.extract_frames)
        left_col.addWidget(self.extract_frames_btn)


        self.merge_btn = QPushButton("Merge")
        self.merge_btn.clicked.connect(self.merge)
        left_col.addWidget(self.merge_btn)

        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.train_network)
        left_col.addWidget(self.train_btn)

        self.export_model_btn = QPushButton("Export Model")
        self.export_model_btn.clicked.connect(self.export_model)
        left_col.addWidget(self.export_model_btn)

        self.track_btn = QPushButton("Track")
        self.track_btn.clicked.connect(self.track_files)
        left_col.addWidget(self.track_btn)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)
        left_col.addWidget(self.preview_btn)

        self.dlc_gui_btn = QPushButton("DLC GUI")
        self.dlc_gui_btn.clicked.connect(self.dlc_gui)
        left_col.addWidget(self.dlc_gui_btn)

        left_col.addStretch(1)

        left_frame = QFrame()
        left_frame.setLayout(left_col)
        left_frame.setFixedWidth(180)
        left_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        # Main table: video info
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Video Path", "Extracted", "Labeled", "Model #"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._populate_table()

        main_layout.addWidget(left_frame)
        main_layout.addWidget(self.table)
        outer_layout.addLayout(main_layout)

        # Progress bar and notification label at the bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        progress_label = QLabel("Progress:")
        progress_bar_layout = QHBoxLayout()
        progress_bar_layout.addWidget(progress_label)
        progress_bar_layout.addWidget(self.progress_bar)
        progress_bar_layout.setStretch(0, 0)
        progress_bar_layout.setStretch(1, 1)

        self.notification_label = QLabel()
        self.notification_label.setText("")
        self.notification_label.setStyleSheet("color: white; padding: 4px;")

        outer_layout.addWidget(self.notification_label)
        outer_layout.addLayout(progress_bar_layout)
        self.setLayout(outer_layout)

    def set_notification(self, text: str):
        self.notification_label.setText(text)

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def add_videos(self):
        # open a file selection dialogue to select a number of videos to add to the videos folder
        video_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if video_files:
            # use deeplabcut.add_new_videos
            ret = deeplabcut.add_new_videos(self.config_path, video_files, copy_videos=True, extract_frames=True)
        self._load_config()  # Always reload config when any function is called
        self.notification_label.setText("Videos added successfully.")

    def extract_frames(self):
        # Extract frames for all highlighted videos files
        video_files = [vid.path for vid in self.get_selected_videos()]
        if video_files:
            deeplabcut.extract_frames(self.config_path, videos_list=video_files, userfeedback=False, mode='automatic')
        self._load_config()
        self.notification_label.setText("Frames were extracted. See console for details.")

    def get_selected_videos(self):
        selected = []
        for idx in self.table.selectionModel().selectedRows():
            item = self.table.item(idx.row(), 0)
            orig_index = item.data(Qt.ItemDataRole.UserRole)
            selected.append(self.videos[orig_index])
        return selected
        
    def dlc_gui(self):
        self.notification_label.setText("Opening DeepLabCut GUI...")
        # open the DeepLabCut GUI for labeling and other features
        import subprocess, sys
        subprocess.Popen([sys.executable, "-m", "deeplabcut", "gui", self.config_path]).wait()
        self._load_config()
        self.notification_label.setText("DeepLabCut GUI closed.")

    def merge(self):
        # we can only merge if all of the extracted videos are also labeled
        extracted_videos = [v for v in self.videos if v.is_extracted()]
        old_iter = copy.copy(self.iteration)
        if all(v.is_labeled() for v in extracted_videos):
            # use deeplabcut.merge_datasets to merge the newly added and labeled videos
            deeplabcut.merge_datasets(self.config_path)
            self._load_config()
            iteration_changed = (self.iteration > old_iter)
            if iteration_changed:
                self.notification_label.setText(f"Datasets merged successfully for iteration #{self.iteration}.")
            else:
                self.notification_label.setText("Merge failed.")
        else:
            self.notification_label.setText("Not all extracted videos are labeled.")

    def train_network(self):
        import deeplabcut
        self.notification_label.setText("Creating training dataset...")
        deeplabcut.create_training_dataset(self.config_path)
        self.notification_label.setText("Training network...")
        deeplabcut.train_network(self.config_path)
        self.notification_label.setText("Evaluating network...")
        deeplabcut.evaluate_network(self.config_path)
        # todo: print the evaluation results
        self.notification_label.setText("Training and evaluation complete.")

    def export_model(self):
        """Open export model dialog and execute export."""
        dialog = ExportModelDialog(self.config_path, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            options = dialog.get_options()
            self._execute_export(options)
    
    def _execute_export(self, options):
        """Execute the model export with the provided options."""
        try:
            self.notification_label.setText("Exporting model...")
            self.set_progress(0)
            
            # Call export_model with all the provided options
            deeplabcut.export_model(
                self.config_path,
                shuffle=options.get('shuffle', 1),
                trainingsetindex=options.get('trainingsetindex', 0),
                snapshotindex=options.get('snapshotindex'),
                iteration=options.get('iteration'),
                TFGPUinference=options.get('TFGPUinference', True),
                overwrite=options.get('overwrite', False),
                make_tar=options.get('make_tar', True),
                wipepaths=options.get('wipepaths', False),
                without_detector=options.get('without_detector', False)
            )
            
            self.set_progress(100)
            self.notification_label.setText("Model exported successfully!")
        except Exception as e:
            self.notification_label.setText(f"Export failed: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export model:\n{str(e)}")

    def track_files(self):
        # Example: run DLCProcessWorker for all videos in config
        from pyqt_gui_shell import DLCProcessWorker
        save_as_csv = False  # or get from UI
        make_labeled_video = False  # or get from UI
        pcutoff = 0.6  # or get from UI
        overwrite = False
        batch_process = False
        # get every selected row
        selected_files = [v.path for v in self.get_selected_videos()]
        processed_files = []
        self.worker = DLCProcessWorker(
            selected_files,
            processed_files,
            self.config_path,
            save_as_csv,
            make_labeled_video,
            pcutoff,
            overwrite,
            batch_process=batch_process
        )
        # Connect signals for progress and notification
        self.set_progress(0)
        self.worker.update_notification.connect(self.set_notification)
        self.worker.progress_update.connect(self.set_progress)
        self.worker.finished.connect(self.tracking_finished)
        self.worker.start()

    def tracking_finished(self):
        self._load_config()
        # self._populate_table()

    def preview(self):
        # Preview the selected video in the table
        selected = self.get_selected_videos()
        if len(selected) == 0:
            self.set_notification("No video selected for preview.")
            return
        video = selected[0]
        from tracker_preview import TrackerPreview
        abs_path = video.path
        points_fn = None
        # Try to find the latest .h5 for this video
        base_path = os.path.splitext(abs_path)[0]
        h5_files = glob.glob(f"{base_path}*.h5")
        if h5_files:
            h5_files.sort(key=os.path.getmtime)
            points_fn = h5_files[-1]
        # use the parts list from the config if available
        self.parts = None
        if 'bodyparts' in self.config:
            self.parts = self.config['bodyparts']
        self.tracker_preview = TrackerPreview(abs_path, points_fn, parts=self.parts)
        self.tracker_preview.show()

    def _update_config_label(self):
        pretty = pprint.pformat(self.iteration, width=120, compact=True)
        self.config_label.setText(f"Iteration <b>{pretty}</b>")

    def _populate_table(self):
        self.table.setRowCount(len(self.videos))
        for i, video in enumerate(self.videos):
            basename = os.path.basename(video.path)
            item = QTableWidgetItem(basename)
            item.setToolTip(video.path)
            item.setData(Qt.ItemDataRole.UserRole, i)  # Store original index
            self.table.setItem(i, 0, item)
            # Extracted (checkmark if extracted)
            extracted_item = QTableWidgetItem()
            extracted_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if video.is_extracted():
                extracted_item.setText("✓")
            else:
                extracted_item.setText("")
            self.table.setItem(i, 1, extracted_item)
            # Labeled (checkmark if labeled)
            labeled_item = QTableWidgetItem()
            labeled_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if video.is_labeled():
                labeled_item.setText("✓")
            else:
                labeled_item.setText("")
            self.table.setItem(i, 2, labeled_item)

            # Model number
            model_num = video.get_model_num()
            model_item = QTableWidgetItem(str(model_num) if model_num is not None else "")
            self.table.setItem(i, 3, model_item)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_dlc_gui.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    # config_path = "C:\\Users\\johnp\\OneDrive\\Desktop\\smellovision\\visuo_olfactory_tracking_w_bar_exp\\dlc\\super_fly_videos\\SuperFly-Pablo-2025-07-29\\config.yaml"
    app = QApplication(sys.argv)
    window = TrainDLCGui(config_path)
    window.show()
    sys.exit(app.exec())
