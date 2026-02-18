#! C:\Users\johnp\conda\envs\DEEPLABCUT\python.exe

# === PyQt6 Threading Imports ===
from glob import glob
from PyQt6.QtCore import QThread, pyqtSignal


class DLCProcessWorker(QThread):
    file_processed = pyqtSignal(str, bool, str)  # filename, success, message
    update_lists = pyqtSignal(list, list)
    update_notification = pyqtSignal(str)
    update_file_list_signal = pyqtSignal()
    progress_update = pyqtSignal(int)  # percentage of files processed

    def __init__(self, imported_files, processed_files, config_path, save_as_csv, make_labeled_video, pcutoff, overwrite, parent=None, batch_process=False):
        super().__init__(parent)
        self.batch_process = batch_process
        self.imported_files = imported_files.copy()
        self.processed_files = processed_files.copy()
        self.config_path = config_path
        self.save_as_csv = save_as_csv
        self.make_labeled_video = make_labeled_video
        self.pcutoff = pcutoff
        self.overwrite = overwrite
        self._abort = False

    def run(self):
        self._abort = False
        import subprocess, sys, os, time, traceback
        worker_script = os.path.join(os.path.dirname(__file__), "dlc_worker.py")
        files_to_process = self.imported_files[:]
        failed_files = []
        attempt = 1
        total_files = len(files_to_process)
        while files_to_process and not self._abort:
            # Call analyze_videos in a subprocess for all files in this batch
            if self.batch_process:
                self.update_notification.emit(f"Batch attempt {attempt}: analyzing {len(files_to_process)} files...")
                result = subprocess.run([
                    sys.executable, worker_script, self.config_path, *files_to_process, "--batch", str(self.save_as_csv), str(self.pcutoff), str(self.overwrite)
                ], capture_output=True, text=True)
                # Parse output for failed files (assume worker prints a line for each failed file)
                failed_files = []
                if result.returncode == 0:
                    # All succeeded
                    for idx, fn in enumerate(files_to_process):
                        msg = f"Processed: {fn}"
                        self.imported_files.remove(fn)
                        self.processed_files.append(fn)
                        self.file_processed.emit(fn, True, msg)
                        self.update_lists.emit(self.imported_files, self.processed_files)
                        self.update_notification.emit(msg)
                        self.update_file_list_signal.emit()
                        percent = int(100 * len(self.processed_files) / total_files) if total_files > 0 else 100
                        self.progress_update.emit(percent)
                        time.sleep(0.05)
                    break
                else:
                    # Try to parse failed files from stderr (expect lines like 'FAILED: <filename>')
                    for line in result.stderr.splitlines():
                        if line.startswith("FAILED: "):
                            failed_files.append(line[len("FAILED: "):].strip())
                    succeeded = [f for f in files_to_process if f not in failed_files]
                    for fn in succeeded:
                        msg = f"Processed: {fn}"
                        self.imported_files.remove(fn)
                        self.processed_files.append(fn)
                        self.file_processed.emit(fn, True, msg)
                        self.update_lists.emit(self.imported_files, self.processed_files)
                        self.update_notification.emit(msg)
                        self.update_file_list_signal.emit()
                        time.sleep(0.05)
                    for fn in failed_files:
                        msg = f"Error processing {fn}: see log. Will retry."
                        self.file_processed.emit(fn, False, msg)
                        self.update_notification.emit(msg)
                if not failed_files or len(failed_files) == len(files_to_process):
                    # No progress or all failed, stop to avoid infinite loop
                    self.update_notification.emit("No progress made or all files failed. Stopping batch retries.")
                    break
                files_to_process = failed_files
                attempt += 1
                # After all analysis, create labeled videos if requested
                if self.make_labeled_video:
                    for fn in self.processed_files:
                        if self._abort:
                            break
                        try:
                            import deeplabcut
                            deeplabcut.create_labeled_video(
                                self.config_path,
                                [fn],
                                pcutoff=self.pcutoff,
                                overwrite=self.overwrite,
                                fastmode=True,
                                videotype=os.path.splitext(fn)[1],
                            )
                            msg = f"Labeled video created: {fn}"
                            self.update_notification.emit(msg)
                        except Exception as e:
                            tb = traceback.format_exc()
                            msg = f"Error creating labeled video for {fn}: {e}\n{tb}"
                            self.update_notification.emit(msg)
            else:
                # Single file mode
                self.update_notification.emit(f"Processing {len(files_to_process)} files...")
                for idx, video_file in enumerate(files_to_process):
                    if not self._abort:
                        output_dir = os.path.dirname(video_file)
                        result = subprocess.run([
                            sys.executable, worker_script, self.config_path, video_file, output_dir, 
                            str(self.save_as_csv).lower(), str(self.make_labeled_video).lower(), str(self.pcutoff).lower(), str(self.overwrite).lower()
                        ], capture_output=True, text=True)
                        if result.returncode == 0:
                            if video_file in self.imported_files:
                                self.imported_files.remove(video_file)
                            self.processed_files.append(video_file)
                            self.update_notification.emit(f"Processed {video_file}")
                            self.update_lists.emit(self.imported_files, self.processed_files)
                            percent = int(100 * len(self.processed_files) / total_files) if total_files > 0 else 100
                            self.progress_update.emit(percent)
                        else:
                            self.update_notification.emit(f"Error processing {video_file}: see log.")

    def abort(self):
        self._abort = True

# Import TrackerPreview for video previewing
from tracker_preview import TrackerPreview

# === Standard Library Imports ===
import os
import sys
import time

# === Third-Party Imports ===
import numpy as np
import scipy.io
import skvideo.io

# === PyQt6 Imports ===
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QDoubleValidator, QAction
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
    QFileDialog, QComboBox, QLabel, QLineEdit, QTreeView, QListWidget, QCheckBox, QGroupBox,
    QGridLayout, QProgressBar, QDialog, QMessageBox, QMenu
)

# === Optional/Conditional Imports ===
# (deeplabcut is imported inside the function to avoid import errors if not installed)

class MainWindow(QWidget):
    """
    MainWindow is a PyQt6 GUI for managing video file import, processing, and export options
    for DeepLabCut workflows. It features:
    - A fixed-width left column with buttons for loading config, importing files/folders, tracking, and closing.
    - Dropdown for video format selection.
    - Export options with checkboxes.
    - Two main columns: imported files (left) and processed files (right), each with a QListWidget.
    - Progress bars for import and video processing, with notification area.
    - The 'track' button is only enabled after a config.yaml is loaded.
    """
    def __init__(self):
        """
        Initialize the main window, attributes, and UI.
        """
        super().__init__()
        self.imported_files = []  # List to keep track of files to process
        self.processed_files = []  # List to keep track of processed files
        self.config_loaded = False  # Track if config.yaml is loaded
        self.config_path = None
        self.batch_process = False
        self.init_ui()
        self.init_menu()
        self.project = None

    def init_menu(self):
        """Set up the menu bar and toolbar actions."""
        from PyQt6.QtWidgets import QMenuBar, QMenu
        from PyQt6.QtGui import QAction
        self.menu_bar = QMenuBar(self)

        # File menu with Save As
        self.file_menu = QMenu("File", self)
        self.menu_bar.addMenu(self.file_menu)

        self.file_actions = {}
        for lbl, method in zip(['Save As', 'Save', 'Load'], [self.saveas_project, self.save_project, self.load_project]):
            action = QAction(lbl, self)
            action.triggered.connect(method)
            self.file_menu.addAction(action)
            self.file_actions[lbl] = action

        self.file_actions['Save'].setEnabled(False)

        # Main menu
        self.tools_menu = QMenu("Options", self)
        self.menu_bar.addMenu(self.tools_menu)

        # Train DLC Model action (combines add videos and label unlabeled videos)
        self.train_dlc_action = QAction("Train DLC model", self)
        self.train_dlc_action.setEnabled(False)  # Disabled by default
        self.train_dlc_action.triggered.connect(self.launch_train_dlc)
        self.tools_menu.addAction(self.train_dlc_action)

        # Add more actions as needed

        # Place the menu bar at the top of the layout
        if hasattr(self, 'layout') and self.layout() is not None:
            self.layout().setMenuBar(self.menu_bar)
        else:
            # If not using QMainWindow, add manually to the top
            main_layout = self.layout() or QVBoxLayout(self)
            main_layout.setMenuBar(self.menu_bar)
            self.setLayout(main_layout)

    def saveas_project(self, **kwargs):
        """Save imported and processed files to a user-defined CSV."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import csv
        file_path = kwargs.get("file_path", None)
        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Type", "File"])
                    for f in self.imported_files:
                        writer.writerow(["imported", f])
                    for f in self.processed_files:
                        writer.writerow(["processed", f])
                self.project = file_path
                self.file_actions['Save'].setEnabled(True)
                self.set_notification(f"Project saved as: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save project: {e}")
        else:
            self.set_notification("Save As cancelled.")

    def save_project(self):
        # this button should only be enabled if a project has been previously saved or loaded
        if self.project:
            self.saveas_project(self.project)
        else:
            self.set_notification("Failed to save. No project loaded.")

    def load_project(self, **kwargs):
        file_path = kwargs.get("file_path", None)
        # choose the file
        if file_path is None:
            # use file dialog to choose the right .csv
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "CSV Files (*.csv)")
        if file_path:
            self.project = file_path
            # Load the project data from the CSV
            with open(file_path, 'r', newline='') as csvfile: file_list = csvfile.readlines()
            # we have to clean up the file_list
            file_list = file_list[1:]
            file_list = [f.replace("\r\n", "").split(',') for f in file_list]
            categories, paths = zip(*file_list) if file_list else ([], [])
            self.set_notification(f"{file_path} loaded {len(paths)} files.")
            if len(paths) > 0:
                self.import_files(files=paths, categories=categories)
            self.file_actions['Save'].setEnabled(True)
        else:
            self.set_notification("Load Project cancelled.")

    def launch_train_dlc(self):
        """Launch the Train DLC Model GUI as a subprocess (placeholder)."""
        import subprocess, sys
        if self.config_path:
            subprocess.Popen([sys.executable, 'train_dlc_gui.py', self.config_path])
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Config Loaded", "Please load a config.yaml file first.")

    def init_ui(self):
        """
        Set up all widgets, layouts, and connect signals for the GUI.
        """
        # Section 1: Fixed-width left column for buttons and dropdown
        button_height = 40
        button_min_width = 100
        button_names = ["load config", "import files", "import folder", "convert .mat files", "remove selected", "track", "abort", "close"]
        self.buttons = []
        button_vlayout = QVBoxLayout()
        button_vlayout.setSpacing(10)
        for idx, name in enumerate(button_names):
            btn = QPushButton(name)
            btn.setMinimumHeight(button_height)
            btn.setMinimumWidth(button_min_width)
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            button_vlayout.addWidget(btn)
            self.buttons.append(btn)
        # Connect buttons to methods
        self.buttons[0].clicked.connect(self.load_config)
        self.buttons[1].clicked.connect(self.import_files)
        self.buttons[2].clicked.connect(self.import_folder)
        self.buttons[3].clicked.connect(self.import_mat_video)
        self.buttons[4].clicked.connect(self.remove_selected_files)
        self.buttons[5].clicked.connect(self.track_files)
        self.buttons[5].setEnabled(False)  # Track button disabled until config is loaded
        self.buttons[6].clicked.connect(self.abort_operation)
        self.buttons[7].clicked.connect(self.close_app)
        self.abort = False
        # Track button disabled until config is loaded
        # Add Preview button at the bottom
        self.preview_button = QPushButton("Preview")
        self.preview_button.setMinimumHeight(button_height)
        self.preview_button.setMinimumWidth(button_min_width)
        self.preview_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.preview_button.clicked.connect(self.preview)
        self.preview_button.setEnabled(True)
        button_vlayout.addWidget(self.preview_button)

        # Add dropdown for video formats
        self.format_dropdown = QComboBox()
        self.format_dropdown.setMinimumHeight(button_height)
        self.format_dropdown.setMinimumWidth(button_min_width)
        self.formats = [".mp4", ".avi", ".mov", ".mpeg", ".mat"]
        self.format_dropdown.addItems(self.formats)
        format_label = QLabel("Format:")
        format_label.setStyleSheet("color: white;")
        button_vlayout.addWidget(format_label)
        button_vlayout.addWidget(self.format_dropdown)

        # Export options table with checkboxes
        export_options = [
            ('.csv per file', False),
            ('.h5 per file', True),
            ('.h5 combined', True),
            ('.mat combined', False),
            ('.mp4 preview', False),
            ('rerun', False),
            ('batch analysis', False)
        ]
        self.export_checkboxes = {}
        export_group = QGroupBox("Export options:")
        export_group.setStyleSheet("color: white; border: none; margin-top: 12px;")
        export_layout = QGridLayout()
        export_layout.setContentsMargins(0, 12, 0, 0)
        export_layout.setSpacing(0)  # No extra space between checkboxes
        for i, (label, checked) in enumerate(export_options):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.setStyleSheet("color: white;")
            export_layout.addWidget(cb, i, 0)
            self.export_checkboxes[label] = cb

        # Add pcutoff slider and input at the bottom of export options
        pcutoff_layout = QHBoxLayout()
        pcutoff_label = QLabel('p-cutoff:')
        pcutoff_label.setStyleSheet("color: white; margin-right: 4px;")
        self.pcutoff_edit = QLineEdit()
        self.pcutoff_edit.setText("0.6")
        self.pcutoff = 0.6
        validator = QDoubleValidator(0.0, 1.0, 2)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.pcutoff_edit.setValidator(validator)
        self.pcutoff_edit.editingFinished.connect(self._on_pcutoff_edit)
        pcutoff_layout.addWidget(pcutoff_label)
        pcutoff_layout.addWidget(self.pcutoff_edit)
        export_layout.addLayout(pcutoff_layout, len(export_options)+1, 0)
        export_group.setLayout(export_layout)

        button_vlayout.addWidget(export_group)
        button_vlayout.addStretch(1)

        # Restore button_column frame and layout
        self.button_column = QFrame()
        self.button_column.setFixedWidth(180)
        button_column_layout = QVBoxLayout(self.button_column)
        button_column_layout.setContentsMargins(8, 8, 8, 8)
        button_column_layout.addLayout(button_vlayout)

        self.left_frame = QFrame()
        self.left_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_frame.setStyleSheet("background-color: #333333;")
        self.left_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(self.left_frame)
        self.file_tree_view = QTreeView(self.left_frame)
        self.file_tree_view.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        self.file_tree_model = QStandardItemModel()
        self.file_tree_model.setHorizontalHeaderLabels(["Imported Files"])
        self.file_tree_view.setModel(self.file_tree_model)
        self.file_tree_view.setHeaderHidden(False)
        left_layout.addWidget(self.file_tree_view)

        # Right main column: QFrame with file list (processed files)
        self.right_frame = QFrame()
        self.right_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.right_frame.setStyleSheet("background-color: #333333;")
        self.right_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(self.right_frame)
        self.processed_tree_view = QTreeView(self.right_frame)
        self.processed_tree_view.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        self.processed_tree_model = QStandardItemModel()
        self.processed_tree_model.setHorizontalHeaderLabels(["Processed Files"])
        self.processed_tree_view.setModel(self.processed_tree_model)
        self.processed_tree_view.setHeaderHidden(False)
        right_layout.addWidget(self.processed_tree_view)
        # Enable preview button only if a processed video is selected
        self.processed_tree_view.selectionModel().selectionChanged.connect(self.update_preview_button_state)
        self.processed_tree_model.setHorizontalHeaderLabels(["Processed Files"])
        self.processed_tree_view.setModel(self.processed_tree_model)
        self.processed_tree_view.setHeaderHidden(False)
        right_layout.addWidget(self.processed_tree_view)
        # Enable preview button only if a processed video is selected
        self.processed_tree_view.selectionModel().selectionChanged.connect(self.update_preview_button_state)

        # # Right main column: QFrame with file list (processed files)
        # self.right_frame = QFrame()
        # self.right_frame.setFrameShape(QFrame.Shape.StyledPanel)
        # self.right_frame.setStyleSheet("background-color: #333333;")
        # self.right_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # right_layout = QVBoxLayout(self.right_frame)
        # self.processed_tree_view = QTreeView(self.right_frame)
        # self.processed_tree_view.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        # self.processed_tree_model = QStandardItemModel()
        # self.processed_tree_model.setHorizontalHeaderLabels(["Processed Files"])
        # self.processed_tree_view.setModel(self.processed_tree_model)
        # self.processed_tree_view.setHeaderHidden(False)
        # right_layout.addWidget(self.processed_tree_view)
        # Enable preview button only if a processed video is selected
        self.processed_tree_view.selectionModel().selectionChanged.connect(self.update_preview_button_state)

        # Combine the two main columns (left_frame, right_frame)
        main_columns_layout = QHBoxLayout()
        main_columns_layout.addWidget(self.left_frame)
        main_columns_layout.addWidget(self.right_frame)
        main_columns_layout.setStretch(0, 1)
        main_columns_layout.setStretch(1, 1)

        main_columns_widget = QWidget()
        main_columns_widget.setLayout(main_columns_layout)
        main_columns_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Top-level layout: button column + main columns
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.button_column)
        top_layout.addWidget(main_columns_widget)
        top_layout.setStretch(0, 0)  # Button column: fixed
        top_layout.setStretch(1, 1)  # Main columns: expand

        # Progress bars and notification label at the bottom
        self.import_progress = QProgressBar()
        self.import_progress.setMinimum(0)
        self.import_progress.setMaximum(100)
        self.import_progress.setValue(0)
        self.import_progress.setTextVisible(True)
        self.import_progress.setFormat("%p%")
        import_label = QLabel("Import Progress:")
        import_label.setStyleSheet("color: white; padding-right: 8px;")
        import_bar_layout = QHBoxLayout()
        import_bar_layout.addWidget(import_label)
        import_bar_layout.addWidget(self.import_progress)
        import_bar_layout.setStretch(0, 0)
        import_bar_layout.setStretch(1, 1)

        self.video_progress = QProgressBar()
        self.video_progress.setMinimum(0)
        self.video_progress.setMaximum(100)
        self.video_progress.setValue(0)
        self.video_progress.setTextVisible(True)
        self.video_progress.setFormat("%p%")
        video_label = QLabel("Video Progress:")
        video_label.setStyleSheet("color: white; padding-right: 8px;")
        video_bar_layout = QHBoxLayout()
        video_bar_layout.addWidget(video_label)
        video_bar_layout.addWidget(self.video_progress)
        video_bar_layout.setStretch(0, 0)
        video_bar_layout.setStretch(1, 1)

        self.notification_label = QLabel()
        self.notification_label.setText("")
        self.notification_label.setStyleSheet("color: white; padding: 4px;")

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.notification_label)
        main_layout.addLayout(import_bar_layout)
        main_layout.addLayout(video_bar_layout)
        main_layout.setStretch(0, 1)  # Top layout: expand
        main_layout.setStretch(1, 0)  # Notification label: no stretch
        main_layout.setStretch(2, 0)  # Import progress bar: no stretch
        main_layout.setStretch(3, 0)  # Video progress bar: no stretch
        self.setLayout(main_layout)
        self.setWindowTitle("PyQt6 GUI Shell")
        self.resize(1100, 700)

    def abort_operation(self):
        """Set abort flag to True to interrupt processing."""
        self.set_notification("Abort requested. Waiting for current operation to stop...")
        self.abort = True
        # kill the worker thread if it's running
        if hasattr(self, 'worker'):
            if self.worker.isRunning():
                self.worker.abort()
                self.worker.wait()
                self.set_notification("Current operation aborted.")

    def remove_selected_files(self):
        """
        Remove selected files from the imported files list and update the display.
        """
        for tree, tree_model, files, lbl in zip(
            [self.file_tree_view, self.processed_tree_view],
            [self.file_tree_model, self.processed_tree_model],
            [self.imported_files, self.processed_files],
            ['imported', 'processed']
        ):
            selected_indexes = tree.selectedIndexes()
            if selected_indexes:    
                num_selected = len(selected_indexes)
                self.set_notification(f"{num_selected} {lbl} files selected to remove.")
                to_remove = set()
                for index in selected_indexes:
                    if tree_model.hasChildren(index):
                        continue
                    parts = []
                    idx = index
                    while idx.isValid():
                        parts.insert(0, idx.data())
                        idx = idx.parent()
                    if parts and parts[0].endswith(os.sep):
                        parts[0] = parts[0][:-1]
                    rel_path = os.sep.join(parts)
                    # if self.config_loaded:
                    #     abs_path = os.path.abspath(os.sep.join([self.config_directory, rel_path]))
                    # else:
                    abs_path = os.path.abspath(rel_path)
                    to_remove.add(abs_path)
                for fname in to_remove:
                    try:
                        files.remove(fname)
                    except Exception as e:
                        breakpoint()
            # self.imported_files = [f for f in self.imported_files if os.path.abspath(f) not in to_remove]
        self.update_file_list()

    def _on_pcutoff_edit(self):
        """
        Update the pcutoff attribute from the QLineEdit when editing is finished.
        """
        text = self.pcutoff_edit.text()
        try:
            value = float(text)
            if 0.0 <= value <= 1.0:
                self.pcutoff = value
            else:
                self.pcutoff_edit.setText(str(self.pcutoff))
        except ValueError:
            self.pcutoff_edit.setText(str(self.pcutoff))

    def option_checked(self, label):
        """
        Returns True if the export option checkbox with the given label is checked.
        Example: self.option_checked('.h5 combined')
        """
        cb = self.export_checkboxes.get(label)
        return cb.isChecked() if cb else False

    def preview(self):
        """
        Open a TrackerPreview window to show the selected video. If from imported list, show video only.
        If from processed list, show video with tracked points (if available).
        """
        # Check imported (left) and processed (right) tree views for selection
        imported_selected = self.file_tree_view.selectedIndexes()
        processed_selected = self.processed_tree_view.selectedIndexes()
        selected = None
        source = None
        # Prefer processed if both are selected
        if processed_selected:
            selected = processed_selected
            source = 'processed'
            files = self.processed_files
        elif imported_selected:
            selected = imported_selected
            source = 'imported'
            files = self.imported_files
        else:
            self.set_notification("No video selected for preview.")
            return
        # Only use the first selected leaf node (file)
        for index in selected:
            model = self.processed_tree_model if source == 'processed' else self.file_tree_model
            if model.hasChildren(index):
                continue
            # # Reconstruct the relative path from the tree
            # parts = []
            # idx = index
            # while idx.isValid():
            #     parts.insert(0, idx.data())
            #     idx = idx.parent()
            # if parts and parts[0].endswith(os.sep):
            #     parts[0] = parts[0][:-1]
            # rel_path = os.path.join(*parts)
            # if self.config_loaded:
            #     abs_path = os.path.abspath(os.path.join(self.config_directory, rel_path))
            # else:
            #     abs_path = os.path.abspath(rel_path)
            # points_fn = None
            substr = index.data()
            if not os.path.isdir(substr) and substr.endswith(os.sep):
                # try removing the last backslashes
                substr = substr[:-1]
            abs_path = [fn for fn in files if substr in fn or fn == substr][0]
            points_fn = None
            if source == 'processed':
                # todo: split the file and look for a file starting with everything before the file extension and ending in .h5
                # so something line base_path * .h5
                base_path = os.path.splitext(abs_path)[0]
                h5_files = glob(f"{base_path}*.h5")
                if len(h5_files) > 0:
                    # take the most recent file
                    h5_files.sort(key=os.path.getmtime)
                    points_fn = h5_files[-1]
            # if the config is loaded, grab the list of parts in the right order
            self.parts = None
            if self.config_loaded:
                # import the config file and get the list of parts
                import yaml
                with open(self.config_path, 'r') as f: 
                    self.config = yaml.safe_load(f)
                    if 'bodyparts' in self.config:
                        self.parts = self.config['bodyparts']
            self.tracker_preview = TrackerPreview(abs_path, points_fn, parts=self.parts)
            self.tracker_preview.show()
            self.set_notification(f"Previewing: {abs_path}")
            return
        self.set_notification("No video file selected for preview.")


    def load_config(self):
        """
        Open a file dialog to select a config.yaml file. Sets config_loaded to True and enables
        the track button if a file is selected. Updates notification area with status.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select config.yaml", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            self.config_path = file_path
            self.config_directory = os.path.dirname(file_path)
            self.config_loaded = True
            self.set_notification(f"Loaded config: {file_path}")
            self.buttons[3].setEnabled(True)
            self.buttons[5].setEnabled(True)  # Enable track button
            self.train_dlc_action.setEnabled(True)  # Enable Train DLC Model menu option
        else:
            self.set_notification("No config file selected.")

    def set_import_progress(self, value: int):
        """
        Set the value of the import progress bar.
        """
        self.import_progress.setValue(value)

    def set_video_progress(self, value: int):
        """
        Set the value of the video progress bar.
        """
        self.video_progress.setValue(value)

    def set_notification(self, text: str):
        """
        Update the notification label with the provided text.
        """
        self.notification_label.setText(text)

    def import_file(self, fname, other_fns=None, category=None):
        """Import an individual file."""
        selected_exts = self.get_selected_formats()
        if any(fname.lower().endswith(ext) for ext in selected_exts) and 'labeled' not in os.path.basename(fname):
            if category is None:
                # check if the video has already been analyzed
                prefix = fname.split('.')[0] + 'DLC_Resnet'
                # check if any file exists with prefix * .h5
                if other_fns is not None:
                    h5_files = [fn for fn in other_fns if fn.startswith(prefix) and fn.endswith('.h5')]
                else:
                    h5_files = glob(f"{prefix}*.h5")
                analyzed = len(h5_files) > 0
            else:
                analyzed = category == 'processed'
            if not analyzed or self.option_checked('rerun'):
                # Only add if not already imported or if rerun is selected
                    self.imported_files.append(fname)
            if analyzed and not self.option_checked('rerun'):
                # add to the processed list
                self.processed_files.append(fname)
    
    def import_files(self, **kwargs):
        """
        Open a file dialog to select multiple video or .mat files and add them to the imported files list.
        Updates the imported files QListWidget and progress bar.
        """
        files = kwargs.get("files", None)
        categories = kwargs.get("categories", None)
        if files is None:
            selected_exts = self.formats
            file_types = "Video/Mat Files ({});;All Files (*)".format(' '.join(['*'+ext for ext in selected_exts]))
            files, _ = QFileDialog.getOpenFileNames(self, "Select Video or .mat Files", "", file_types)
            if os.sep == '\\':
                files = [f.replace('/', '\\') for f in files]
        if categories is None:
            categories = [None] * len(files)
        if files:
            for fname, category in zip(files, categories):
                self.import_file(fname, category=category)
        self.update_file_list()

    def import_folder(self):
        """
        Open a folder dialog to select a directory and import all video or .mat files of the selected format.
        Updates the imported files QListWidget and progress bar.
        """
        folder = os.path.abspath(QFileDialog.getExistingDirectory(self, "Select Folder"))
        if folder:
            fns = os.listdir(folder)
            fns = [os.sep.join([folder, fn]) for fn in fns]
            for fname in fns:
                self.import_file(fname, other_fns=fns)
        self.update_file_list()

    def import_mat_video(self):
        # Reset abort flag at the start of the operation
        self.abort = False
        """
        Convert all imported .mat files to .mp4 using scipy.io.loadmat and skvideo.io.vwrite.
        Prompts user to select which keys in the .mat files correspond to video matrices.
        """
        mat_files = [f for f in self.imported_files if f.lower().endswith('.mat')]
        if not mat_files:
            self.set_notification("No .mat files in imported files.")
            return
        # Gather all unique keys from all mat files
        all_keys = set()
        # todo: use a random sample of the files for getting the keys
        sample_files = np.random.choice(mat_files, size=min(len(mat_files), 10), replace=False)
        for f in sample_files:
            try:
                data = scipy.io.loadmat(f)
                all_keys.update(data.keys())
            except Exception as e:
                self.set_notification(f"Error reading {f}: {e}")
        all_keys = sorted([k for k in all_keys if not k.startswith('__')])
        if not all_keys:
            self.set_notification("No valid keys found in .mat files.")
            return
        # Dialog for key selection
        class KeySelectDialog(QDialog):
            def __init__(self, keys, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Select Video Keys")
                self.selected_keys = []
                layout = QVBoxLayout()
                layout.addWidget(QLabel("Select keys that correspond to video matrices:"))
                self.checkboxes = []
                for k in keys:
                    cb = QCheckBox(k)
                    layout.addWidget(cb)
                    self.checkboxes.append(cb)
                ok_btn = QPushButton("OK")
                ok_btn.clicked.connect(self.accept)
                layout.addWidget(ok_btn)
                self.setLayout(layout)
            def get_selected(self):
                return [cb.text() for cb in self.checkboxes if cb.isChecked()]
        dlg = KeySelectDialog(all_keys, self)
        if not dlg.exec():
            self.set_notification("Conversion cancelled.")
            return
        selected_keys = dlg.get_selected()
        if not selected_keys:
            self.set_notification("No keys selected for conversion.")
            return
        # Ask user if they want to use GPU (ffmpeg h264_nvenc) or CPU (default)
        from PyQt6.QtWidgets import QMessageBox
        use_gpu = False
        msg = QMessageBox(self)
        msg.setWindowTitle("Use GPU Acceleration?")
        msg.setText("Use GPU-accelerated ffmpeg (h264_nvenc) for .mp4 encoding? (Requires NVIDIA GPU and compatible ffmpeg)")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Yes:
            use_gpu = True
        # Convert each mat file for each selected key
        for num, f in enumerate(mat_files):
            if self.abort:
                self.set_notification("Conversion aborted by user.")
                break
            try:
                data = scipy.io.loadmat(f)
                for key in selected_keys:
                    if self.abort:
                        self.set_notification("Conversion aborted by user.")
                        break
                    if key in data:
                        arr = data[key]
                        arr = np.squeeze(np.asarray(arr))
                        time_ind = np.argmax(arr.shape)
                        other_inds = np.delete(np.arange(arr.ndim), time_ind)
                        if arr.ndim >= 3:
                            arr_to_write = arr.transpose((time_ind,) + tuple(other_inds))
                        else:
                            self.set_notification(f"Key {key} in {f} is not a 3D/4D array.")
                            continue
                        out_fn = f.replace('.mat', f'_{key}.mp4')
                        if not os.path.exists(out_fn) or self.option_checked('rerun'):
                            if use_gpu:
                                writer = skvideo.io.FFmpegWriter(
                                    out_fn,
                                    outputdict={
                                        '-vcodec': 'h264_nvenc',
                                        '-preset': 'fast',
                                        '-pix_fmt': 'yuv420p'
                                    }
                                )
                                for frame in arr_to_write:
                                    if self.abort:
                                        self.set_notification("Conversion aborted by user.")
                                        break
                                    writer.writeFrame(frame.astype(np.uint8))
                                writer.close()
                            else:
                                skvideo.io.vwrite(out_fn, arr_to_write)
                        elif os.path.exists(out_fn) and not self.option_checked('rerun'):
                            self.set_notification(f"File {out_fn} already exists. Re-import the .mat file and use 'rerun' option to overwrite.")
                        ind = self.imported_files.index(f)
                        self.imported_files[ind] = out_fn
                        self.set_notification(f"Converted {f} [{key}] to {out_fn}")
                    else:
                        self.set_notification(f"Key {key} not found in {f}.")
            except Exception as e:
                self.set_notification(f"Error converting {f}: {e}")
            self.update_file_list()
            # self.set_video_progress(int(100 * (num + 1) / len(mat_files)))
            # QApplication.processEvents()
            # time.sleep(0.05)

        self.set_notification("All conversions completed.")

    def track_files(self):
        # Reset abort flag at the start of the operation
        self.abort = False
        # post a notification of all of the selected options
        self.export_options = [label for label, cb in self.export_checkboxes.items() if cb.isChecked()]
        self.set_notification(f"Export options: {', '.join(self.export_options)}")

        # if rerun is selected, move all of the processed files back to imported_files
        if self.option_checked('rerun'):
            self.imported_files += self.processed_files.copy()
            self.processed_files = []
            self.update_file_list()

        save_as_csv = self.option_checked('.csv per file')
        make_labeled_video = self.option_checked('.mp4 preview')
        pcutoff = self.pcutoff
        overwrite = self.option_checked('rerun')
        batch_process = self.option_checked('batch analysis')

        self.worker = DLCProcessWorker(
            self.imported_files,
            self.processed_files,
            self.config_path,
            save_as_csv,
            make_labeled_video,
            pcutoff,
            overwrite,
            batch_process=batch_process
        )
        self.worker.file_processed.connect(self.on_file_processed)
        self.worker.update_lists.connect(self.on_update_lists)
        self.worker.update_notification.connect(self.set_notification)
        self.worker.update_file_list_signal.connect(self.update_file_list)
        self.worker.start()

    def on_file_processed(self, filename, success, message):
        # Optionally handle per-file completion (already handled in update_lists)
        pass

    def on_update_lists(self, imported, processed):
        self.imported_files = imported
        self.processed_files = processed
        self.update_file_list()

    def close_app(self):
        """
        Close the application after updating the file lists.
        """
        self.update_file_list()
        self.close()

    def update_file_list(self):
        """
        Update both imported and processed file QTreeViews and the import progress bar.
        """

        def build_tree_dict(file_list, root_dir=None):
            import os
            if not file_list:
                return [], {}
            rel_paths = []
            for path in file_list:
                if root_dir:
                    rel_path = os.path.relpath(path, root_dir)
                else:
                    rel_path = os.path.abspath(path)
                rel_paths.append(rel_path)
            split_paths = [p.split(os.sep) for p in rel_paths]
            common = []
            for parts in zip(*split_paths):
                if all(part == parts[0] for part in parts):
                    common.append(parts[0])
                else:
                    break
            def insert_path(tree, parts):
                if not parts:
                    return
                head, *tail = parts
                if head not in tree:
                    tree[head] = {}
                insert_path(tree[head], tail)
            tree = {}
            for rel_path in rel_paths:
                parts = rel_path.split(os.sep)
                parts = parts[len(common):] if common else parts
                insert_path(tree, parts)
            def merge_single_child_nodes(node, prefix=None):
                if prefix is None:
                    prefix = []
                items = list(node.items())
                if len(items) == 1 and items[0][1]:
                    child_key, child_val = items[0]
                    # Merge this node with its single child
                    return merge_single_child_nodes(child_val, prefix + [child_key])
                else:
                    merged = {}
                    for k, v in node.items():
                        if v:
                            merged_key, merged_val = merge_single_child_nodes(v, prefix + [k])
                            merged[merged_key] = merged_val
                        else:
                            merged[k] = {}
                    return os.sep.join(prefix), merged
            _, merged_tree = merge_single_child_nodes(tree)
            return common, merged_tree

        def update_model(model, new_common, new_tree, title):
            model.setHorizontalHeaderLabels([title])
            root = model.invisibleRootItem()
            # Helper to find a child by text
            def find_child(parent, text):
                for i in range(parent.rowCount()):
                    if parent.child(i).text() == text:
                        return parent.child(i)
                return None
            # Recursive update
            def update_items(parent, subtree):
                # Remove children not in subtree
                existing_keys = set(parent.child(i).text() for i in range(parent.rowCount()))
                new_keys = set(subtree.keys())
                for i in reversed(range(parent.rowCount())):
                    child = parent.child(i)
                    if child.text() not in new_keys:
                        parent.removeRow(i)
                # Add/update children
                for key, val in subtree.items():
                    child = find_child(parent, key)
                    if child is None:
                        child = QStandardItem(key)
                        child.setEditable(False)
                        parent.appendRow(child)
                    if val:
                        update_items(child, val)
                    else:
                        # Remove any children if this is now a leaf
                        child.removeRows(0, child.rowCount())

            # Handle root node for common prefix
            if new_common:
                root_label = os.sep.join(new_common) + os.sep
                root_item = find_child(root, root_label)
                if root_item is None:
                    root_item = QStandardItem(root_label)
                    root_item.setEditable(False)
                    root.appendRow(root_item)
                # Remove any other root-level items except the root_item
                for i in reversed(range(root.rowCount())):
                    if root.child(i) != root_item:
                        root.removeRow(i)
                update_items(root_item, new_tree)
            else:
                # Remove all root-level items not in new_tree
                existing_keys = set(root.child(i).text() for i in range(root.rowCount()))
                new_keys = set(new_tree.keys())
                for i in reversed(range(root.rowCount())):
                    if root.child(i).text() not in new_keys:
                        root.removeRow(i)
                # Add/update root-level items
                for key, val in new_tree.items():
                    child = find_child(root, key)
                    if child is None:
                        child = QStandardItem(key)
                        child.setEditable(False)
                        root.appendRow(child)
                    if val:
                        update_items(child, val)
                    else:
                        child.removeRows(0, child.rowCount())

        # Imported files
        common, merged_tree = build_tree_dict(self.imported_files)
        update_model(self.file_tree_model, common, merged_tree, "Imported Files")

        # Processed files
        common, merged_tree = build_tree_dict(self.processed_files)
        update_model(self.processed_tree_model, common, merged_tree, "Processed Files")

        # # Imported files
        # build_tree_dict(self.file_tree_model, self.imported_files, title="Imported Files")

        # # Processed files
        # build_tree_dict(self.processed_tree_model, self.processed_files, title="Processed Files")

        total = len(self.imported_files) + len(self.processed_files)
        progress = int(100 * len(self.processed_files) / total) if total > 0 else 0
        self.set_import_progress(progress)
        QApplication.processEvents()
        time.sleep(0.05)
    

    def update_preview_button_state(self):
        """
        Enable the preview button only if a processed video is selected in the right column.
        """
        selected = self.processed_tree_view.selectedIndexes()
        self.preview_button.setEnabled(bool(selected))

    def get_selected_formats(self):
        """
        Returns a list of selected formats from the dropdown (currently only one can be selected).
        """
        # Returns a list of selected formats from the dropdown (currently only one can be selected)
        return [self.format_dropdown.currentText()]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
