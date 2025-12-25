# app.py
# Author: Chris Yang
# ============================================================
# GPU Photo Manager (CLIP) - Windows UI (PySide6 / Qt)
#
# What this UI does:
# - Lets user pick an input folder (root_dir)
# - Lets user set thresholds (Screenshot / Review) + Dry Run
# - Runs the job in a background thread (so UI won't freeze)
# - Shows progress bar + live log output
# - Shows final stats + CSV path
# - Provides buttons to open:
#     - CSV log file
#     - Review folder (_AI_REVIEW)
#     - Photos folder (Photos)
#     - Videos folder (Videos)
#
# Prerequisites:
#   1) engine.py must be in the same folder as app.py
#   2) engine.py must expose:
#       - run_job(...)
#       - StopFlag
#
# Install dependencies:
#   pip install pyside6 pyinstaller
#
# Run:
#   python app.py
#
# Package (example):
#   pyinstaller --noconsole --name "PhotoManager" ^
#     --collect-all torch ^
#     --collect-all torchvision ^
#     --collect-all clip ^
#     --collect-all pillow_heif ^
#     app.py
# ============================================================

import os
import sys
import traceback

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar,
    QPlainTextEdit, QLineEdit, QDoubleSpinBox, QCheckBox, QMessageBox
)

import engine


def open_in_explorer(path: str) -> None:
    """Open a folder or file with Windows default handler."""
    try:
        if not path:
            return
        if os.path.isdir(path) or os.path.isfile(path):
            os.startfile(path)  # Windows-only
    except Exception:
        pass

def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class Worker(QThread):
    """
    Background worker thread.
    Signals allow safe UI updates from the worker.
    """
    sig_log = Signal(str)
    sig_progress = Signal(int)     # 0..100
    sig_stage = Signal(str)        # "scanning / others / videos / images"
    sig_done = Signal(dict)        # final stats dict
    sig_failed = Signal(str)       # traceback / error message

    def __init__(self, root_dir: str, ss_th: float, rv_th: float, dry_run: bool):
        super().__init__()
        self.root_dir = root_dir
        self.ss_th = ss_th
        self.rv_th = rv_th
        self.dry_run = dry_run
        self.stop_flag = engine.StopFlag()

    def request_stop(self):
        self.stop_flag.request_stop()

    def run(self):
        try:
            def log_cb(msg: str):
                self.sig_log.emit(msg)

            def progress_cb(done: int, total: int, stage: str):
                pct = int(done * 100 / max(total, 1))
                self.sig_progress.emit(pct)
                self.sig_stage.emit(f"{stage} ({done}/{total})")

            stats = engine.run_job(
                root_dir=self.root_dir,
                screenshot_threshold=self.ss_th,
                review_threshold=self.rv_th,
                dry_run=self.dry_run,
                stop_flag=self.stop_flag,
                log_cb=log_cb,
                progress_cb=progress_cb,
            )
            self.sig_done.emit(stats)
        except Exception:
            self.sig_failed.emit(traceback.format_exc())


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU Photo Manager (CLIP)")
        self.resize(920, 640)

        self.worker = None
        self.last_stats = None

        # ---------- Layout ----------
        root = QVBoxLayout(self)

        # Title
        title = QLabel("GPU-Accelerated Local Photo Manager (CLIP-Based)")
        f = QFont()
        f.setPointSize(14)
        f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        # Info line
        info = QLabel(f"Device hint: {engine.get_device_str()} (cuda_available={engine.is_cuda_available()})")
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(info)

        # Controls grid
        grid = QGridLayout()
        row = 0

        grid.addWidget(QLabel("Input Folder:"), row, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(r"D:\Pictures")
        grid.addWidget(self.input_edit, row, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.on_browse)
        grid.addWidget(self.btn_browse, row, 2)

        row += 1

        grid.addWidget(QLabel("Screenshot Threshold:"), row, 0)
        self.ss_th = QDoubleSpinBox()
        self.ss_th.setRange(0.0, 1.0)
        self.ss_th.setSingleStep(0.01)
        self.ss_th.setValue(0.80)
        grid.addWidget(self.ss_th, row, 1)

        row += 1

        grid.addWidget(QLabel("Review Threshold:"), row, 0)
        self.rv_th = QDoubleSpinBox()
        self.rv_th.setRange(0.0, 1.0)
        self.rv_th.setSingleStep(0.01)
        self.rv_th.setValue(0.60)
        grid.addWidget(self.rv_th, row, 1)

        row += 1

        self.chk_dry_run = QCheckBox("Dry Run (classification only, no file operations)")
        grid.addWidget(self.chk_dry_run, row, 1)

        root.addLayout(grid)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_start.setDefault(True)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        btn_row.addStretch(1)

        self.btn_open_csv = QPushButton("Open CSV Log")
        self.btn_open_csv.clicked.connect(self.on_open_csv)
        self.btn_open_csv.setEnabled(False)
        btn_row.addWidget(self.btn_open_csv)

        self.btn_open_review = QPushButton("Open _AI_REVIEW")
        self.btn_open_review.clicked.connect(self.on_open_review)
        self.btn_open_review.setEnabled(False)
        btn_row.addWidget(self.btn_open_review)

        self.btn_open_photos = QPushButton("Open Photos")
        self.btn_open_photos.clicked.connect(self.on_open_photos)
        self.btn_open_photos.setEnabled(False)
        btn_row.addWidget(self.btn_open_photos)

        self.btn_open_videos = QPushButton("Open Videos")
        self.btn_open_videos.clicked.connect(self.on_open_videos)
        self.btn_open_videos.setEnabled(False)
        btn_row.addWidget(self.btn_open_videos)

        root.addLayout(btn_row)

        # Progress + stage
        prog_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        prog_row.addWidget(self.progress, stretch=2)

        self.lbl_stage = QLabel("Stage: idle")
        self.lbl_stage.setTextInteractionFlags(Qt.TextSelectableByMouse)
        prog_row.addWidget(self.lbl_stage, stretch=3)
        root.addLayout(prog_row)

        # Log area
        root.addWidget(QLabel("Log:"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        root.addWidget(self.log_view, stretch=1)

        # Status / stats
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_status)

    # ---------- UI Helpers ----------
    def append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_browse.setEnabled(not running)
        self.input_edit.setEnabled(not running)
        self.ss_th.setEnabled(not running)
        self.rv_th.setEnabled(not running)
        self.chk_dry_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def enable_open_buttons(self, enabled: bool) -> None:
        self.btn_open_csv.setEnabled(enabled)
        self.btn_open_review.setEnabled(enabled)
        self.btn_open_photos.setEnabled(enabled)
        self.btn_open_videos.setEnabled(enabled)

    def validate_inputs(self) -> bool:
        root_dir = self.input_edit.text().strip()
        if not root_dir or not os.path.isdir(root_dir):
            QMessageBox.warning(self, "Invalid folder", "Please select a valid input folder.")
            return False
        if self.rv_th.value() > self.ss_th.value():
            QMessageBox.warning(self, "Invalid thresholds", "Review threshold must be <= Screenshot threshold.")
            return False
        return True

    # ---------- UI Callbacks ----------
    def on_browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", os.getcwd())
        if d:
            self.input_edit.setText(d)

    def on_start(self):
        if not self.validate_inputs():
            return

        root_dir = self.input_edit.text().strip()
        ss_th = float(self.ss_th.value())
        rv_th = float(self.rv_th.value())
        dry_run = self.chk_dry_run.isChecked()

        # Reset UI
        self.last_stats = None
        self.enable_open_buttons(False)
        self.log_view.clear()
        self.progress.setValue(0)
        self.lbl_stage.setText("Stage: starting...")
        self.lbl_status.setText("Status: Running...")

        self.set_running(True)

        # Start worker thread
        self.worker = Worker(root_dir=root_dir, ss_th=ss_th, rv_th=rv_th, dry_run=dry_run)
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_progress.connect(self.progress.setValue)
        self.worker.sig_stage.connect(lambda s: self.lbl_stage.setText(f"Stage: {s}"))
        self.worker.sig_done.connect(self.on_done)
        self.worker.sig_failed.connect(self.on_failed)
        self.worker.start()

    def on_stop(self):
        if self.worker:
            self.worker.request_stop()
            self.append_log("Stop requested. Finishing current item(s)...")
            self.lbl_status.setText("Status: Stop requested...")

    def on_done(self, stats: dict):
        self.last_stats = stats
        self.set_running(False)
        self.enable_open_buttons(True)

        stopped = stats.get("stopped", False)
        csv_log = stats.get("csv_log", "")
        photos_dir = stats.get("photos_dir", "")
        videos_dir = stats.get("videos_dir", "")
        review_dir = stats.get("review_dir", "")

        self.lbl_status.setText(
            "Status: Done\n"
            f"Stopped: {stopped}\n"
            f"CSV: {csv_log}\n"
            f"Photos moved: {stats.get('photos_moved', 0)} | "
            f"Screenshots: {stats.get('screenshots_to_trash', 0)} | "
            f"Review: {stats.get('images_to_review', 0)} | "
            f"Videos moved: {stats.get('videos_moved', 0)} | "
            f"Other trashed: {stats.get('others_to_trash', 0)} | "
            f"Errors: {stats.get('errors', 0)}\n"
            f"Photos: {photos_dir}\n"
            f"Videos: {videos_dir}\n"
            f"Review: {review_dir}"
        )
        self.append_log("Done.")

    def on_failed(self, err: str):
        self.set_running(False)
        self.enable_open_buttons(False)
        self.lbl_status.setText("Status: Failed")
        self.append_log("FAILED:\n" + err)

        QMessageBox.critical(self, "Job Failed", "The job failed. See the log for details.")

    def on_open_csv(self):
        if self.last_stats:
            open_in_explorer(self.last_stats.get("csv_log", ""))

    def on_open_review(self):
        if self.last_stats:
            open_in_explorer(self.last_stats.get("review_dir", ""))

    def on_open_photos(self):
        if self.last_stats:
            open_in_explorer(self.last_stats.get("photos_dir", ""))

    def on_open_videos(self):
        if self.last_stats:
            open_in_explorer(self.last_stats.get("videos_dir", ""))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("assets/icon.ico")))

    w = App()
    w.setWindowIcon(QIcon(resource_path("assets/icon.ico")))
    w.show()

    sys.exit(app.exec())
