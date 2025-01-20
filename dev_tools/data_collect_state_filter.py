import os
import json
from utils.general import STATE_LABEL2ID
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class ImageLabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        self.image_dir = ""
        self.image_files = []
        self.current_index = 0
        self.labels = list(STATE_LABEL2ID.keys())
        self.annotations = {}

        self.initUI()

    def initUI(self):
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.show_prev_image)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.show_next_image)

        self.open_button = QPushButton("Open Directory", self)
        self.open_button.clicked.connect(self.open_directory)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_annotations)

        self.label_buttons = []
        for label in self.labels:
            button = QPushButton(label, self)
            button.clicked.connect(lambda checked, l=label: self.annotate_image(l))
            self.label_buttons.append(button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

        label_layout = QHBoxLayout()
        for button in self.label_buttons:
            label_layout.addWidget(button)
        layout.addLayout(label_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_directory(self):
        self.image_dir = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if self.image_dir:
            self.image_files = [image_name.name for image_name in Path(self.image_dir).rglob("*.jpg")]
            self.image_files = sorted(self.image_files, key=lambda x: int(x.split('.')[0]))
            self.current_index = 0
            self.load_annotations()
            self.show_image()

    def show_image(self):
        if self.image_files:
            image_path = str(Path(self.image_dir) / self.image_files[self.current_index])
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            self.setWindowTitle(f"Image Labeling Tool - {self.image_files[self.current_index]}")

    def show_prev_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def annotate_image(self, label):
        if self.image_files:
            image_file = self.image_files[self.current_index]
            self.annotations[image_file] = STATE_LABEL2ID[label]
            self.save_annotations_to_file()
            if self.current_index < len(self.image_files) - 1:
                self.show_next_image()
            else:
                print("All images have been labeled.")
                self.save_annotations_to_file()

    def save_annotations(self):
        self.save_annotations_to_file()
        self.show_save_message()

    def save_annotations_to_file(self):
        jsonl_path = str(Path(self.image_dir) / "state.jsonl")
        with open(jsonl_path, 'w') as jsonl_file:
            for image_file, label in self.annotations.items():
                jsonl_file.write(json.dumps({"image_name": image_file, "state": label}) + '\n')

    def load_annotations(self):
        jsonl_path = str(Path(self.image_dir) / "state.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as jsonl_file:
                for line in jsonl_file:
                    annotation = json.loads(line.strip())
                    self.annotations[annotation["image_name"]] = annotation["state"]
            # 跳过已经标注的图片
            self.image_files = [f for f in self.image_files if f not in self.annotations]

    @staticmethod
    def show_save_message():
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Annotations have been saved successfully.")
        msg_box.setWindowTitle("Save Complete")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()


if __name__ == "__main__":
    app = QApplication([])
    window = ImageLabelingTool()
    window.show()
    app.exec()
