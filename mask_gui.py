import sys
import cv2
import numpy as np
import muller_utils
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QSlider, QColorDialog, QSpinBox, QRubberBand, QComboBox, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QRectF, QEvent

import muller_features
from muller_features import parameters

parameters  = ["CalibratedM11"] + parameters
muller_features.parameters = parameters  # 确保muller_features中的参数列表被正确引用    

def get_mask(image, threshold):
    return np.where(image < threshold, 255, 0).astype(np.uint8)

import students_submission
import types
if hasattr(students_submission, "get_mask") and isinstance(getattr(students_submission, "get_mask"), types.FunctionType):
    get_mask = students_submission.get_mask

class ClickableBlock(QGraphicsRectItem):
    def __init__(self, row, col, block_size, value, update_fn, fg_color):
        super().__init__(col * block_size, row * block_size, block_size, block_size)
        self.row = row
        self.col = col
        self.value = value
        self.update_fn = update_fn
        self.fg_color = fg_color
        self.setBrush(self._get_brush())
        self.setPen(QPen(Qt.NoPen))  # 修正这里
        self.setZValue(1)  # 显示在图像上方
        self.setAcceptHoverEvents(True)

    def _get_brush(self):
        if self.value == 255:
            color = QColor(self.fg_color)
            color.setAlpha(100)  # 半透明
            return QBrush(color)
        else:
            return QBrush(Qt.transparent)

    def mousePressEvent(self, event):
        self.value = 0 if self.value == 255 else 255
        self.setBrush(self._get_brush())
        self.update_fn(self.row, self.col, self.value)


class MaskOverlayEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask 编辑器")
        self.resize(1024, 1024)
        self.colormap = None # 你可以换成其它OpenCV支持的colormap
        self.original_img = None  # shape: (H, W, 16)
        self.gray_img = None
        self.downsampled = None
        self.mask = None
        self.fg_color = QColor("red")
        self.block_size = 20
        self._zoom = 1.0
        self.current_channel = 0  # 新增：当前显示的通道索引

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.load_btn = QPushButton("加载mat文件")
        self.load_btn.clicked.connect(self.load_image)

        self.save_btn = QPushButton("保存 mask")
        self.save_btn.clicked.connect(self.save_mask)
        self.save_btn.setEnabled(False)

        self.load_mask_btn = QPushButton("加载mask")
        self.load_mask_btn.clicked.connect(self.load_mask)
        # 添加到控件布局

        self.color_btn = QPushButton("选择前景颜色")
        self.color_btn.clicked.connect(self.select_color)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(127)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_mask)

        self.spinbox = QSpinBox()
        self.spinbox.setRange(0, 255)
        self.spinbox.setValue(127)
        self.spinbox.setEnabled(False)
        self.spinbox.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.spinbox.setValue)

        self.block_spin = QSpinBox()
        self.block_spin.setRange(1, 512)
        self.block_spin.setValue(self.block_size)
        self.block_spin.setSingleStep(1)
        self.block_spin.setSuffix(" px")
        self.block_spin.valueChanged.connect(self.change_block_size)

        # 新增：通道选择
        self.channel_combo = QComboBox()
        for i, name in enumerate(parameters):
            self.channel_combo.addItem(name, i)
        self.channel_combo.currentIndexChanged.connect(self.change_channel)

        self.lock_mask_checkbox = QCheckBox("锁定mask（禁止修改）")
        self.lock_mask_checkbox.setChecked(False)
        self.show_mask_checkbox = QCheckBox("显示mask")

        self.show_mask_checkbox.setChecked(True)
        self.show_mask_checkbox.stateChanged.connect(self.draw_scene)

        layout = QVBoxLayout()
        controls = QHBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.load_mask_btn)
        controls.addWidget(self.color_btn)
        controls.addWidget(QLabel("块大小:"))
        controls.addWidget(self.block_spin)
        controls.addWidget(QLabel("显示通道:"))
        controls.addWidget(self.channel_combo)
        controls.addWidget(self.lock_mask_checkbox)  # 新增
        controls.addWidget(self.show_mask_checkbox)

        threshold_controls = QHBoxLayout()
        threshold_controls.addWidget(self.slider)
        threshold_controls.addWidget(self.spinbox)

        layout.addLayout(controls)
        layout.addLayout(threshold_controls)
        layout.addWidget(self.view)
        self.setLayout(layout)

        self._dragging = False
        self._rect_start = None
        self._rect_end = None
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self.view.viewport())
        self.view.viewport().installEventFilter(self)
        self.view.viewport().grabGesture(Qt.PinchGesture)

    def load_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择mask文件", "", "Image Files (*.png *.jpg *.bmp)")
        if not path:
            return
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        # 尺寸适配
        h, w = self.downsampled.shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        self.draw_scene()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择mat文件", "", "MAT files (*.mat)")
        if not path:
            return
        mat = muller_utils.load_mat_file(path)
        img_stack = None
        for i, key in enumerate(parameters):
            if key in mat:
                if img_stack is None:
                    img_stack = np.zeros((mat[key].shape[0], mat[key].shape[1], len(parameters)), dtype=np.float32)
                img_stack[..., i] = mat[key]
        print(img_stack.shape)
        self.original_img = img_stack  # (H, W, 16)
        self.set_channel(self.current_channel)
        self.downsampled = self._downsample(self.gray_img)
        self.slider.setEnabled(True)
        self.spinbox.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.update_mask(self.slider.value())
        h, w = self.gray_img.shape
        self.view.setSceneRect(QRectF(0, 0, w, h))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = self._get_current_view_scale()

    def set_channel(self, idx):
        """设置当前显示的通道，并更新gray_img"""
        self.current_channel = idx
        if self.original_img is not None:
            channel_img = self.original_img[..., idx]
            # 归一化到0-255
            arr = channel_img.astype(np.float32)
            arr = arr - np.nanmin(arr)
            arr = arr / (np.nanmax(arr) + 1e-8)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            self.gray_img = arr
            self.downsampled = self._downsample(self.gray_img)  # downsample to 2D
            self.update_mask(self.slider.value())

    def change_channel(self, idx):
        self.set_channel(idx)

    def _get_current_view_scale(self):
        m = self.view.transform()
        return m.m11()

    def _downsample(self, img):
        h, w = img.shape
        bs = self.block_size
        h, w = h - h % bs, w - w % bs
        img = img[:h, :w]
        return img.reshape(h // bs, bs, w // bs, bs).mean(axis=(1, 3)).astype(np.uint8)

    def _upsample_mask(self):
        h, w = self.gray_img.shape
        return cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def update_mask(self, threshold):
        if self.downsampled is None:
            return
        if not self.lock_mask_checkbox.isChecked():
            # 如果锁定mask，则不更新
            self.mask = get_mask(self.downsampled, threshold)
        self.draw_scene()

    def draw_scene(self):
        self.scene.clear()
        h, w = self.gray_img.shape

        # 应用伪彩色
        if self.colormap is None:
            color_img = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2RGB)
        else:
            color_img = cv2.applyColorMap(self.gray_img, self.colormap)
        # OpenCV输出是BGR，QImage要RGB
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(color_img.data, w, h, color_img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.scene.addItem(QGraphicsPixmapItem(pixmap))
        if self.show_mask_checkbox.isChecked() and self.mask is not None:
            full_mask = self._upsample_mask()
            mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            mask_bool = full_mask == 255
            mask_rgba[mask_bool] = [self.fg_color.red(), self.fg_color.green(), self.fg_color.blue(), 100]
            qmask = QImage(mask_rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
            mask_pixmap = QPixmap.fromImage(qmask)
            self.scene.addItem(QGraphicsPixmapItem(mask_pixmap))

        self.view.setSceneRect(QRectF(0, 0, w, h))
        self._apply_zoom()

    def update_block(self, i, j, value):
        self.mask[i, j] = value

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.fg_color = color
            self.draw_scene()

    def save_mask(self):
        if self.mask is None:
            return
        full_mask = self._upsample_mask()
        save_path, _ = QFileDialog.getSaveFileName(self, "保存 Mask 图", "", "PNG Files (*.png)")
        if save_path:
            cv2.imwrite(save_path, full_mask)
            print("已保存：", save_path)

    def change_block_size(self, val):
        self.block_size = val
        if self.gray_img is not None:
            self.downsampled = self._downsample(self.gray_img)
            self.update_mask(self.slider.value())

    def eventFilter(self, source, event):
        # 框选开始
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._dragging = True
            self._rect_start = event.pos()
            self._rect_end = event.pos()
            self._rubber_band.setGeometry(QRectF(self._rect_start, self._rect_end).toRect().normalized())
            self._rubber_band.show()
            return True
        # 框选中
        if event.type() == QEvent.MouseMove and self._dragging:
            self._rect_end = event.pos()
            self._rubber_band.setGeometry(QRectF(self._rect_start, self._rect_end).toRect().normalized())
            return True
        # 框选结束
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._rect_end = event.pos()
            rect = QRectF(self._rect_start, self._rect_end).normalized().toRect()
            self._rubber_band.hide()
            # 判断是否为点按（而不是拖动）
            if rect.width() < 3 and rect.height() < 3:
                self._handle_click(self._rect_start)
            else:
                self._handle_rect(rect)
            self._rect_start = None
            self._rect_end = None
            return True
        # 触控板捏缩放
        if event.type() == QEvent.Gesture:
            pinch = event.gesture(Qt.PinchGesture)
            if pinch:
                scale_factor = pinch.scaleFactor()
                self._zoom *= scale_factor
                self._apply_zoom()
                return True
        # Ctrl+滚轮缩放
        if event.type() == QEvent.Wheel and (event.modifiers() & Qt.ControlModifier):
            if event.angleDelta().y() > 0:
                self._zoom *= 1.2
            else:
                self._zoom /= 1.2
            self._apply_zoom()
            return True
        return super().eventFilter(source, event)

    def _handle_click(self, pos):
        if self.mask is None or self.original_img is None or self.lock_mask_checkbox.isChecked():
            return
        scene_pos = self.view.mapToScene(pos)
        x, y = int(scene_pos.x()), int(scene_pos.y())
        h, w = self.original_img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        block_i = y // self.block_size
        block_j = x // self.block_size
        if 0 <= block_i < self.mask.shape[0] and 0 <= block_j < self.mask.shape[1]:
            self.mask[block_i, block_j] = 0 if self.mask[block_i, block_j] == 255 else 255
            self.draw_scene()

    def _handle_rect(self, rect):
        if self.mask is None or self.original_img is None or self.lock_mask_checkbox.isChecked():
            return
        # scene坐标
        scene_rect = self.view.mapToScene(rect).boundingRect()
        x1, y1, x2, y2 = int(scene_rect.left()), int(scene_rect.top()), int(scene_rect.right()), int(scene_rect.bottom())
        h, w = self.original_img.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        block_i1 = y1 // self.block_size
        block_i2 = y2 // self.block_size
        block_j1 = x1 // self.block_size
        block_j2 = x2 // self.block_size
        low_i = min(block_i1, block_i2)
        low_j = min(block_j1, block_j2)
        high_i = max(block_i1, block_i2)
        high_j = max(block_j1, block_j2)

        if np.sum(self.mask[low_i:high_i + 1, low_j:high_j + 1]) == 0:
            # 如果选中的区域全是0，则反转
            self.mask[low_i:high_i + 1, low_j:high_j + 1] = 255
        else:
            self.mask[low_i:high_i + 1, low_j:high_j + 1] = 0

        # 下面的代码是原来逐个像素反转的逻辑，但效率较低，因此注释掉
        # self.mask[low_i:high_i + 1, low_j
        # for i in range(low_i, high_i + 1):
        #     for j in range(low_j, high_j + 1):
        #         if 0 <= i < self.mask.shape[0] and 0 <= j < self.mask.shape[1]:
        #             self.mask[i, j] = 0 if self.mask[i, j] == 255 else 255
        self.draw_scene()

    def _apply_zoom(self):
        self.view.resetTransform()
        self.view.scale(self._zoom, self._zoom)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MaskOverlayEditor()
    win.show()
    sys.exit(app.exec_())
