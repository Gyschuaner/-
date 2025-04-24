# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui
import cv2
from yolov8 import Ui_Form
from ultralytics import YOLO
import os
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import csv
from PyQt5.QtWidgets import QMainWindow

class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        try:
            super().__init__()
            self.setupUi(self)

            # 初始化变量
            self.cap = None
            self.is_detection_active = False
            self.current_frame = None
            self.current_detected_image = None
            self.last_detection_boxes = None
            self.original_frame_backup = None
            self.detected_frame_backup = None
            self.setWindowTitle("草地虫害在线识别系统")  # 设置窗口标题

            # 创建目录
            self.doubtful_cases_dir = "doubtful_cases"
            self.auto_save_dir = "detection_results"
            self.models_dir = "models"
            self.compare_csv_dir = "compare"
            os.makedirs(self.doubtful_cases_dir, exist_ok=True)
            os.makedirs(self.auto_save_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.compare_csv_dir, exist_ok=True)

            # 初始化模型系统
            self.model = None  # 先初始化为None
            self.current_model = ""

            # 按钮连接
            self.picture_detect_pushButton.clicked.connect(self.load_picture)
            self.stop_detect_pushButton.clicked.connect(self.stop_detection)
            self.save_result_pushButton.clicked.connect(self.save_results)
            self.resultTable.cellDoubleClicked.connect(self.highlight_target)
            self.doubt_accuracy_pushButton.clicked.connect(self.save_doubtful_case)
            self.exit_system_pushButton.clicked.connect(self.exit_system)

            # 模型选择UI初始化 (确保这些控件在UI文件中存在)
            if hasattr(self, 'model_combobox'):
                self.model_combobox.currentTextChanged.connect(self.load_selected_model)
            if hasattr(self, 'refresh_models_button'):
                self.refresh_models_button.clicked.connect(self.refresh_models_list)
            if hasattr(self, 'browse_model_button'):
                self.browse_model_button.clicked.connect(self.browse_model_file)

            # 自动保存复选框 (需要确保有verticalLayout等布局)
            self.auto_save_checkbox = QtWidgets.QCheckBox("自动保存检测结果", self)
            self.auto_save_checkbox.setChecked(True)
            if hasattr(self, 'verticalLayout'):
                self.verticalLayout.addWidget(self.auto_save_checkbox)

            # 初始加载模型
            self.refresh_models_list()

            # 默认加载第一个可用模型
            if (hasattr(self, 'model_combobox') and
                    self.model_combobox.count() > 1):
                self.model_combobox.setCurrentIndex(1)

            #timer
            self.timer = QtCore.QTimer(self)


        except Exception as e:
            print(f"初始化错误: {str(e)}")

    def exit_system(self):
        """安全退出系统"""
        # 创建退出确认对话框
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出系统吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 关闭窗口
            self.close()

            # 完全退出应用程序
            QtWidgets.QApplication.quit()

    def refresh_models_list(self):
        """刷新模型列表"""
        current_text = self.model_combobox.currentText()
        self.model_combobox.clear()
        self.model_combobox.addItem("请选择模型...")

        # 扫描models目录下的.pt文件
        model_files = []
        for f in os.listdir(self.models_dir):
            if f.endswith('.pt'):
                model_files.append(f)

        # 添加到下拉框
        if model_files:
            self.model_combobox.addItems(model_files)
            # 恢复之前选中的模型
            if current_text in model_files:
                self.model_combobox.setCurrentText(current_text)
        else:
            self.model_combobox.setCurrentIndex(0)

    def browse_model_file(self):
        """浏览并添加模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "PyTorch模型文件 (*.pt);;所有文件 (*)"
        )

        if file_path:
            # 复制到models目录
            dst_path = os.path.join(self.models_dir, os.path.basename(file_path))
            try:
                if not os.path.exists(dst_path):
                    import shutil
                    shutil.copy(file_path, dst_path)
                    self.refresh_models_list()
                    self.model_combobox.setCurrentText(os.path.basename(file_path))
                else:
                    QMessageBox.information(self, "提示", "该模型已存在！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"复制模型失败:\n{str(e)}")

    def load_selected_model(self, model_name):
        """加载选中的模型"""
        if model_name == "请选择模型..." or not model_name:
            return

        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"模型文件不存在:\n{model_path}")
            return

        try:
            # 显示加载状态
            QtWidgets.QApplication.processEvents()  # 更新UI

            # 加载模型
            self.model = YOLO(model_path)


            # 如果已有图片，重新检测
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                self.redetect_current_image()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")

    def redetect_current_image(self):
        """用新模型重新检测当前图片"""
        try:
            results = self.model.predict(self.current_frame)
            self.detected_frame = results[0].plot()
            self.display_image(self.detected_frame, self.detected_image)
            self.update_result_table(results[0])

            # 如果开启了自动保存，重新保存结果
            if hasattr(self, 'auto_save_checkbox') and self.auto_save_checkbox.isChecked():
                self.auto_save_results("current_image", results[0])

        except Exception as e:
            print(f"重新检测失败: {e}")

    def save_doubtful_case(self):
        """保存质疑案例（包含用户备注）"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的图片！")
            return

        # 第一步：获取用户备注
        comment, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "添加质疑说明",
            "请详细描述您质疑的原因（例如：检测框不准确、漏检、误检等）：",
            ""
        )
        if not ok:  # 用户点击取消
            return

        try:
            # 创建按日期分类的目录
            date_dir = datetime.now().strftime("%Y%m%d")
            save_dir = os.path.join(self.doubtful_cases_dir, date_dir)
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%H%M%S")
            base_name = f"doubtful_{timestamp}"

            # 保存文件路径
            img_path = os.path.join(save_dir, f"{base_name}.jpg")
            data_path = os.path.join(save_dir, f"{base_name}.txt")
            comment_path = os.path.join(save_dir, f"{base_name}_comment.txt")

            # 保存原始图片
            cv2.imwrite(img_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))

            # 保存检测数据
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 检测信息 ===\n")
                f.write(f"时间: {datetime.now()}\n")
                f.write(f"图像路径: {img_path}\n\n")

                if hasattr(self, 'last_detection_results'):
                    f.write("检测结果:\n")
                    for i, box in enumerate(self.last_detection_results.boxes):
                        cls = self.last_detection_results.names[int(box.cls[0])]
                        conf = box.conf[0].item()
                        coords = [f"{x:.1f}" for x in box.xyxy[0].tolist()]
                        f.write(f"目标{i + 1}: {cls} (置信度: {conf:.4f})\n")
                        f.write(f"坐标: {coords}\n\n")

            # 保存用户备注
            with open(comment_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 用户备注 ===\n")
                f.write(f"时间: {datetime.now()}\n")
                f.write(f"用户描述:\n{comment}\n")

            # 显示保存结果
            self.show_save_result(save_dir, base_name)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def show_save_result(self, save_dir, base_name):
        """显示保存结果对话框"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("保存成功")

        # 创建更详细的显示内容
        text = f"""
        质疑案例已保存到:

        图片文件: {base_name}.jpg
        检测数据: {base_name}.txt
        用户备注: {base_name}_comment.txt

        保存目录: {save_dir}
        """
        msg.setText(text.strip())

        # 添加自定义按钮
        open_btn = msg.addButton("打开目录", QMessageBox.ActionRole)
        view_btn = msg.addButton("查看内容", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Close)

        msg.exec_()

        if msg.clickedButton() == open_btn:
            os.startfile(save_dir)
        elif msg.clickedButton() == view_btn:
            self.show_file_content(save_dir, base_name)

    def show_file_content(self, save_dir, base_name):
        """显示文件内容"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("质疑案例详情")
        layout = QtWidgets.QVBoxLayout()

        # 添加选项卡
        tab_widget = QtWidgets.QTabWidget()

        # 读取并显示检测数据
        data_text = QtWidgets.QTextEdit()
        data_text.setReadOnly(True)
        with open(os.path.join(save_dir, f"{base_name}.txt"), 'r', encoding='utf-8') as f:
            data_text.setText(f.read())
        tab_widget.addTab(data_text, "检测数据")

        # 读取并显示用户备注
        comment_text = QtWidgets.QTextEdit()
        comment_text.setReadOnly(True)
        with open(os.path.join(save_dir, f"{base_name}_comment.txt"), 'r', encoding='utf-8') as f:
            comment_text.setText(f.read())
        tab_widget.addTab(comment_text, "用户备注")

        layout.addWidget(tab_widget)
        dialog.setLayout(layout)
        dialog.resize(600, 400)
        dialog.exec_()

    def highlight_target(self, row, column):
        """在原始图和检测图上同时高亮显示选中的目标"""
        if not hasattr(self, 'last_detection_boxes') or not self.last_detection_boxes:
            return

        try:
            box = self.last_detection_boxes[row]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 高亮原始图
            original_highlight = self.current_frame.copy()
            cv2.rectangle(original_highlight, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(original_highlight, "Selected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # 高亮检测图
            detected_highlight = self.detected_frame.copy()
            cv2.rectangle(detected_highlight, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(detected_highlight, "Selected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # 同时显示两个高亮图像
            self.display_image(original_highlight, self.original_image)
            self.display_image(detected_highlight, self.detected_image)

            # 3秒后恢复
            QtCore.QTimer.singleShot(3000, self.restore_images)

        except Exception as e:
            print(f"高亮目标失败: {e}")

    def restore_images(self):
        """使用备份恢复图像"""
        if self.original_frame_backup is not None:
            self.display_image(self.original_frame_backup, self.original_image)
        if self.detected_frame_backup is not None:
            self.display_image(self.detected_frame_backup, self.detected_image)

    def restore_detection_image(self):
        """恢复原始检测图像"""
        if hasattr(self, 'detected_frame'):
            self.display_image(self.detected_frame, self.detected_image)

    def save_results(self):
        """多功能保存（图片+表格）"""
        if not hasattr(self, 'current_detected_image') or self.current_detected_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的检测结果！")
            return

        # 弹出保存选项对话框
        choice = QMessageBox.question(
            self,
            "保存选项",
            "请选择保存内容：",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.SaveAll | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save
        )

        if choice == QMessageBox.StandardButton.Save:  # 只保存图片
            self._save_combined_image()
        elif choice == QMessageBox.StandardButton.SaveAll:  # 保存图片和表格
            self._save_combined_image()
            self._export_table_to_csv()
        # Cancel则不操作

    def _save_combined_image(self):
        """保存合并图片（原图+检测图）"""
        if self.current_frame is None or self.current_detected_image is None:
            return

        # 水平拼接图片
        h1, w1 = self.current_frame.shape[:2]
        h2, w2 = self.current_detected_image.shape[:2]

        # 调整高度一致
        if h1 != h2:
            scale = h1 / h2
            detected_resized = cv2.resize(
                self.current_detected_image,
                (int(w2 * scale), h1
                ))
        else:
            detected_resized = self.current_detected_image

            # 合并图片（BGR格式）
            combined = np.hstack((
                cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(detected_resized, cv2.COLOR_RGB2BGR)
            ))

            # 添加分隔线
            cv2.line(combined, (w1, 0), (w1, h1), (0, 0, 0), 2) # 黑色分隔线

            # 保存文件
            default_name = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存结果图",
                os.path.join(os.getcwd(), default_name),
                "JPEG Images (*.jpg);;PNG Images (*.png)"
            )

            if file_path:
                cv2.imwrite(file_path, combined)
            QMessageBox.information(self, "成功", f"结果图已保存至：\n{file_path}")

    def _export_table_to_csv(self):
        """导出表格数据到CSV（含坐标信息）"""
        if self.resultTable.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有可导出的数据！")
            return

        default_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出检测结果",
            os.path.join(os.getcwd(), default_name),
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["类别", "置信度(%)", "x", "y", "w", "h"])

                    for row in range(self.resultTable.rowCount()):
                        row_data = [
                            self.resultTable.item(row, 0).text(),  # 类别
                            self.resultTable.item(row, 1).text().replace('%', ''),  # 置信度（去掉%）
                            *[self.resultTable.item(row, col).text()
                              for col in range(2, 6)]  # x,y,w,h
                        ]
                        writer.writerow(row_data)

                QMessageBox.information(self, "成功", f"数据已导出至：\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败：\n{str(e)}")

    def update_result_table(self, result):
        """更新结果表格（含坐标信息）"""
        table = self.resultTable

        # 清空并初始化表格
        table.setRowCount(0)
        table.setColumnCount(6)  # 6列
        table.setHorizontalHeaderLabels(["类别", "置信度", "x", "y", "w", "h"])

        # 表格样式设置
        table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        table.verticalHeader().setVisible(False)

        # 按置信度从高到低排序
        boxes = sorted(result.boxes, key=lambda box: box.conf[0].item(), reverse=True)

        for i, box in enumerate(boxes):
            # 获取检测信息
            class_id = box.cls[0].item()
            class_name = result.names[class_id]
            conf = box.conf[0].item()

            # 从xyxy格式转换为xywh格式
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x = int((x1 + x2) / 2)  # 中心点x
            y = int((y1 + y2) / 2)  # 中心点y
            w = int(x2 - x1)  # 宽度
            h = int(y2 - y1)  # 高度

            # 添加行
            table.insertRow(i)

            # 填充数据
            self._set_table_item(i, 0, class_name)  # 类别
            self._set_table_item(i, 1, f"{conf * 100:.2f}%", conf)  # 置信度（带颜色）
            self._set_table_item(i, 2, str(x))  # x坐标
            self._set_table_item(i, 3, str(y))  # y坐标
            self._set_table_item(i, 4, str(w))  # 宽度
            self._set_table_item(i, 5, str(h))  # 高度

        # 自动调整列宽
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        # 保存检测框引用
        self.last_detection_boxes = result.boxes

    def _set_table_item(self, row, col, text, conf=None):
        """辅助方法：设置表格单元格"""
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(QtCore.Qt.AlignCenter)

        # 置信度列特殊样式
        if col == 1 and conf is not None:
            if conf > 0.8:
                item.setBackground(QtGui.QColor(200, 255, 200))
            elif conf > 0.5:
                item.setBackground(QtGui.QColor(255, 255, 200))
            else:
                item.setBackground(QtGui.QColor(255, 200, 200))

        self.resultTable.setItem(row, col, item)

    def load_picture(self):
        try:
            fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.png)")
            # self.is_detection_active = False

            if fileName:
                if self.timer.isActive():
                    self.timer.stop()
                if self.cap:
                    self.cap.release()
                    self.cap = None

                self.current_frame = cv2.imread(fileName)
                self.display_image(self.current_frame, self.original_image)

                # 检测图片并保存结果
                results = self.model.predict(self.current_frame)
                self.detected_frame = results[0].plot()

                # 保存当前检测结果（转换为RGB格式）
                self.current_detected_image = cv2.cvtColor(self.detected_frame, cv2.COLOR_BGR2RGB)

                self.display_image(self.detected_frame, self.detected_image)
                self.update_result_table(results[0])

                # 自动保存结果
                if self.auto_save_checkbox.isChecked():
                    self.auto_save_results(fileName, results[0])

                # 备份原始图像
                self.original_frame_backup = self.current_frame.copy()
                self.detected_frame_backup = self.detected_frame.copy()

        except Exception as e:
            print(e)

    def auto_save_results(self, original_path, result):
        """自动保存检测结果"""
        try:
            # 创建按日期分类的子目录
            date_str = datetime.now().strftime("%Y%m%d")
            save_dir = os.path.join(self.auto_save_dir, date_str)
            os.makedirs(save_dir, exist_ok=True)

            # 生成基础文件名
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            timestamp = datetime.now().strftime("%H%M%S")

            # 保存原始图片
            orig_save_path = os.path.join(save_dir, f"{base_name}_{timestamp}_orig.jpg")
            cv2.imwrite(orig_save_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))

            # 保存检测结果图片
            detected_save_path = os.path.join(save_dir, f"{base_name}_{timestamp}_detected.jpg")
            cv2.imwrite(detected_save_path, cv2.cvtColor(self.detected_frame, cv2.COLOR_RGB2BGR))

            # 保存检测数据
            data_save_path = os.path.join(save_dir, f"{base_name}_{timestamp}_data.txt")
            with open(data_save_path, 'w', encoding='utf-8') as f:
                f.write(f"检测时间: {datetime.now()}\n")
                f.write(f"原始图片: {orig_save_path}\n")
                f.write(f"结果图片: {detected_save_path}\n\n")
                f.write("检测结果明细:\n")

                for i, box in enumerate(result.boxes):
                    cls = result.names[int(box.cls[0])]
                    conf = box.conf[0].item()
                    coords = box.xyxy[0].tolist()
                    f.write(f"目标{i + 1}: {cls} (置信度: {conf:.4f})\n")
                    f.write(f"坐标: {[f'{x:.1f}' for x in coords]}\n\n")

        except Exception as e:
            print(f"自动保存失败: {e}")

    def display_image(self, frame, target_label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        scaled_pixmap = pixmap.scaled(target_label.size(), QtCore.Qt.KeepAspectRatio)
        target_label.setPixmap(scaled_pixmap)

    def stop_detection(self):
        self.is_detection_active = False

        if self.timer.isActive():
            self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.clear_display(self.original_image)
        self.clear_display(self.detected_image)

    def clear_display(self, target_label):
        target_label.clear()
        target_label.setText('')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
