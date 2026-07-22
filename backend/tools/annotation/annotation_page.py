#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注页面（独立文件）

从 hdf5_visualizer.py 抽离出来的“标注”整页 UI：
- 左：数据集 + 标注语言
- 中：数据可视化（帧预览） + 底部进度/播放/倍速
- 右：Agent 自动标注输出

该模块不直接依赖 HDF5Visualizer 的实现细节，只要求传入的 visualizer 对象提供：
- dataset_list: List[str]
- find_image_group_for_annotation(h5py.File) -> (group_path, group)
- list_cameras_for_annotation(h5py.Group) -> List[str]
- perform_batch_annotation(List[str])
- add_dataset_to_list(path: str)
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QComboBox,
    QTextEdit,
    QPushButton,
    QSlider,
    QMessageBox,
    QFileDialog,
    QProgressDialog,
    QDialog,
    QDialogButtonBox,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter

from PyQt5.QtWidgets import QOpenGLWidget


# -----------------------------
# 可选：导入 label_task_description 的标注能力
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from label_task_description import gen_task_description, get_episode_length, extract_episode_index  # type: ignore
    _LABEL_DESC_IMPORT_ERROR = None
except Exception as e:
    gen_task_description = None
    get_episode_length = None
    extract_episode_index = None
    _LABEL_DESC_IMPORT_ERROR = str(e)


def _fallback_extract_episode_index(hdf5_path: str) -> int:
    """fallback：取文件名中的最后一段数字"""
    base = os.path.basename(hdf5_path)
    nums = []
    cur = ""
    for ch in base:
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                nums.append(cur)
                cur = ""
    if cur:
        nums.append(cur)
    return int(nums[-1]) if nums else 0


def build_frame_reader(hdf: h5py.File, group_path: str, camera_name: str, preload: bool = False):
    """
    构建帧读取器（不依赖 hdf5_visualizer.HDF5DataLoader，避免循环导入）
    返回 (reader(i)->frame, total_frames)
    """
    grp = hdf[group_path]
    node = grp[camera_name]

    # 情况1：直接是 (T,H,W,C) 的数据集
    if isinstance(node, h5py.Dataset):
        total = int(node.shape[0])
        if preload:
            data = node[:]

            def reader(i: int):
                if i >= total:
                    return None
                return data[i]
        else:

            def reader(i: int):
                if i >= total:
                    return None
                return node[i]

        return reader, total

    # 情况2：是子组，内部再有 dataset
    if isinstance(node, h5py.Group):
        candidate_ds = ["data", "images", "frames"] + list(node.keys())
        for ds in candidate_ds:
            if ds in node and isinstance(node[ds], h5py.Dataset):
                ds_node = node[ds]
                if ds_node.ndim >= 3:
                    total = int(ds_node.shape[0])
                    if preload:
                        data = ds_node[:]

                        def reader(i: int):
                            if i >= total:
                                return None
                            return data[i]
                    else:

                        def reader(i: int):
                            if i >= total:
                                return None
                            return ds_node[i]

                    return reader, total

    raise RuntimeError("无法识别该相机的数据存储方式")


class AnnotationAppAdapter:
    """
    独立运行时的适配器：提供 AnnotationPanel 依赖的 API。
    """

    def __init__(self):
        self.dataset_list = []  # List[str]
        self.gen_task_description = gen_task_description
        self.get_episode_length = get_episode_length
        self.extract_episode_index = extract_episode_index

    def add_dataset_to_list(self, path: str):
        if path not in self.dataset_list:
            self.dataset_list.append(path)

    def find_image_group_for_annotation(self, hdf: h5py.File):
        candidate_groups = [
            "observations/images",
            "images",
            "observations",
        ]
        for g in candidate_groups:
            if g in hdf:
                grp = hdf[g]
                if isinstance(grp, h5py.Group):
                    for k in grp.keys():
                        obj = grp[k]
                        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
                            return g, grp
                        if isinstance(obj, h5py.Group):
                            for kk in obj.keys():
                                sub = obj[kk]
                                if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                                    return g, grp
        return None, None

    def list_cameras_for_annotation(self, grp: h5py.Group):
        cameras = []
        for k in grp.keys():
            obj = grp[k]
            if "timestamp" in k.lower():
                continue
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
                cameras.append(k)
            elif isinstance(obj, h5py.Group):
                for kk in obj.keys():
                    sub = obj[kk]
                    if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                        cameras.append(k)
                        break
        return cameras

    def perform_batch_annotation(
        self,
        dataset_paths,
        parent_widget=None,
        camera_name: Optional[str] = None,
        fallback_first_camera: bool = True,
    ):
        """
        批量自动标注：
        - 可指定 camera_name（对所有数据集复用；若某数据集不存在该相机，则回退第一个）
        - 生成/更新 instruction.json（JSON Lines）
        返回：results = List[dict]，每个元素包含 episode_data / dataset_path / output_path / camera_used
        """
        if self.gen_task_description is None:
            msg = "label_task_description 导入失败，自动标注不可用。"
            if _LABEL_DESC_IMPORT_ERROR:
                msg += f"\n\n原因: {_LABEL_DESC_IMPORT_ERROR}"
            QMessageBox.warning(parent_widget, "警告", msg)
            return []

        total = len(dataset_paths)
        progress = QProgressDialog("正在执行自动标注...", "取消", 0, total, parent_widget)
        progress.setWindowTitle("自动标注进度")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        success = 0
        fail = 0
        results = []
        errors = []

        for idx, dataset_path in enumerate(dataset_paths):
            progress.setValue(idx)
            progress.setLabelText(f"正在处理: {os.path.basename(dataset_path)}\n({idx + 1}/{total})")
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            try:
                with h5py.File(dataset_path, "r") as f:
                    group_path, grp = self.find_image_group_for_annotation(f)
                    if grp is None:
                        fail += 1
                        continue
                    cameras = self.list_cameras_for_annotation(grp)
                    if not cameras:
                        fail += 1
                        continue
            except Exception:
                fail += 1
                errors.append({"dataset_path": dataset_path, "stage": "open_or_scan", "error": "打开/扫描HDF5失败"})
                continue

            # 选择相机：优先使用传入 camera_name；否则默认第一个
            selected_camera = None
            if camera_name and camera_name in cameras:
                selected_camera = camera_name
            elif camera_name and fallback_first_camera:
                selected_camera = cameras[0]
            else:
                selected_camera = cameras[0]

            try:
                task_description = self.gen_task_description(dataset_path, camera_name=selected_camera)
                ep_idx = (
                    int(self.extract_episode_index(dataset_path))
                    if self.extract_episode_index is not None
                    else _fallback_extract_episode_index(dataset_path)
                )
                # length 记录任务描述文本的字符数
                length = len(task_description)

                out_dir = os.path.dirname(dataset_path)
                out_path = os.path.join(out_dir, "instruction.json")

                # 读取已存在的文件（如果存在）
                existing_episodes = {}  # episode_index -> instruction_text
                if os.path.exists(out_path):
                    try:
                        # 尝试读取为单个JSON对象格式
                        with open(out_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                try:
                                    data = json.loads(content)
                                    if isinstance(data, dict) and "instructions" in data:
                                        # 新格式：单个JSON对象
                                        instructions_list = data["instructions"]
                                        # 重建 existing_episodes 映射（按索引顺序）
                                        for idx, inst in enumerate(instructions_list):
                                            existing_episodes[idx] = inst
                                    else:
                                        # 旧格式：JSON Lines，尝试转换
                                        f.seek(0)
                                        for line in f:
                                            line = line.strip()
                                            if line:
                                                try:
                                                    ep = json.loads(line)
                                                    if isinstance(ep, dict) and "episode_index" in ep:
                                                        ep_idx = ep["episode_index"]
                                                        task = ep.get("tasks", [""])[0] if ep.get("tasks") else ""
                                                        task_text = task
                                                        if "```json" in task or "{" in task:
                                                            try:
                                                                if "```json" in task:
                                                                    start = task.find("```json") + 7
                                                                    end = task.find("```", start)
                                                                    if end != -1:
                                                                        task = task[start:end].strip()
                                                                parsed = json.loads(task)
                                                                if isinstance(parsed, dict) and "instructions" in parsed:
                                                                    inst = parsed["instructions"]
                                                                    task_text = inst[0] if isinstance(inst, list) and inst else str(inst)
                                                            except:
                                                                pass
                                                        existing_episodes[ep_idx] = task_text
                                                except json.JSONDecodeError:
                                                    continue
                                except json.JSONDecodeError:
                                    # 如果整个文件不是JSON，尝试按JSON Lines解析
                                    f.seek(0)
                                    for line in f:
                                        line = line.strip()
                                        if line:
                                            try:
                                                ep = json.loads(line)
                                                if isinstance(ep, dict) and "episode_index" in ep:
                                                    ep_idx = ep["episode_index"]
                                                    task = ep.get("tasks", [""])[0] if ep.get("tasks") else ""
                                                    task_text = task
                                                    if "```json" in task or "{" in task:
                                                        try:
                                                            if "```json" in task:
                                                                start = task.find("```json") + 7
                                                                end = task.find("```", start)
                                                                if end != -1:
                                                                    task = task[start:end].strip()
                                                            parsed = json.loads(task)
                                                            if isinstance(parsed, dict) and "instructions" in parsed:
                                                                inst = parsed["instructions"]
                                                                task_text = inst[0] if isinstance(inst, list) and inst else str(inst)
                                                        except:
                                                            pass
                                                    existing_episodes[ep_idx] = task_text
                                            except json.JSONDecodeError:
                                                continue
                    except Exception:
                        pass

                # 更新或添加当前episode的描述
                existing_episodes[int(ep_idx)] = task_description

                # 按episode_index排序，构建instructions数组
                sorted_episodes = sorted(existing_episodes.items())
                instructions_list = [desc for _, desc in sorted_episodes]

                # 写入单个JSON对象格式
                output_data = {"instructions": instructions_list}
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                # 为了向后兼容，仍然构建 episode_data
                episode_data = {
                    "episode_index": ep_idx,
                    "tasks": [task_description],
                    "length": length,
                }

                self.add_dataset_to_list(out_path)
                success += 1
                results.append(
                    {
                        "dataset_path": dataset_path,
                        "output_path": out_path,
                        "camera_used": selected_camera,
                        "episode_data": episode_data,
                    }
                )
            except Exception as e:
                fail += 1
                import traceback

                errors.append(
                    {
                        "dataset_path": dataset_path,
                        "stage": "annotate_or_write",
                        "camera": selected_camera,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )

        progress.setValue(total)
        progress.close()

        # 失败时给出更友好的提示（详细错误由调用方展示）
        QMessageBox.information(parent_widget, "自动标注完成", f"成功: {success}\n失败: {fail}")
        for err in errors:
            results.append({"error": err})
        return results


class AutoAnnotateScopeDialog(QDialog):
    """选择自动标注范围：全选 or 仅当前选中"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择标注范围")
        self.setModal(True)
        layout = QVBoxLayout()

        self.rb_all = QRadioButton("全选：对列表中全部数据集进行标注")
        self.rb_selected = QRadioButton("仅标注：当前选中的数据集")
        self.rb_selected.setChecked(True)

        self.group = QButtonGroup(self)
        self.group.addButton(self.rb_all)
        self.group.addButton(self.rb_selected)

        layout.addWidget(self.rb_all)
        layout.addWidget(self.rb_selected)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)

    def get_scope(self) -> str:
        return "all" if self.rb_all.isChecked() else "selected"


class CameraChoiceDialog(QDialog):
    """选择用于自动标注的相机，并可选择是否复用到所有数据集"""

    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择标注相机")
        self.setModal(True)
        layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("相机："))
        self.combo = QComboBox()
        for c in cameras:
            self.combo.addItem(str(c), str(c))
        row.addWidget(self.combo)
        layout.addLayout(row)

        self.apply_all_ck = QCheckBox("对所有数据集复用该相机（若不存在则使用第一个）")
        self.apply_all_ck.setChecked(True)
        layout.addWidget(self.apply_all_ck)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)

    def get_camera(self) -> Optional[str]:
        return self.combo.currentData()

    def apply_all(self) -> bool:
        return self.apply_all_ck.isChecked()


class AutoAnnotateWorker(QThread):
    """
    后台自动标注线程：
    - 避免阻塞 UI（拖拽/缩放/切换控件不会卡死）
    - 通过 signals 回传进度/日志/结果
    """

    progress = pyqtSignal(int, int, str)  # idx(1-based), total, dataset_basename
    log = pyqtSignal(str)
    item_done = pyqtSignal(dict)  # {"dataset_path","output_path","camera_used","episode_data"}
    item_error = pyqtSignal(dict)  # {"dataset_path","stage","camera","error","traceback"}
    finished = pyqtSignal(int, int)  # success, fail

    def __init__(self, visualizer, dataset_paths: list[str], camera_name: str | None, fallback_first_camera: bool = True):
        super().__init__()
        self.visualizer = visualizer
        self.dataset_paths = [str(p) for p in dataset_paths]
        self.camera_name = str(camera_name) if camera_name else None
        self.fallback_first_camera = bool(fallback_first_camera)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @staticmethod
    def _safe_extract_episode_index(visualizer, dataset_path: str) -> int:
        try:
            if hasattr(visualizer, "extract_episode_index") and visualizer.extract_episode_index is not None:
                return int(visualizer.extract_episode_index(dataset_path))
        except Exception:
            pass
        return _fallback_extract_episode_index(dataset_path)

    @staticmethod
    def _read_existing_instructions(out_path: str) -> dict[int, str]:
        """
        兼容读取：
        - 新格式：{"instructions":[...]}
        - 旧格式：JSON Lines (episode_index/tasks)
        返回：episode_index -> instruction_text
        """
        existing: dict[int, str] = {}
        if not os.path.exists(out_path):
            return existing
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return existing
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "instructions" in data and isinstance(data["instructions"], list):
                        for i, inst in enumerate(data["instructions"]):
                            existing[int(i)] = str(inst)
                        return existing
                except json.JSONDecodeError:
                    pass
                # JSON Lines
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ep = json.loads(line)
                        if isinstance(ep, dict) and "episode_index" in ep:
                            ep_idx = int(ep["episode_index"])
                            task = ""
                            tasks = ep.get("tasks") or []
                            if tasks:
                                task = str(tasks[0])
                            existing[ep_idx] = task
                    except Exception:
                        continue
        except Exception:
            return existing
        return existing

    def _write_instructions(self, out_path: str, episode_index: int, task_description: str):
        existing = self._read_existing_instructions(out_path)
        existing[int(episode_index)] = str(task_description)
        instructions_list = [desc for _, desc in sorted(existing.items(), key=lambda kv: kv[0])]
        # 原子写入：避免强制终止线程时写坏文件
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"instructions": instructions_list}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)

    def run(self):
        success = 0
        fail = 0
        total = len(self.dataset_paths)

        for i, dataset_path in enumerate(self.dataset_paths, start=1):
            if self._cancelled:
                break

            base = os.path.basename(dataset_path)
            self.progress.emit(i, total, base)

            try:
                if self._cancelled:
                    break
                # 扫描相机
                with h5py.File(dataset_path, "r") as f:
                    group_path, grp = self.visualizer.find_image_group_for_annotation(f)
                    if grp is None:
                        raise RuntimeError("未找到图像组")
                    cameras = self.visualizer.list_cameras_for_annotation(grp)
                    if not cameras:
                        raise RuntimeError("未找到可用相机")

                # 选择相机
                selected_camera = None
                if self.camera_name and self.camera_name in cameras:
                    selected_camera = self.camera_name
                elif self.camera_name and self.fallback_first_camera:
                    selected_camera = cameras[0]
                else:
                    selected_camera = cameras[0]

                if self._cancelled:
                    break
                # 生成描述（可能含网络调用）
                task_description = self.visualizer.gen_task_description(dataset_path, camera_name=selected_camera)
                episode_index = self._safe_extract_episode_index(self.visualizer, dataset_path)
                length = len(task_description)

                if self._cancelled:
                    break
                out_dir = os.path.dirname(dataset_path)
                out_path = os.path.join(out_dir, "instruction.json")
                self._write_instructions(out_path, episode_index, task_description)

                episode_data = {"episode_index": episode_index, "tasks": [task_description], "length": length}
                self.item_done.emit(
                    {
                        "dataset_path": dataset_path,
                        "output_path": out_path,
                        "camera_used": selected_camera,
                        "episode_data": episode_data,
                    }
                )
                success += 1

            except Exception as e:
                fail += 1
                import traceback

                self.item_error.emit(
                    {
                        "dataset_path": dataset_path,
                        "stage": "annotate_or_write",
                        "camera": self.camera_name,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )

        self.finished.emit(success, fail)


class AnnotationPanel(QWidget):
    """
    标注面板：点击“标注”后显示的整页布局
    左：上半部分数据集， 下半部分标注语言输入
    中：数据可视化 + 底部进度条/播放/倍数按钮
    右：Agent 自动标注，点击后在右下显示生成内容
    """

    def __init__(self, visualizer, parent=None):
        super().__init__(parent)
        self.visualizer = visualizer

        # 播放控制
        self.play_speed = 1.0
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._next_frame)

        # 当前选择/读取器（支持单相机或多相机）
        self._manual_camera_key: Optional[str] = None
        self._manual_frame_reader = None
        self._manual_total_frames: int = 0
        self._multi_camera_mode: bool = True  # 默认开启“6格多相机”
        self._camera_readers = []  # List[callable]
        self._camera_names = []  # List[str]
        self._camera_totals = []  # List[int]
        self._grid_cell_cameras = [""] * 6  # 每格选择的相机（最多6）
        self._current_hdf5_path: Optional[str] = None
        self._h5: Optional[h5py.File] = None  # 保持文件句柄，避免每帧重复打开/整段预读
        self._auto_worker: Optional[AutoAnnotateWorker] = None
        self._auto_progress: Optional[QProgressDialog] = None

        self._build_ui()
        self.refresh_datasets()

    def _build_ui(self):
        root = QHBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        # 全局浅色主题（仅本页）
        self.setStyleSheet(
            """
            QWidget { background: #ffffff; color: #111827; }
            QGroupBox {
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              margin-top: 12px;
              background: #ffffff;
            }
            QGroupBox::title {
              subcontrol-origin: margin;
              left: 10px;
              padding: 0 6px;
              color: #111827;
              font-weight: 600;
            }
            QLabel { color: #111827; }
            QPushButton {
              background: #f9fafb;
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              padding: 8px 12px;
            }
            QPushButton:hover { background: #f3f4f6; }
            QPushButton:pressed { background: #e5e7eb; }
            QPushButton:disabled { color: #9ca3af; background: #f9fafb; }

            QTextEdit {
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              padding: 10px;
              background: #ffffff;
              selection-background-color: #dbeafe;
            }
            QListWidget {
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              background: #ffffff;
            }
            QListWidget::item { padding: 6px 8px; border-radius: 8px; }
            QListWidget::item:selected { background: #eef2ff; color: #111827; }

            QSlider::groove:horizontal {
              border: 1px solid #e5e7eb;
              height: 6px;
              background: #f3f4f6;
              border-radius: 3px;
            }
            QSlider::handle:horizontal {
              background: #2563eb;
              border: 1px solid #1d4ed8;
              width: 16px;
              margin: -6px 0;
              border-radius: 8px;
            }

            QComboBox {
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              padding: 6px 10px;
              background: #ffffff;
              min-height: 28px;
            }
            QComboBox:hover { border-color: #cbd5e1; }
            QComboBox:focus { border-color: #2563eb; }
            QComboBox::drop-down {
              border: none;
              width: 26px;
              subcontrol-origin: padding;
              subcontrol-position: top right;
            }
            QComboBox::down-arrow {
              width: 10px;
              height: 10px;
            }
            QAbstractItemView {
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              selection-background-color: #eef2ff;
              background: #ffffff;
              padding: 6px;
            }

            /* 2x3 网格分割线（灰色网格线） */
            QSplitter::handle {
              background: #e5e7eb;
            }
            QSplitter::handle:horizontal { width: 6px; }
            QSplitter::handle:vertical { height: 6px; }
            """
        )

        # ---------------- 左侧：数据集 + 标注语言 ----------------
        left_box = QVBoxLayout()

        dataset_group = QGroupBox("数据集")
        dataset_layout = QVBoxLayout()

        import_row = QHBoxLayout()
        self.import_btn = QPushButton("导入HDF5")
        self.import_btn.clicked.connect(self._import_hdf5_files)
        import_row.addWidget(self.import_btn)
        self.import_dir_btn = QPushButton("导入目录")
        self.import_dir_btn.clicked.connect(self._import_directory)
        import_row.addWidget(self.import_dir_btn)
        import_row.addStretch()
        dataset_layout.addLayout(import_row)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.SingleSelection)
        # 单击切换
        self.dataset_list.itemClicked.connect(self.on_dataset_clicked)
        dataset_layout.addWidget(self.dataset_list)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("相机："))
        self.manual_camera_combo = QComboBox()
        self.manual_camera_combo.currentIndexChanged.connect(self.on_manual_camera_changed)
        cam_row.addWidget(self.manual_camera_combo)
        dataset_layout.addLayout(cam_row)

        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("Episode："))
        self.manual_episode_label = QLabel("-")
        info_row.addWidget(self.manual_episode_label)
        info_row.addStretch()
        dataset_layout.addLayout(info_row)

        dataset_group.setLayout(dataset_layout)
        left_box.addWidget(dataset_group, stretch=1)

        language_group = QGroupBox("数据标注语言")
        lang_layout = QVBoxLayout()
        self.manual_task_edit = QTextEdit()
        self.manual_task_edit.setPlaceholderText("在这里输入任务描述（例如：pick the block from the pliers）")
        lang_layout.addWidget(self.manual_task_edit)

        save_row = QHBoxLayout()
        self.manual_length_label = QLabel("长度: -")
        save_row.addWidget(self.manual_length_label)
        save_row.addStretch()
        self.manual_save_btn = QPushButton("保存到 instruction.json")
        self.manual_save_btn.clicked.connect(self.on_manual_save)
        save_row.addWidget(self.manual_save_btn)
        lang_layout.addLayout(save_row)

        language_group.setLayout(lang_layout)
        left_box.addWidget(language_group, stretch=1)

        root.addLayout(left_box, stretch=2)

        # ---------------- 中间：数据可视化 ----------------
        center_box = QVBoxLayout()

        self.grid_widget = CameraGridWidget()
        # 绑定每格下拉事件：选择某格相机后立即更新网格播放
        for cell in self.grid_widget.cells:
            cell.camera_changed.connect(self._on_grid_cell_camera_changed)
        center_box.addWidget(self.grid_widget, stretch=6)

        control_row = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self._toggle_play)
        control_row.addWidget(self.play_btn)

        self.speed_btn = QPushButton("1.0x")
        self.speed_btn.setFixedWidth(60)
        self.speed_btn.clicked.connect(self._cycle_speed)
        control_row.addWidget(self.speed_btn)

        control_row.addStretch()
        center_box.addLayout(control_row)

        slider_row = QHBoxLayout()
        self.manual_frame_slider = QSlider(Qt.Horizontal)
        self.manual_frame_slider.setMinimum(0)
        self.manual_frame_slider.setMaximum(0)
        self.manual_frame_slider.valueChanged.connect(self.on_manual_frame_changed)
        slider_row.addWidget(self.manual_frame_slider)

        self.manual_frame_label = QLabel("帧: 0/0")
        slider_row.addWidget(self.manual_frame_label)
        center_box.addLayout(slider_row)

        root.addLayout(center_box, stretch=5)

        # ---------------- 右侧：Agent 自动标注 ----------------
        right_box = QVBoxLayout()

        agent_group = QGroupBox("Agent 自动标注")
        agent_layout = QVBoxLayout()

        agent_tip = QLabel("选择数据集后点击自动标注。生成内容显示在下方。")
        agent_tip.setWordWrap(True)
        agent_tip.setStyleSheet("color: #555;")
        agent_layout.addWidget(agent_tip)

        btn_row = QHBoxLayout()
        self.agent_run_btn = QPushButton("自动标注")
        self.agent_run_btn.clicked.connect(self.on_run_auto_annotation)
        btn_row.addWidget(self.agent_run_btn)
        btn_row.addStretch()
        agent_layout.addLayout(btn_row)

        self.agent_output = QTextEdit()
        self.agent_output.setReadOnly(True)
        self.agent_output.setPlaceholderText("生成内容将在这里显示")
        agent_layout.addWidget(self.agent_output, stretch=1)

        agent_group.setLayout(agent_layout)
        right_box.addWidget(agent_group, stretch=1)
        right_box.addStretch()

        root.addLayout(right_box, stretch=3)
        self.setLayout(root)

    def _on_grid_cell_camera_changed(self, idx: int, camera_name: str):
        if 0 <= int(idx) < 6:
            self._grid_cell_cameras[int(idx)] = str(camera_name or "")
        # 仅在网格模式下立即生效
        if self.manual_camera_combo.currentData() == "__AUTO6__":
            self.on_manual_camera_changed()

    # （已移除“相机导航栏”复选框逻辑；现在每格用下拉框单独选择相机）

    def refresh_datasets(self):
        """从主窗口数据集列表刷新到标注页"""
        hdf5_list = [
            p for p in getattr(self.visualizer, "dataset_list", []) if str(p).lower().endswith((".hdf5", ".h5"))
        ]
        current = None
        if self.dataset_list.selectedItems():
            current = self.dataset_list.selectedItems()[0].data(Qt.UserRole)

        self.dataset_list.blockSignals(True)
        self.dataset_list.clear()
        for p in hdf5_list:
            it = QListWidgetItem(os.path.basename(p))
            it.setData(Qt.UserRole, p)
            self.dataset_list.addItem(it)
        self.dataset_list.blockSignals(False)

        # 恢复选择
        if current:
            for i in range(self.dataset_list.count()):
                if self.dataset_list.item(i).data(Qt.UserRole) == current:
                    self.dataset_list.setCurrentRow(i)
                    break
        elif hdf5_list:
            self.dataset_list.setCurrentRow(0)
        else:
            self.on_manual_dataset_changed()

    def on_dataset_clicked(self, item: QListWidgetItem):
        """单击数据集：立刻切换加载"""
        if not item:
            return
        path = item.data(Qt.UserRole)
        if not path:
            return
        self._switch_dataset(str(path))

    def _switch_dataset(self, hdf5_path: str):
        """切换当前数据集：停止播放、关闭旧文件、打开新文件并刷新相机/预览"""
        # 停止播放
        if self.is_playing:
            self.is_playing = False
            self.play_timer.stop()
            self.play_btn.setText("播放")

        # 关闭旧文件句柄
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
        self._h5 = None
        self._current_hdf5_path = None

        if not os.path.exists(hdf5_path):
            self.grid_widget.set_placeholder("文件不存在")
            return

        try:
            self._h5 = h5py.File(hdf5_path, "r")
            self._current_hdf5_path = hdf5_path
        except Exception as e:
            self.grid_widget.set_placeholder(f"打开HDF5失败: {e}")
            return

        # 刷新相机等信息
        self.on_manual_dataset_changed()

    def _import_hdf5_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择HDF5文件（可多选）",
            "",
            "HDF5文件 (*.hdf5 *.h5);;所有文件 (*)",
        )
        if not paths:
            return
        for p in paths:
            if p.lower().endswith((".hdf5", ".h5")):
                self.visualizer.add_dataset_to_list(p)
        self.refresh_datasets()

    def _import_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择目录", "")
        if not directory:
            return
        try:
            for name in os.listdir(directory):
                if name.lower().endswith((".hdf5", ".h5")):
                    self.visualizer.add_dataset_to_list(os.path.join(directory, name))
        except Exception as e:
            QMessageBox.warning(self, "导入失败", f"扫描目录失败:\n{e}")
        self.refresh_datasets()

    def on_manual_dataset_changed(self, *_):
        # 优先使用当前打开的数据集（单击切换时会设置）
        hdf5_path = self._current_hdf5_path
        if not hdf5_path:
            selected_items = self.dataset_list.selectedItems()
            hdf5_path = selected_items[0].data(Qt.UserRole) if selected_items else None

        # 每次切换数据集时，先根据 instruction.json 中已有内容预填手动标注文本
        # 这样上一条已经保存过的描述不会“消失”，而是自动带入文本框
        self.manual_task_edit.blockSignals(True)
        self.manual_task_edit.clear()
        self.manual_task_edit.blockSignals(False)

        self.manual_camera_combo.blockSignals(True)
        self.manual_camera_combo.clear()
        self.manual_camera_combo.blockSignals(False)

        self._manual_camera_key = None
        self._manual_frame_reader = None
        self._manual_total_frames = 0
        self._camera_readers = []
        self._camera_names = []
        self._camera_totals = []

        if not hdf5_path or not os.path.exists(str(hdf5_path)):
            self.manual_episode_label.setText("-")
            self.grid_widget.set_placeholder("请选择有效的数据集")
            self.manual_frame_slider.setMaximum(0)
            self.manual_frame_slider.setValue(0)
            self._update_manual_frame_label()
            return

        # episode 提取（优先从 visualizer.extract_episode_index 存在时使用）
        episode_index = None
        if hasattr(self.visualizer, "extract_episode_index") and self.visualizer.extract_episode_index is not None:
            try:
                episode_index = int(self.visualizer.extract_episode_index(str(hdf5_path)))
            except Exception:
                episode_index = None
        if episode_index is None:
            episode_index = _fallback_extract_episode_index(hdf5_path)
        self.manual_episode_label.setText(str(episode_index))

        # 预填当前 episode 对应的 instruction 文本（如果 instruction.json 已存在）
        try:
            out_dir = os.path.dirname(str(hdf5_path))
            out_path = os.path.join(out_dir, "instruction.json")
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        try:
                            data = json.loads(content)
                            if isinstance(data, dict) and "instructions" in data:
                                instructions = data.get("instructions") or []
                                if isinstance(instructions, list) and 0 <= int(episode_index) < len(instructions):
                                    self.manual_task_edit.setPlainText(str(instructions[int(episode_index)]))
                        except json.JSONDecodeError:
                            # 旧格式（JSON Lines）简单兼容：找到 episode_index 匹配的行
                            f.seek(0)
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    ep = json.loads(line)
                                    if (
                                        isinstance(ep, dict)
                                        and "episode_index" in ep
                                        and int(ep["episode_index"]) == int(episode_index)
                                    ):
                                        tasks = ep.get("tasks") or []
                                        if tasks:
                                            self.manual_task_edit.setPlainText(str(tasks[0]))
                                        break
                                except json.JSONDecodeError:
                                    continue
        except Exception:
            # 预填失败不影响后续浏览/播放
            pass

        # 读取相机列表
        try:
            f = self._h5 if self._h5 is not None else h5py.File(str(hdf5_path), "r")
            group_path, grp = self.visualizer.find_image_group_for_annotation(f)
            if grp is None:
                self.grid_widget.set_placeholder("该数据集未找到图像组（无相机数据）")
                return
            cams = self.visualizer.list_cameras_for_annotation(grp)
        except Exception as e:
            self.grid_widget.set_placeholder(f"读取相机失败: {e}")
            return

        # 刷新网格每格下拉选项（6个格子=6个相机选择）
        try:
            available = [str(c) for c in cams]
            defaults = available[:6]
            self._grid_cell_cameras = (defaults + [""] * 6)[:6]
            self.grid_widget.set_camera_options_for_all(available, self._grid_cell_cameras)
        except Exception:
            pass

        self.manual_camera_combo.blockSignals(True)
        self.manual_camera_combo.clear()
        # 默认提供“自动(前6个)”作为多相机入口
        self.manual_camera_combo.addItem("全部相机", "__AUTO6__")
        for cam in cams:
            self.manual_camera_combo.addItem(cam, cam)
        self.manual_camera_combo.blockSignals(False)

        if cams:
            # 默认选择“自动(前6个)”
            self.manual_camera_combo.setCurrentIndex(0)
            self.on_manual_camera_changed()
        else:
            self.grid_widget.set_placeholder("未找到可用相机")

    def on_manual_camera_changed(self, *_):
        hdf5_path = self._current_hdf5_path
        if not hdf5_path:
            selected_items = self.dataset_list.selectedItems()
            hdf5_path = selected_items[0].data(Qt.UserRole) if selected_items else None
        cam = self.manual_camera_combo.currentData()
        if not hdf5_path or not cam:
            return

        try:
            f = self._h5 if self._h5 is not None else h5py.File(str(hdf5_path), "r")
            group_path, grp = self.visualizer.find_image_group_for_annotation(f)
            if grp is None:
                self.grid_widget.set_placeholder("未找到图像组")
                return

            # 多相机：自动选择前6个
            if cam == "__AUTO6__":
                self._multi_camera_mode = True
                # 使用“每格下拉”选择；(空) 代表该格子不播放任何相机
                selected_per_cell = []
                for i in range(6):
                    v = ""
                    try:
                        v = self.grid_widget.cells[i].get_selected_camera()
                    except Exception:
                        v = self._grid_cell_cameras[i] if i < len(self._grid_cell_cameras) else ""
                    selected_per_cell.append(str(v or ""))

                # 构建 6 个 reader：空位为 None
                readers = []
                totals = []
                names = []
                for c in selected_per_cell[:6]:
                    if not c:
                        readers.append(None)
                        totals.append(0)
                        names.append("")
                        continue
                    r, t = build_frame_reader(f, group_path, c, preload=False)
                    readers.append(r)
                    totals.append(int(t))
                    names.append(str(c))
                self._camera_readers = readers
                self._camera_totals = totals
                self._camera_names = names
                self._manual_frame_reader = None
                self._manual_camera_key = "__AUTO6__"
                non_zero = [t for t in totals if t and t > 0]
                self._manual_total_frames = min(non_zero) if non_zero else 0
            else:
                # 单相机
                self._multi_camera_mode = False
                reader, total = build_frame_reader(f, group_path, cam, preload=False)
                self._manual_frame_reader = reader
                self._manual_total_frames = int(total)
                self._manual_camera_key = str(cam)
                self._camera_readers = []
                self._camera_names = []
                self._camera_totals = []
        except Exception as e:
            self.grid_widget.set_placeholder(f"构建读取器失败: {e}")
            self._manual_frame_reader = None
            self._manual_total_frames = 0
            self._camera_readers = []
            self._camera_names = []
            self._camera_totals = []
            return

        if self._multi_camera_mode:
            self.manual_length_label.setText(f"长度: {self._manual_total_frames} 帧（多相机取最小）")
        else:
            self.manual_length_label.setText(f"长度: {self._manual_total_frames} 帧")

        self.manual_frame_slider.blockSignals(True)
        self.manual_frame_slider.setMinimum(0)
        self.manual_frame_slider.setMaximum(max(self._manual_total_frames - 1, 0))
        self.manual_frame_slider.setValue(0)
        self.manual_frame_slider.blockSignals(False)

        self._update_manual_frame_label()
        self._render_manual_frame(0)

    def on_manual_frame_changed(self, value):
        self._update_manual_frame_label()
        self._render_manual_frame(int(value))

    def _update_manual_frame_label(self):
        total = max(self._manual_total_frames, 0)
        cur = int(self.manual_frame_slider.value()) if total > 0 else 0
        self.manual_frame_label.setText(f"帧: {cur + 1}/{total if total > 0 else 0}")

    def _render_manual_frame(self, frame_idx: int):
        if self._manual_total_frames <= 0:
            return
        frame_idx = max(0, min(frame_idx, self._manual_total_frames - 1))
        # 多相机 3x2 网格
        if self._multi_camera_mode and self._camera_readers:
            for i in range(6):
                if i < len(self._camera_readers):
                    reader = self._camera_readers[i]
                    # (空) 相机：不播放，保持空白
                    if reader is None:
                        self.grid_widget.set_camera_frame(i, None, "")
                        continue
                    im = reader(frame_idx)
                    if im is None:
                        self.grid_widget.set_camera_frame(i, None, "")
                    else:
                        self.grid_widget.set_camera_frame(i, self._to_rgb_qimage(im), "")
                else:
                    self.grid_widget.set_camera_frame(i, None, "")
            self.grid_widget.clear_placeholder()
            return

        # 单相机
        if self._manual_frame_reader is None:
            return
        img = self._manual_frame_reader(frame_idx)
        if img is None:
            self.grid_widget.set_placeholder("该帧无数据")
            return
        self.grid_widget.clear_placeholder()
        self.grid_widget.set_camera_frame(0, self._to_rgb_qimage(img), str(self._manual_camera_key or ""))
        # 其余格子清空
        for i in range(1, 6):
            self.grid_widget.set_camera_frame(i, None, "")

    def _to_bgr_uint8(self, img):
        """统一把图像转成 BGR uint8"""
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        if img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _to_rgb_qimage(self, img):
        """把 ndarray 转成 QImage(RGB888)，供 OpenGLWidget 绘制"""
        bgr = self._to_bgr_uint8(img)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    # ---------------- 自动标注（右侧） ----------------
    def on_run_auto_annotation(self):
        # 自动标注能力依赖 label_task_description
        if not hasattr(self.visualizer, "gen_task_description") or self.visualizer.gen_task_description is None:
            msg = "无法使用 label_task_description 的自动标注函数。"
            if _LABEL_DESC_IMPORT_ERROR:
                msg += f"\n\n原因: {_LABEL_DESC_IMPORT_ERROR}"
            msg += "\n\n请设置环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY 后重启程序。"
            QMessageBox.warning(self, "警告", msg)
            return

        # 1) 选择范围：全选 or 当前选中
        scope_dlg = AutoAnnotateScopeDialog(self)
        if scope_dlg.exec_() != QDialog.Accepted:
            return
        scope = scope_dlg.get_scope()

        # 计算待标注数据集列表
        if scope == "all":
            dataset_paths = [p for p in getattr(self.visualizer, "dataset_list", []) if str(p).lower().endswith((".hdf5", ".h5"))]
        else:
            dataset_paths = [
                it.data(Qt.UserRole)
                for it in self.dataset_list.selectedItems()
                if it.data(Qt.UserRole) and str(it.data(Qt.UserRole)).lower().endswith((".hdf5", ".h5"))
            ]

        if not dataset_paths:
            QMessageBox.warning(self, "提示", "未选择任何 HDF5 数据集")
            return

        # 2) 选择相机（基于第一个数据集的相机列表）
        try:
            with h5py.File(str(dataset_paths[0]), "r") as f:
                group_path, grp = self.visualizer.find_image_group_for_annotation(f)
                if grp is None:
                    QMessageBox.warning(self, "提示", "第一个数据集未找到图像组，无法选择相机")
                    return
                cams = self.visualizer.list_cameras_for_annotation(grp)
        except Exception as e:
            QMessageBox.warning(self, "提示", f"读取相机失败：{e}")
            return

        if not cams:
            QMessageBox.warning(self, "提示", "未找到可用相机")
            return

        cam_dlg = CameraChoiceDialog(cams, self)
        if cam_dlg.exec_() != QDialog.Accepted:
            return
        chosen_cam = cam_dlg.get_camera()
        apply_all = cam_dlg.apply_all()

        self.agent_output.setPlainText(
            f"开始自动标注：{len(dataset_paths)} 个数据集\n"
            f"范围：{'全选' if scope == 'all' else '仅当前选中'}\n"
            f"相机：{chosen_cam}（{'复用' if apply_all else '仅用于匹配存在的'}）\n"
        )

        # 3) 后台执行标注（QThread），保持 UI 可拖拽/可交互
        if self._auto_worker is not None and self._auto_worker.isRunning():
            QMessageBox.information(self, "提示", "自动标注正在运行，请先等待完成或取消。")
            return

        self.agent_output.append("\n开始后台自动标注...\n")
        self.agent_run_btn.setEnabled(False)

        self._auto_progress = QProgressDialog("正在执行自动标注...", "取消", 0, len(dataset_paths), self)
        self._auto_progress.setWindowTitle("自动标注进度")
        self._auto_progress.setWindowModality(Qt.WindowModal)
        self._auto_progress.setMinimumDuration(0)
        self._auto_progress.setValue(0)
        self._auto_progress.setAutoClose(False)
        self._auto_progress.setAutoReset(False)
        self._auto_progress.show()

        # Cancel / 右上角 X：立即发出取消请求
        try:
            self._auto_progress.canceled.connect(self._on_auto_cancel_requested)  # type: ignore
            self._auto_progress.rejected.connect(self._on_auto_cancel_requested)  # type: ignore
        except Exception:
            pass

        self._auto_worker = AutoAnnotateWorker(
            visualizer=self.visualizer,
            dataset_paths=dataset_paths,
            camera_name=str(chosen_cam) if apply_all else None,
            fallback_first_camera=True,
        )

        def _on_progress(idx: int, total: int, name: str):
            if self._auto_progress is not None:
                self._auto_progress.setMaximum(total)
                self._auto_progress.setValue(max(0, idx - 1))
                self._auto_progress.setLabelText(f"正在处理: {name}\n({idx}/{total})")
                if self._auto_progress.wasCanceled():
                    self._on_auto_cancel_requested()

        def _on_item_done(r: dict):
            ds = r.get("dataset_path")
            cam_used = r.get("camera_used")
            outp = r.get("output_path")
            ep = r.get("episode_data")
            self.agent_output.append(f"\n[成功] {os.path.basename(str(ds))} | camera={cam_used}")
            self.agent_output.append(json.dumps(ep, ensure_ascii=False, indent=2))
            self.agent_output.append(f"写入：{outp}")

        def _on_item_error(err: dict):
            ds = err.get("dataset_path")
            self.agent_output.append(f"\n[失败] {os.path.basename(str(ds)) if ds else ds}")
            self.agent_output.append(f"阶段: {err.get('stage')}")
            if err.get("camera") is not None:
                self.agent_output.append(f"相机: {err.get('camera')}")
            self.agent_output.append(f"错误: {err.get('error')}")
            tb = err.get("traceback")
            if tb:
                self.agent_output.append("Traceback:\n" + tb)

        def _on_finished(success: int, fail: int):
            if self._auto_progress is not None:
                self._auto_progress.setValue(self._auto_progress.maximum())
                self._auto_progress.close()
                self._auto_progress = None
            self.agent_output.append(f"\n完成。成功: {success} 失败: {fail}\n")
            self.agent_run_btn.setEnabled(True)
            self._auto_worker = None
            self.refresh_datasets()

        self._auto_worker.progress.connect(_on_progress)
        self._auto_worker.item_done.connect(_on_item_done)
        self._auto_worker.item_error.connect(_on_item_error)
        self._auto_worker.finished.connect(_on_finished)
        self._auto_worker.start()

    def _on_auto_cancel_requested(self):
        """
        用户点击“取消”或关闭(X)时：
        - 先请求协作取消（尽快在下一轮停止）
        - 如果卡在网络请求里，短延迟后强制终止线程，保证 UI 真的停得下来
        """
        if self._auto_progress is not None:
            try:
                self._auto_progress.setLabelText("正在取消...（如卡在网络请求，将强制终止）")
            except Exception:
                pass
        if self._auto_worker is None:
            return
        try:
            self._auto_worker.cancel()
        except Exception:
            pass

        # 500ms 后若仍在运行，则强制终止（避免 requests 长 timeout 导致“取消无反应”）
        def _force_if_still_running():
            if self._auto_worker is None:
                return
            if self._auto_worker.isRunning():
                try:
                    self.agent_output.append("\n[取消] 后台任务卡住，已强制终止线程。\n")
                    self._auto_worker.terminate()
                    self._auto_worker.wait(1000)
                except Exception:
                    pass
            # 收尾 UI
            if self._auto_progress is not None:
                try:
                    self._auto_progress.close()
                except Exception:
                    pass
                self._auto_progress = None
            self.agent_run_btn.setEnabled(True)
            self._auto_worker = None
            self.refresh_datasets()

        QTimer.singleShot(500, _force_if_still_running)

    def on_manual_save(self):
        selected_items = self.dataset_list.selectedItems()
        hdf5_path = selected_items[0].data(Qt.UserRole) if selected_items else None
        cam = self.manual_camera_combo.currentData()
        task = self.manual_task_edit.toPlainText().strip()

        if not hdf5_path or not os.path.exists(hdf5_path):
            QMessageBox.warning(self, "提示", "请选择有效的数据集")
            return
        if not cam:
            QMessageBox.warning(self, "提示", "请选择相机")
            return
        if not task:
            QMessageBox.warning(self, "提示", "请输入任务描述")
            return

        episode_index = _fallback_extract_episode_index(hdf5_path)

        # length 记录任务描述文本的字符数
        length = len(task)

        out_dir = os.path.dirname(hdf5_path)
        out_path = os.path.join(out_dir, "instruction.json")

        # 读取已存在的文件（如果存在）
        existing_episodes = {}  # episode_index -> instruction_text
        if os.path.exists(out_path):
            try:
                # 尝试读取为单个JSON对象格式
                with open(out_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        try:
                            data = json.loads(content)
                            if isinstance(data, dict) and "instructions" in data:
                                # 新格式：单个JSON对象
                                instructions_list = data["instructions"]
                                # 重建 existing_episodes 映射（按索引顺序）
                                for idx, inst in enumerate(instructions_list):
                                    existing_episodes[idx] = inst
                            else:
                                # 旧格式：JSON Lines，尝试转换
                                f.seek(0)
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            ep = json.loads(line)
                                            if isinstance(ep, dict) and "episode_index" in ep:
                                                ep_idx = ep["episode_index"]
                                                task = ep.get("tasks", [""])[0] if ep.get("tasks") else ""
                                                task_text = task
                                                if "```json" in task or "{" in task:
                                                    try:
                                                        if "```json" in task:
                                                            start = task.find("```json") + 7
                                                            end = task.find("```", start)
                                                            if end != -1:
                                                                task = task[start:end].strip()
                                                        parsed = json.loads(task)
                                                        if isinstance(parsed, dict) and "instructions" in parsed:
                                                            inst = parsed["instructions"]
                                                            task_text = inst[0] if isinstance(inst, list) and inst else str(inst)
                                                    except:
                                                        pass
                                                existing_episodes[ep_idx] = task_text
                                        except json.JSONDecodeError:
                                            continue
                        except json.JSONDecodeError:
                            # 如果整个文件不是JSON，尝试按JSON Lines解析
                            f.seek(0)
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        ep = json.loads(line)
                                        if isinstance(ep, dict) and "episode_index" in ep:
                                            ep_idx = ep["episode_index"]
                                            task = ep.get("tasks", [""])[0] if ep.get("tasks") else ""
                                            task_text = task
                                            if "```json" in task or "{" in task:
                                                try:
                                                    if "```json" in task:
                                                        start = task.find("```json") + 7
                                                        end = task.find("```", start)
                                                        if end != -1:
                                                            task = task[start:end].strip()
                                                    parsed = json.loads(task)
                                                    if isinstance(parsed, dict) and "instructions" in parsed:
                                                        inst = parsed["instructions"]
                                                        task_text = inst[0] if isinstance(inst, list) and inst else str(inst)
                                                except:
                                                    pass
                                            existing_episodes[ep_idx] = task_text
                                    except json.JSONDecodeError:
                                        continue
            except Exception:
                pass

        # 更新或添加当前episode的描述
        existing_episodes[int(episode_index)] = task

        # 按episode_index排序，构建instructions数组
        sorted_episodes = sorted(existing_episodes.items())
        instructions_list = [desc for _, desc in sorted_episodes]

        # 写入单个JSON对象格式
        output_data = {"instructions": instructions_list}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        if hasattr(self.visualizer, "dataset_list") and out_path not in self.visualizer.dataset_list:
            self.visualizer.add_dataset_to_list(out_path)

        QMessageBox.information(
            self,
            "保存成功",
            f"已保存：\n{out_path}\n\nEpisode {episode_index}\n相机: {cam}\n长度: {length}",
        )
        self.refresh_datasets()

    # ---------------- 播放控制 ----------------
    def _toggle_play(self):
        if self._manual_total_frames <= 0:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("暂停")
            interval = int(100 / self.play_speed)
            self.play_timer.start(max(10, interval))
        else:
            self.play_btn.setText("播放")
            self.play_timer.stop()

    def _cycle_speed(self):
        speeds = [0.5, 1.0, 2.0, 4.0]
        cur_idx = speeds.index(self.play_speed) if self.play_speed in speeds else 1
        nxt = speeds[(cur_idx + 1) % len(speeds)]
        self.play_speed = float(nxt)
        self.speed_btn.setText(f"{nxt:.1f}x")
        if self.is_playing:
            interval = int(100 / self.play_speed)
            self.play_timer.start(max(10, interval))

    def _next_frame(self):
        if self._manual_total_frames <= 0:
            self._toggle_play()
            return

        cur = int(self.manual_frame_slider.value())
        if cur < int(self.manual_frame_slider.maximum()):
            self.manual_frame_slider.setValue(cur + 1)
        else:
            self._toggle_play()

    def closeEvent(self, event):
        """关闭窗口时释放 HDF5 文件句柄"""
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
        self._h5 = None
        super().closeEvent(event)


class CameraGLWidget(QOpenGLWidget):
    """
    用 QOpenGLWidget 承载图像显示；渲染采用 QPainter 画 QImage，
    不引入 PyOpenGL 依赖，同时可利用 Qt 的 GPU 合成路径，30fps+ 更稳。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self.setMinimumSize(120, 90)

    def set_image(self, img: Optional[QImage]):
        self._image = img
        self.update()

    def paintGL(self):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        if self._image is not None and not self._image.isNull():
            # 保持宽高比铺满
            target = self.rect()
            src_size = self._image.size()
            scaled = src_size.scaled(target.size(), Qt.KeepAspectRatio)
            x = (target.width() - scaled.width()) // 2
            y = (target.height() - scaled.height()) // 2
            painter.drawImage(x, y, self._image.scaled(scaled, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        painter.end()


class CameraCell(QWidget):
    """单格：标题 + OpenGL 显示"""
    camera_changed = pyqtSignal(int, str)  # idx, camera_name

    def __init__(self, idx: int, title: str = "", parent=None):
        super().__init__(parent)
        self._idx = int(idx)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 顶部：相机选择下拉（网格模式使用）
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        self.combo = QComboBox()
        self.combo.setMinimumWidth(220)
        self.combo.currentIndexChanged.connect(self._emit_camera_changed)
        top_row.addWidget(self.combo, stretch=1)
        layout.addLayout(top_row)

        # 标题标签会导致“相机名显示两次”（下拉框里一次、标题一次），这里直接隐藏
        self.title_label = QLabel("")
        self.title_label.hide()
        self.gl = CameraGLWidget()
        layout.addWidget(self.gl, stretch=1)
        self.setLayout(layout)
        # 卡片化浅色样式
        self.setStyleSheet(
            "background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;"
        )

    def set_camera_options(self, cameras: list[str], selected: str | None = None):
        """设置下拉可选相机；selected 为相机名或 None"""
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("(空)", "")
        for c in cameras:
            self.combo.addItem(str(c), str(c))
        if selected:
            # 若不在列表，仍然显示为第一项
            idx = self.combo.findData(str(selected))
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
            else:
                self.combo.setCurrentIndex(0)
        else:
            self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

    def get_selected_camera(self) -> str:
        v = self.combo.currentData()
        return str(v) if v is not None else ""

    def set_title(self, title: str):
        # 已隐藏 title_label，保留接口但不再显示，避免重复相机名
        self.title_label.setText("")

    def set_image(self, img: Optional[QImage]):
        self.gl.set_image(img)

    def _emit_camera_changed(self, *_):
        self.camera_changed.emit(self._idx, self.get_selected_camera())


class CameraGridWidget(QWidget):
    """
    2x3 规则网格：QSplitter 嵌套
    - 外层：Vertical splitter（2行）
    - 每行：Horizontal splitter（3列）
    支持拖拽边框放大缩小单图像，其它自适应。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)

        self.placeholder = QLabel("请选择数据集与相机")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet(
            "border: 2px dashed #cbd5e1; color: #6b7280; background-color: #ffffff; border-radius: 12px;"
        )

        self.vsplit = QSplitter(Qt.Vertical)
        self.hsplit_top = QSplitter(Qt.Horizontal)
        self.hsplit_bot = QSplitter(Qt.Horizontal)

        self.cells = [CameraCell(i, "") for i in range(6)]
        for i in range(3):
            self.hsplit_top.addWidget(self.cells[i])
        for i in range(3, 6):
            self.hsplit_bot.addWidget(self.cells[i])

        self.vsplit.addWidget(self.hsplit_top)
        self.vsplit.addWidget(self.hsplit_bot)

        # 默认均分
        self.hsplit_top.setSizes([1, 1, 1])
        self.hsplit_bot.setSizes([1, 1, 1])
        self.vsplit.setSizes([1, 1])

        root.addWidget(self.vsplit)
        root.addWidget(self.placeholder)
        self.setLayout(root)
        self.clear_placeholder()

    def set_placeholder(self, text: str):
        self.placeholder.setText(text)
        self.placeholder.show()
        self.vsplit.hide()

    def clear_placeholder(self):
        self.placeholder.hide()
        self.vsplit.show()

    def set_camera_frame(self, idx: int, image: Optional[QImage], title: str):
        if 0 <= idx < len(self.cells):
            self.cells[idx].set_title(title)
            self.cells[idx].set_image(image)

    def set_camera_options_for_all(self, cameras: list[str], selected: list[str] | None = None):
        """为 6 个格子设置下拉选项；selected 为每格相机名（长度<=6）"""
        selected = selected or []
        for i, cell in enumerate(self.cells):
            sel = selected[i] if i < len(selected) else None
            cell.set_camera_options(cameras, sel)


class AnnotationMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("标注工具")
        self.setGeometry(100, 100, 1400, 850)
        self.adapter = AnnotationAppAdapter()
        self.panel = AnnotationPanel(self.adapter, self)
        self.setCentralWidget(self.panel)


def main():
    app = QApplication([])
    app.setStyle("Fusion")

    # 设置应用字体（Windows 下更友好）
    font = QFont()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(9)
    app.setFont(font)

    win = AnnotationMainWindow()
    win.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
