#!/usr/bin/env python3
"""
驾驶员疲劳检测算法 (六种独立场景：强光/日间/夜间 × 眼镜/无眼镜)
- 亮度预检：< 85 夜间，85~200 日间，≥ 200 强光
- 眼镜场景在校准阶段判断（30帧中≥15帧判定为眼镜）
- 六种场景独立校准参数和眼部/嘴部/头部判定阈值
- 头部前200帧跳过疲劳检测，无3D框、无半透明填充

模块组织：工作模式选择 → 视频采集模块 → 预处理模块 → 人脸检测模块 → 特征提取模块 → 疲劳判定模块 → 报警与可视化模块 → 批量分析模块
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Optional, List
from collections import deque
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("警告：winsound 不可用（非 Windows 系统），将无法播放蜂鸣声")


# ========================= 工作模式选择模块 =========================
class ModeSelector:
    """用户界面选择操作模式：实时检测、视频检测、批量分析、退出"""
    def __init__(self):
        self.mode: Optional[str] = None          # 存储用户选择的模式
        self.root = tk.Tk()                      # 创建Tkinter根窗口
        self.root.title("驾驶员疲劳检测算法")
        self.root.geometry("400x450")
        self.root.resizable(False, False)
        self._center_window()                    # 窗口居中
        self._setup_ui()                         # 构建UI组件

    def _center_window(self):
        """将窗口置于屏幕中央"""
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f'{w}x{h}+{x}+{y}')

    def _setup_ui(self):
        """创建按钮布局"""
        tk.Label(self.root, text="请选择操作模式", font=("Microsoft YaHei", 16, "bold"), pady=20).pack()
        btn_frame = tk.Frame(self.root, pady=20)
        btn_frame.pack()
        # 实时检测按钮
        tk.Button(btn_frame, text="实时检测", command=self._select_realtime,
                  font=("Microsoft YaHei", 12), width=15, height=2,
                  bg="#4CAF50", fg="white").grid(row=0, column=0, padx=10, pady=5)
        # 视频检测按钮
        tk.Button(btn_frame, text="视频检测", command=self._select_videofile,
                  font=("Microsoft YaHei", 12), width=15, height=2,
                  bg="#FF9800", fg="white").grid(row=1, column=0, padx=10, pady=5)
        # 批量分析按钮
        tk.Button(btn_frame, text="批量分析", command=self._select_batch,
                  font=("Microsoft YaHei", 12), width=15, height=2,
                  bg="#2196F3", fg="white").grid(row=2, column=0, padx=10, pady=5)
        # 退出按钮
        tk.Button(btn_frame, text="退出", command=self._exit_program,
                  font=("Microsoft YaHei", 10), width=10,
                  bg="#f44336", fg="white").grid(row=3, column=0, padx=10, pady=20)

    def _select_realtime(self):
        """选择实时摄像头模式"""
        self.mode = "realtime"
        self.root.quit()
        self.root.destroy()

    def _select_videofile(self):
        """选择单视频文件模式"""
        self.mode = "videofile"
        self.root.quit()
        self.root.destroy()

    def _select_batch(self):
        """选择批量分析模式"""
        self.mode = "batch"
        self.root.quit()
        self.root.destroy()

    def _exit_program(self):
        """退出程序"""
        self.mode = "exit"
        self.root.quit()
        self.root.destroy()

    def run(self) -> str:
        """启动界面，返回用户选择的模式字符串"""
        self.root.mainloop()
        return self.mode if self.mode else "exit"


# ========================= 视频采集模块 =========================
# 视频采集功能集成在 run_realtime、run_video_file 等函数中，此处保留占位说明
# 具体实现见后文主控制部分


# ========================= 预处理模块 =========================
def auto_enhance(frame):
    """
    自动图像增强：根据亮度进行不同的处理
    - 低照度（亮度<70）：伽马校正（gamma=0.6）增强暗部细节
    - 高照度（亮度>200）：CLAHE（限制对比度自适应直方图均衡化）抑制反光
    - 正常亮度：仅去噪（高斯模糊）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    denoised = cv2.GaussianBlur(frame, (3, 3), 0)      # 基础去噪
    if brightness < 70:
        # 低照度：伽马校正，映射表加速
        gamma = 0.6
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(denoised, table)
    elif brightness > 200:
        # 高照度：转换到LAB颜色空间，对L通道做CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced = denoised
    return enhanced


def remove_glare(image, level=1):
    """
    去除图像反光（如眼镜片反光）
    level: 处理强度，0=不处理，1=单次修复，2=双重修复+中值模糊
    原理：顶帽变换提取高光区域，形态学闭/开操作优化掩码，然后使用inpaint修复
    """
    if level == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)   # 提取高光
    thresh = 20 if level >= 2 else 25
    _, mask = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=level)
    mask = mask.astype(np.uint8)
    if np.count_nonzero(mask) > 0:
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        if level >= 2:
            result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
        result = cv2.medianBlur(result, 3)      # 轻微平滑
        return result
    return image


def classify_scene(frame, face_rect=None):
    """
    基于边缘+轮廓分析 + 高光区域（眼镜片反光）判断是否戴眼镜
    亮度自适应：夜间禁用高光检测，强光提高Canny阈值
    返回 'glasses' 或 'normal'
    """
    if face_rect is None:
        return "normal"
    
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overall_brightness = np.mean(gray_full)
    is_night = (overall_brightness < 85)       # 夜间标志
    is_strong = (overall_brightness >= 200)    # 强光标志
    
    x1, y1, x2, y2 = face_rect
    h_face = y2 - y1
    # 提取眼睛区域（人脸上半部分）
    eye_roi_y1 = y1
    eye_roi_y2 = y1 + int(h_face * 0.6)
    eye_roi = frame[eye_roi_y1:eye_roi_y2, x1:x2]
    if eye_roi.size == 0:
        return "normal"
    
    # 高光检测：夜间跳过，强光提高阈值
    if not is_night:
        hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
        mask_highlight = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))   # 高亮区域
        highlight_area = np.sum(mask_highlight > 0)
        strong_thresh = 200 if is_strong else 120
        if highlight_area > strong_thresh:
            return "glasses"
    
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)                    # 增强对比度
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应Canny边缘检测阈值
    if is_night:
        edges = cv2.Canny(blurred, 80, 200)
    elif is_strong:
        edges = cv2.Canny(blurred, 100, 250)
    else:
        edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)   # 闭合边缘
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    roi_w = x2 - x1
    glasses_candidates = []
    min_area = 150 if (is_night or is_strong) else 100
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > 6000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        # 眼镜框通常宽高比较大
        if aspect_ratio < 1.2 or aspect_ratio > 8.0:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
        if convexity < 0.4:        # 凸性过低不是眼镜框
            continue
        glasses_candidates.append((x, y, w, h, area))
    
    min_candidates = 2 if not is_night else 3
    if len(glasses_candidates) >= min_candidates:
        roi_center_x = roi_w / 2
        offset = 25 if is_night else 15
        # 分别统计左右两侧的候选轮廓
        left_cnts = [c for c in glasses_candidates if c[0] + c[2]/2 < roi_center_x - offset]
        right_cnts = [c for c in glasses_candidates if c[0] + c[2]/2 > roi_center_x + offset]
        if is_night:
            if len(left_cnts) >= 2 and len(right_cnts) >= 2:
                left_avg_area = np.mean([c[4] for c in left_cnts]) if left_cnts else 0
                right_avg_area = np.mean([c[4] for c in right_cnts]) if right_cnts else 0
                if left_avg_area > 150 and right_avg_area > 150:
                    return "glasses"
        else:
            if len(left_cnts) >= 1 and len(right_cnts) >= 1:
                return "glasses"
    
    # 单一大面积候选（如大框眼镜）
    if len(glasses_candidates) >= 1:
        best = max(glasses_candidates, key=lambda c: c[4])
        min_single_area = 800 if is_night else 500
        if best[4] > min_single_area:
            return "glasses"
    
    return "normal"


def check_video_brightness_level(video_path: str, sample_frames: int = 10) -> str:
    """
    预检测视频的整体亮度等级
    采样 sample_frames 帧，计算平均灰度值，返回 'night', 'day', 'strong'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "day"
    brightness_sum = 0.0
    count = 0
    for _ in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_sum += np.mean(gray)
        count += 1
    cap.release()
    if count == 0:
        return "day"
    avg = brightness_sum / count
    print(f"--{os.path.basename(video_path)}--")
    print(f"  亮度预检: 平均亮度={avg:.1f}")
    if avg < 85:
        return "night"
    elif avg >= 200:
        return "strong"
    else:
        return "day"


# ========================= Yunet人脸检测模块 =========================
class FaceDetector:
    """使用OpenCV YuNet ONNX模型的人脸检测器"""
    def __init__(self, model_path="face_detection_yunet.onnx", input_size=(320, 320),
                 score_threshold=0.5, nms_threshold=0.3, top_k=5000):
        # 创建YuNet检测器实例
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path, config="", input_size=input_size,
            score_threshold=score_threshold, nms_threshold=nms_threshold, top_k=top_k
        )
    def set_input_size(self, w, h):
        """更新输入图像尺寸（用于缩放检测）"""
        self.detector.setInputSize((w, h))
    def detect(self, image):
        """检测人脸，返回 (num_faces, 15个值) 的数组，每行包含 [x1,y1,w,h,score,...]"""
        _, faces = self.detector.detect(image)
        return faces


class LandmarkExtractor:
    """使用dlib 68点面部关键点提取器"""
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.predictor = dlib.shape_predictor(predictor_path)
    def get_landmarks(self, gray, rect):
        """
        输入灰度图和人脸矩形dlib.rectangle，返回68个关键点的numpy数组 (68,2)
        """
        shape = self.predictor(gray, rect)
        return np.array([[p.x, p.y] for p in shape.parts()])


# ========================= Dlib特征点提取模块 =========================
def eye_aspect_ratio(eye):
    """
    计算眼睛纵横比 EAR (Eye Aspect Ratio)
    eye: 包含6个关键点的数组，顺序为左眼或右眼轮廓
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def mouth_aspect_ratio(mouth):
    """
    计算嘴巴纵横比 MAR (Mouth Aspect Ratio)
    mouth: 20个关键点 (48-67)，取其中代表性点：嘴角左(0)、右(6)，上嘴唇(3)，下嘴唇(9)
    MAR = 垂直距离 / 水平距离
    """
    left = mouth[0]
    right = mouth[6]
    upper = mouth[3]
    lower = mouth[9]
    v = dist.euclidean(upper, lower)
    h = dist.euclidean(left, right)
    return v / (h + 1e-6)


def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为欧拉角（俯仰pitch、偏航yaw、滚转roll）
    避免角度跳变（将大于180度的角调整到[-180,180]范围）
    """
    angles = list(cv2.RQDecomp3x3(R)[0])
    for i in range(3):
        if angles[i] > 180:
            angles[i] -= 360
        elif angles[i] <= -180:
            angles[i] += 360
        if abs(angles[i]) > 90 and abs(angles[i]) < 270:
            angles[i] = angles[i] - 180 if angles[i] > 0 else angles[i] + 180
    return angles[0], angles[1], angles[2]


def estimate_head_pose(p3d, p2d, cam_matrix, dist_coeffs, prev_rvec=None, prev_tvec=None):
    """
    通过PNP算法估计头部姿态
    p3d: 3D世界坐标点（预定义的6个关键点）
    p2d: 对应的2D图像坐标点（由 landmarks 根据索引提取）
    cam_matrix: 相机内参矩阵
    dist_coeffs: 畸变系数
    prev_rvec, prev_tvec: 上一帧的旋转/平移向量，用于加速迭代
    返回 (rvec, tvec, (pitch, yaw, roll))
    """
    flags = cv2.SOLVEPNP_ITERATIVE
    if prev_rvec is not None and prev_tvec is not None:
        success, rvec, tvec = cv2.solvePnP(p3d, p2d, cam_matrix, dist_coeffs,
                                           rvec=prev_rvec, tvec=prev_tvec,
                                           useExtrinsicGuess=True, flags=flags)
    else:
        success, rvec, tvec = cv2.solvePnP(p3d, p2d, cam_matrix, dist_coeffs, flags=flags)
    if not success:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    euler = rotation_matrix_to_euler(R)
    return rvec, tvec, euler


# ========================= 疲劳判定模块 =========================
class FatigueDetector:
    """
    疲劳检测核心类：校准、帧处理、疲劳等级判定
    包含六种场景独立阈值和判定逻辑
    """
    def __init__(self, yunet_path="face_detection_yunet.onnx",
                 predictor_path="shape_predictor_68_face_landmarks.dat", scale=0.5,
                 enable_glare_removal=True):
        # 组件初始化
        self.face_detector = FaceDetector(yunet_path)
        self.landmark_extractor = LandmarkExtractor(predictor_path)
        self.scale = scale                     # 处理缩放比例，提高速度
        self.enable_glare_removal = enable_glare_removal

        # 头部姿态所需的3D参考点（6个关键点的世界坐标，单位毫米）
        self.landmarks_3d = np.array([
            (0.0, 0.0, 0.0),           # 鼻尖
            (0.0, -150.0, -100.0),     # 下巴
            (-150.0, 30.0, -100.0),    # 左眼左角
            (150.0, 30.0, -100.0),     # 右眼右角
            (-100.0, -60.0, -100.0),   # 左嘴角
            (100.0, -60.0, -100.0)     # 右嘴角
        ], dtype=np.float32)
        # 对应的dlib 2D关键点索引
        self.indices_2d = [30, 8, 36, 45, 48, 54]
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1), dtype=np.float32)

        # 校准相关参数
        self.calib_frames = 30               # 校准所需帧数
        self.is_calibrating = True           # 是否正在校准
        self.calib_count = 0
        self.calib_ear = []                  # 存储校准期间的EAR
        self.calib_mar = []                  # 存储校准期间的MAR
        self.calib_pitch = []                # 存储原始俯仰角
        self.calib_yaw = []                  # 存储原始偏航角
        self.calib_roll = []                 # 存储原始滚转角
        self.calib_brightness = []
        self.calib_retry_count = 0
        self.max_calib_retries = 3

        # 默认阈值（会随场景重设）
        self.EAR_THRESHOLD = 0.20
        self.MAR_THRESHOLD = 0.5
        self.PITCH_TH = 20.0
        self.YAW_TH = 25.0
        self.ROLL_TH = 15.0
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        self.roll_offset = 0.0

        # 状态计数器
        self.mouth_open_frames = 0
        self.head_tilt_frames = 0
        self.eye_closed_consecutive = 0
        self.eye_closed_history = deque(maxlen=90)   # 用于PERCLOS计算

        self.alarm_count = 0
        self.angle_buffer = deque(maxlen=5)          # 平滑头部角度
        self.prev_rvec = None
        self.prev_tvec = None
        self.fatigue_history = deque(maxlen=10)      # 疲劳等级平滑
        self.alarm_manager = AlarmManager()

        self.fatigue_lock_counter = 0
        self.LOCK_FRAMES = 30                        # 重度疲劳后锁帧数

        self.night_mode = False
        self.night_brightness_threshold = 60

        self.mouth_ratio_factor = 1.4

        self.raw_pitch = 0.0
        self.raw_yaw = 0.0
        self.raw_roll = 0.0

        self.frame_count = 0
        self.HEAD_SKIP_FRAMES = 200                  # 前200帧不进行头部疲劳判定

        self.brightness_level = "day"                # 'night', 'day', 'strong'
        self.is_glasses_video = False
        self.calib_glasses_count = 0
        self.scene_type = "day_normal"               # 六种场景之一

        self.calib_ear_threshold = None
        self.calib_mar_threshold = None
        self.calib_pitch_offset = None
        self.calib_yaw_offset = None
        self.calib_roll_offset = None

    def set_brightness_level(self, level: str):
        self.brightness_level = level

    def set_camera_matrix(self, w, h):
        """根据图像尺寸构造相机内参矩阵（假设焦距等于宽度）"""
        fx = fy = w
        cx, cy = w//2, h//2
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def reset_state(self):
        """重置所有疲劳相关状态（用于新视频开始时）"""
        self.mouth_open_frames = 0
        self.head_tilt_frames = 0
        self.eye_closed_consecutive = 0
        self.eye_closed_history.clear()
        for _ in range(90):
            self.eye_closed_history.append(False)
        self.alarm_count = 0
        self.angle_buffer.clear()
        self.prev_rvec = None
        self.prev_tvec = None
        self.fatigue_history.clear()
        self.fatigue_lock_counter = 0
        self.frame_count = 0

    def _calibrate_frame(self, ear, mar, pitch, yaw, roll, brightness, is_glasses_frame=False):
        """
        处理校准帧数据，收集特征，根据亮度和眼镜判断场景，计算最终阈值
        返回 True 表示校准完成，False 表示需要重试，None 表示收集尚未完成
        """
        self.calib_ear.append(ear)
        self.calib_mar.append(mar)
        self.calib_pitch.append(pitch)
        self.calib_yaw.append(yaw)
        self.calib_roll.append(roll)
        self.calib_brightness.append(brightness)
        if is_glasses_frame:
            self.calib_glasses_count += 1
        self.calib_count += 1

        if self.calib_count >= self.calib_frames:
            avg_bright = np.mean(self.calib_brightness)
            if avg_bright < 55:
                return "fail_brightness"           # 环境过暗，无法校准

            # 检查头部摆动是否过大，过大则重置校准
            pitch_std = np.std(self.calib_pitch)
            yaw_std = np.std(self.calib_yaw)
            if pitch_std > 15.0 or yaw_std > 15.0:
                if self.calib_retry_count < self.max_calib_retries:
                    self.calib_retry_count += 1
                    print(f"  警告：头部摆动过大(俯仰{pitch_std:.1f}°,偏航{yaw_std:.1f}°)，重新校准（{self.calib_retry_count}次）")
                    self.calib_count = 0
                    self.calib_ear.clear()
                    self.calib_mar.clear()
                    self.calib_pitch.clear()
                    self.calib_yaw.clear()
                    self.calib_roll.clear()
                    self.calib_brightness.clear()
                    self.calib_glasses_count = 0
                    return False
                else:
                    print("  警告：重试次数用尽，强制完成校准")

            # 确定是否为眼镜场景（30帧中≥15帧检测到眼镜）
            self.is_glasses_video = (self.calib_glasses_count >= 15)
            glasses_info = f"{self.calib_glasses_count}/{self.calib_frames} 帧检测到眼镜" + (" → 确定为眼镜场景" if self.is_glasses_video else " → 场景为 normal")
            print(f"  校准检测：{glasses_info}")

            # 组合场景类型字符串
            if self.brightness_level == "night":
                if self.is_glasses_video:
                    self.scene_type = "night_glasses"  #夜间有眼镜
                else:
                    self.scene_type = "night_normal"   #夜间正常（无眼镜）
            elif self.brightness_level == "strong":
                if self.is_glasses_video:
                    self.scene_type = "strong_glasses" #强光有眼镜
                else:
                    self.scene_type = "strong_normal"  #强光正常
            else:
                if self.is_glasses_video:
                    self.scene_type = "day_glasses"   #日间有眼镜
                else:
                    self.scene_type = "day_normal"    #日间正常

            mean_ear = np.mean(self.calib_ear)
            std_ear = np.std(self.calib_ear)
            mean_mar = np.mean(self.calib_mar)
            std_mar = np.std(self.calib_mar)

            # 根据六种场景设定不同的EAR/MAR阈值
            if self.scene_type == "night_normal":
                self.EAR_THRESHOLD = max(0.18, mean_ear - 2 * std_ear)
                self.MAR_THRESHOLD = min(0.70, mean_mar + 1.5 * std_mar)
                mode_str = "夜间非眼镜"
            elif self.scene_type == "night_glasses":
                self.EAR_THRESHOLD = max(0.15, mean_ear - 1.5 * std_ear)
                self.MAR_THRESHOLD = min(0.60, mean_mar + 1.5 * std_mar)
                mode_str = "夜间眼镜"
            elif self.scene_type == "day_normal":
                self.EAR_THRESHOLD = max(0.18, mean_ear - 0.8 * std_ear)
                self.MAR_THRESHOLD = min(0.60, mean_mar + 0.8 * std_mar)
                mode_str = "日间非眼镜"
            elif self.scene_type == "day_glasses":
                self.EAR_THRESHOLD = max(0.15, mean_ear - 0.5 * std_ear)
                self.MAR_THRESHOLD = min(0.60, mean_mar + 1.5 * std_mar)
                mode_str = "日间眼镜"
            elif self.scene_type == "strong_normal":
                self.EAR_THRESHOLD = max(0.18, mean_ear - 1.2 * std_ear)
                self.MAR_THRESHOLD = min(0.70, mean_mar + 1.5 * std_mar)
                mode_str = "强光非眼镜"
            else:  # strong_glasses
                self.EAR_THRESHOLD = max(0.15, mean_ear - 1.0 * std_ear)
                self.MAR_THRESHOLD = min(0.60, mean_mar + 1.2 * std_mar)
                mode_str = "强光眼镜"

            # 头部姿态偏移量（校准姿势）
            self.pitch_offset = np.mean(self.calib_pitch)
            self.yaw_offset = np.mean(self.calib_yaw)
            self.roll_offset = np.mean(self.calib_roll)

            # 设定头部异常阈值（基于校准时的标准差）
            std_pitch = np.std(self.calib_pitch)
            std_yaw = np.std(self.calib_yaw)
            std_roll = np.std(self.calib_roll)
            self.PITCH_TH = max(12.0, std_pitch * 5.0) * 0.25
            self.YAW_TH   = max(15.0, std_yaw   * 5.0) * 1.0
            self.ROLL_TH  = max(10.0, std_roll  * 5.0) * 1.2

            self.is_calibrating = False
            # 保存校准结果供外部查询
            self.calib_ear_threshold = self.EAR_THRESHOLD
            self.calib_mar_threshold = self.MAR_THRESHOLD
            self.calib_pitch_offset = self.pitch_offset
            self.calib_yaw_offset = self.yaw_offset
            self.calib_roll_offset = self.roll_offset

            print(f"  校准完成({mode_str}): EAR阈值={self.EAR_THRESHOLD:.3f}, MAR阈值={self.MAR_THRESHOLD:.3f}, "
                  f"头部位移({self.pitch_offset:.1f},{self.yaw_offset:.1f},{self.roll_offset:.1f})")
            return True
        return None

    def process_frame(self, frame, record_alarm=True, show_alarm_count=False):
        """
        处理单帧图像，返回带标注的图像和疲劳信息字典
        record_alarm: 是否触发报警（批量分析时设为False）
        show_alarm_count: 是否在图像上显示报警次数
        """
        self.frame_count += 1

        # 预处理：自动增强
        enhanced_frame = auto_enhance(frame)
        display_frame = frame.copy()
        proc_frame = enhanced_frame

        h, w = proc_frame.shape[:2]
        small_w = int(w * self.scale)
        small_h = int(h * self.scale)
        small_frame = cv2.resize(proc_frame, (small_w, small_h))

        # 人脸检测（缩放后）
        self.face_detector.set_input_size(small_w, small_h)
        faces = self.face_detector.detect(small_frame)

        overall_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.night_mode = (overall_brightness < self.night_brightness_threshold)

        # 校准阶段：检测当前帧是否戴眼镜（用于最终场景判定）
        if self.is_calibrating:
            if faces is not None and len(faces) > 0:
                face = faces[0]
                x1_s, y1_s, w_s, h_s = face[:4].astype(int)
                x1_s = max(0, x1_s); y1_s = max(0, y1_s)
                w_s = min(w_s, small_w - x1_s); h_s = min(h_s, small_h - y1_s)
                if w_s > 0 and h_s > 0:
                    inv_scale = 1.0 / self.scale
                    x1 = int(x1_s * inv_scale)
                    y1 = int(y1_s * inv_scale)
                    x2 = int((x1_s + w_s) * inv_scale)
                    y2 = int((y1_s + h_s) * inv_scale)
                    scene = classify_scene(frame, (x1, y1, x2, y2))
                    glasses_frame = (scene == "glasses")
                else:
                    glasses_frame = False
            else:
                glasses_frame = False
        else:
            glasses_frame = (self.scene_type.endswith("glasses"))

        # 如果是眼镜场景，对眼睛区域进行CLAHE增强，改善眼部特征
        if not self.is_calibrating and "glasses" in self.scene_type and faces is not None and len(faces) > 0:
            face = faces[0]
            x1_s, y1_s, w_s, h_s = face[:4].astype(int)
            x1_s = max(0, x1_s); y1_s = max(0, y1_s)
            w_s = min(w_s, small_w - x1_s); h_s = min(h_s, small_h - y1_s)
            if w_s > 0 and h_s > 0:
                inv_scale = 1.0 / self.scale
                x1 = int(x1_s * inv_scale)
                y1 = int(y1_s * inv_scale)
                x2 = int((x1_s + w_s) * inv_scale)
                y2 = int((y1_s + h_s) * inv_scale)
                frame = enhance_eye_region(frame, (x1, y1, x2, y2))
                small_frame = cv2.resize(frame, (small_w, small_h))
                small_frame = auto_enhance(small_frame)

        # 重新检测人脸（因为可能增强了图像）
        self.face_detector.set_input_size(small_w, small_h)
        faces = self.face_detector.detect(small_frame)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_small)

        self.face_detector.set_input_size(small_w, small_h)
        faces = self.face_detector.detect(small_frame)
        frame_copy = display_frame.copy()

        # 初始化返回值
        ear_avg = mar = 0.0
        pitch = yaw = roll = 0.0
        perclos_ratio = 0.0
        eye_level = mouth_level = head_level = 0
        detail = ""
        eye_closed_consecutive = 0
        mouth_open_continuous = 0
        head_tilt_continuous = 0
        max_level = 0

        if faces is not None and len(faces) > 0:
            face = faces[0]
            x1_s, y1_s, w_s, h_s = face[:4].astype(int)
            x1_s = max(0, x1_s); y1_s = max(0, y1_s)
            w_s = min(w_s, small_w - x1_s); h_s = min(h_s, small_h - y1_s)
            if w_s > 0 and h_s > 0:
                inv_scale = 1.0 / self.scale
                x1 = int(x1_s * inv_scale)
                y1 = int(y1_s * inv_scale)
                x2 = int((x1_s + w_s) * inv_scale)
                y2 = int((y1_s + h_s) * inv_scale)

                # 绘制人脸框
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0,255,0), 2)

                # 获取68个关键点（在小图上提取后缩放回原始尺寸）
                dlib_rect = dlib.rectangle(x1_s, y1_s, x1_s + w_s, y1_s + h_s)
                landmarks_small = self.landmark_extractor.get_landmarks(gray_small, dlib_rect)
                landmarks = (landmarks_small * inv_scale).astype(np.int32)

                # 绘制关键点（颜色区分区域）
                for i, (x, y) in enumerate(landmarks):
                    if 36 <= i <= 47:
                        color = (0,255,0)      # 眼睛区域绿色
                    elif 48 <= i <= 67:
                        color = (0,0,255)      # 嘴巴区域红色
                    else:
                        color = (255,0,0)      # 其他蓝色
                    cv2.circle(frame_copy, (x, y), 2, color, -1)

                # 计算眼睛纵横比 EAR
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                ear_avg = (ear_left + ear_right) / 2.0

                # 眼镜场景下EAR阈值适当降低（因为眼镜可能遮挡眼睑）
                if "glasses" in self.scene_type:
                    ear_threshold = max(0.15, self.EAR_THRESHOLD * 0.7)
                else:
                    ear_threshold = self.EAR_THRESHOLD

                is_closed = ear_avg < ear_threshold
                self.eye_closed_history.append(is_closed)
                if len(self.eye_closed_history) > 0:
                    perclos_ratio = sum(self.eye_closed_history) / len(self.eye_closed_history)

                if is_closed:
                    self.eye_closed_consecutive += 1
                else:
                    self.eye_closed_consecutive = 0
                eye_closed_consecutive = self.eye_closed_consecutive

                # 根据六种场景设定眼部疲劳等级
                if self.scene_type == "night_glasses":
                    if perclos_ratio >= 0.50 or eye_closed_consecutive >= 8:
                        eye_level = 2
                    elif perclos_ratio >= 0.30 or eye_closed_consecutive >= 4:
                        eye_level = 1
                    else:
                        eye_level = 0
                elif self.scene_type == "night_normal":
                    if perclos_ratio >= 0.70 or eye_closed_consecutive >= 25:
                        eye_level = 2
                    elif perclos_ratio >= 0.50 or eye_closed_consecutive >= 10:
                        eye_level = 1
                    else:
                        eye_level = 0
                elif self.scene_type == "day_glasses":
                    if perclos_ratio >= 0.12 or eye_closed_consecutive >= 5:
                        eye_level = 2
                    elif perclos_ratio >= 0.5 or eye_closed_consecutive >= 2:
                        eye_level = 1
                    else:
                        eye_level = 0
                elif self.scene_type == "day_normal":
                    if perclos_ratio >= 0.60 or eye_closed_consecutive >= 40:
                        eye_level = 2
                    elif perclos_ratio >= 0.30 or eye_closed_consecutive >= 20:
                        eye_level = 1
                    else:
                        eye_level = 0
                elif self.scene_type == "strong_glasses":
                    if perclos_ratio >= 0.60 or eye_closed_consecutive >= 12:
                        eye_level = 2
                    elif perclos_ratio >= 0.40 or eye_closed_consecutive >= 6:
                        eye_level = 1
                    else:
                        eye_level = 0
                else:  # strong_normal
                    if perclos_ratio >= 0.80 or eye_closed_consecutive >= 30:
                        eye_level = 2
                    elif perclos_ratio >= 0.60 or eye_closed_consecutive >= 15:
                        eye_level = 1
                    else:
                        eye_level = 0

                # 计算嘴巴纵横比 MAR
                mouth = landmarks[48:68]
                mar = mouth_aspect_ratio(mouth)
                actual_mar_threshold = self.MAR_THRESHOLD * self.mouth_ratio_factor
                is_mouth_open = mar > actual_mar_threshold
                if is_mouth_open:
                    self.mouth_open_frames += 1
                else:
                    self.mouth_open_frames = max(0, self.mouth_open_frames - 1)
                mouth_open_continuous = self.mouth_open_frames

                # 根据场景设定打哈欠的轻度/重度连续帧阈值
                if self.scene_type == "night_glasses":
                    mouth_mild, mouth_severe = 15, 30
                elif self.scene_type == "night_normal":
                    mouth_mild, mouth_severe = 20, 40
                elif self.scene_type == "day_glasses":
                    mouth_mild, mouth_severe = 2, 5
                elif self.scene_type == "day_normal":
                    mouth_mild, mouth_severe = 3, 8
                elif self.scene_type == "strong_glasses":
                    mouth_mild, mouth_severe = 18, 35
                else:  # strong_normal
                    mouth_mild, mouth_severe = 25, 50

                if self.mouth_open_frames >= mouth_severe:
                    mouth_level = 2
                elif self.mouth_open_frames >= mouth_mild:
                    mouth_level = 1
                else:
                    mouth_level = 0

                # 头部姿态估计
                pts_2d = np.array([landmarks[i] for i in self.indices_2d], dtype=np.float32)
                rvec, tvec, euler = estimate_head_pose(
                    self.landmarks_3d, pts_2d, self.camera_matrix, self.dist_coeffs,
                    self.prev_rvec, self.prev_tvec
                )
                if euler is not None:
                    if rvec is not None and tvec is not None:
                        self.prev_rvec = rvec.copy()
                        self.prev_tvec = tvec.copy()
                    raw_pitch, raw_yaw, raw_roll = euler
                    self.raw_pitch, self.raw_yaw, self.raw_roll = raw_pitch, raw_yaw, raw_roll
                    # 减去校准时的偏移量
                    pitch = raw_pitch - self.pitch_offset
                    yaw = raw_yaw - self.yaw_offset
                    roll = raw_roll - self.roll_offset
                    self.angle_buffer.append((pitch, yaw, roll))
                    if len(self.angle_buffer) == self.angle_buffer.maxlen:
                        pitch = np.median([a[0] for a in self.angle_buffer])
                        yaw = np.median([a[1] for a in self.angle_buffer])
                        roll = np.median([a[2] for a in self.angle_buffer])
                    is_head_abnormal = (abs(pitch) > self.PITCH_TH or abs(yaw) > self.YAW_TH or abs(roll) > self.ROLL_TH)
                    if is_head_abnormal:
                        self.head_tilt_frames += 1
                    else:
                        self.head_tilt_frames = max(0, self.head_tilt_frames - 1)
                    head_tilt_continuous = int(self.head_tilt_frames)
                else:
                    self.head_tilt_frames = max(0, self.head_tilt_frames - 1)
                    head_tilt_continuous = int(self.head_tilt_frames)

                # 头部疲劳等级（前200帧跳过）
                if self.frame_count < self.HEAD_SKIP_FRAMES:
                    head_level = 0
                else:
                    if self.scene_type == "night_glasses":
                        head_mild, head_severe = 30, 60
                    elif self.scene_type == "night_normal":
                        head_mild, head_severe = 5, 10
                    elif self.scene_type == "day_glasses":
                        head_mild, head_severe = 5, 15
                    elif self.scene_type == "day_normal":
                        head_mild, head_severe = 40, 70
                    elif self.scene_type == "strong_glasses":
                        head_mild, head_severe = 40, 70
                    else:  # strong_normal
                        head_mild, head_severe = 35, 65
                    
                    if head_tilt_continuous >= head_severe:
                        head_level = 2
                    elif head_tilt_continuous >= head_mild:
                        head_level = 1
                    else:
                        head_level = 0

                # 校准阶段处理
                if self.is_calibrating:
                    calib_res = self._calibrate_frame(ear_avg, mar, pitch, yaw, roll, brightness, glasses_frame)
                    if calib_res == "fail_brightness":
                        frame_copy = draw_chinese_text(frame_copy, "环境过暗，无法检测", (10,60), 20, (0,0,255))
                        return frame_copy, None
                    progress = f"校准中: {self.calib_count}/{self.calib_frames}"
                    frame_copy = draw_chinese_text(frame_copy, progress, (10,60), 20, (255,255,0))
                    frame_copy = draw_chinese_text(frame_copy, "请保持正脸正常状态", (10,90), 18, (255,255,0))
                    return frame_copy, None

                # 综合疲劳等级（取眼部、嘴部、头部的最大值）
                max_level_i = max(eye_level, mouth_level, head_level)

                # 疲劳锁定：一旦触发重度疲劳，后续 LOCK_FRAMES 帧强制输出重度疲劳，避免频繁跳变
                if self.fatigue_lock_counter > 0:
                    max_level_i = 2
                    self.fatigue_lock_counter -= 1
                elif max_level_i == 2 and record_alarm:
                    self.fatigue_lock_counter = self.LOCK_FRAMES
                    self.alarm_count += 1
                    self.alarm_manager.trigger_beep_async()
                    self.alarm_manager.activate_red_flash(duration=1.0)

                # 平滑疲劳等级（取最近10帧的中位数）
                self.fatigue_history.append(max_level_i)
                max_level = int(np.median(self.fatigue_history))

                # 生成疲劳原因文本
                reasons = []
                if eye_level == 2:
                    reasons.append(f"眼部重度(PER{perclos_ratio*100:.0f}%/闭{eye_closed_consecutive}帧)")
                elif eye_level == 1:
                    reasons.append(f"眼部轻度(PER{perclos_ratio*100:.0f}%/闭{eye_closed_consecutive}帧)")
                if mouth_level == 2:
                    reasons.append(f"哈欠重度({self.mouth_open_frames}帧)")
                elif mouth_level == 1:
                    reasons.append(f"哈欠轻度({self.mouth_open_frames}帧)")
                if head_level == 2:
                    reasons.append("头部严重异常")
                elif head_level == 1:
                    reasons.append("头部异常")
                detail = " ".join(reasons)

        else:
            # 未检测到人脸时，缓慢降低各项计数器
            self.mouth_open_frames = max(0, self.mouth_open_frames - 2)
            self.head_tilt_frames = max(0, self.head_tilt_frames - 2)
            self.eye_closed_consecutive = max(0, self.eye_closed_consecutive - 1)
            if self.is_calibrating:
                frame_copy = draw_chinese_text(frame_copy, f"校准中: {self.calib_count}/{self.calib_frames}", (10,60), 20, (255,255,0))
                frame_copy = draw_chinese_text(frame_copy, "未检测到人脸", (10,90), 18, (255,255,0))
                return frame_copy, None

        # ========== 可视化信息绘制 ==========
        start_x, start_y = 10, 30
        line_h = 28
        level_name = {"night": "夜间", "day": "日间", "strong": "强光"}
        scene_display = level_name.get(self.brightness_level, "日间") + ("眼镜" if "glasses" in self.scene_type else "无眼镜")
        frame_copy = draw_chinese_text(frame_copy, f"场景: {scene_display}", (10, frame.shape[0]-90), 16, (200,200,200), (30,30,30))
        frame_copy = draw_chinese_text(frame_copy, f"EAR: {ear_avg:.2f}  MAR: {mar:.2f}  PERCLOS: {perclos_ratio*100:.1f}%",
                                       (start_x, start_y + line_h*0), 18, (255,255,255), (30,30,30))
        frame_copy = draw_chinese_text(frame_copy, f"Raw(P/Y/R): {self.raw_pitch:.1f}/{self.raw_yaw:.1f}/{self.raw_roll:.1f}°",
                                       (start_x, start_y + line_h*1), 18, (255,255,255), (30,30,30))
        eye_str = "眼部: " + ("重度疲劳" if eye_level==2 else ("轻度疲劳" if eye_level==1 else "正常"))
        mouth_str = "嘴部: " + ("重度打哈欠" if mouth_level==2 else ("打哈欠" if mouth_level==1 else "正常"))
        head_str = "头部: " + ("重度头偏" if head_level==2 else ("头偏" if head_level==1 else "正常"))
        frame_copy = draw_chinese_text(frame_copy, f"{eye_str}  |  {mouth_str}  |  {head_str}",
                                       (start_x, start_y + line_h*2), 16, (200,200,200), (30,30,30))
        if show_alarm_count:
            frame_copy = draw_chinese_text(frame_copy, f"报警次数: {self.alarm_count}",
                                           (start_x, start_y + line_h*3), 16, (255,255,0), (30,30,30))

        # 右侧状态显示
        right_x = frame.shape[1] - 260
        right_y = 30
        if max_level == 2:
            state_text = "警告！！！重度疲劳"
            text_color = (255, 0, 0)
            font_size = 28
            # 添加黑色阴影效果
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                frame_copy = draw_chinese_text(frame_copy, state_text, (right_x + dx, right_y + dy), font_size, (0, 0, 0), None)
            frame_copy = draw_chinese_text(frame_copy, state_text, (right_x, right_y), font_size, text_color, None)
        elif max_level == 1:
            state_text = "轻度疲劳"
            text_color = (0, 165, 255)   # 橙色
            font_size = 27
            frame_copy = draw_chinese_text(frame_copy, state_text, (right_x, right_y), font_size, text_color, None)
        else:
            state_text = "正常"
            text_color = (0, 255, 0)
            font_size = 27
            frame_copy = draw_chinese_text(frame_copy, state_text, (right_x, right_y), font_size, text_color, None)

        # 更新红色闪烁报警效果
        self.alarm_manager.update_flash()
        if max_level == 2:
            frame_copy = self.alarm_manager.draw_red_flash(frame_copy)

        # 校准中或未检测到人脸时不返回疲劳信息
        if self.is_calibrating or faces is None or len(faces) == 0:
            return frame_copy, None

        # 组装疲劳信息字典
        fatigue_info = {
            'ear': ear_avg, 'mar': mar, 'perclos': perclos_ratio,
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'eye_level': eye_level, 'mouth_level': mouth_level, 'head_level': head_level,
            'max_level': max_level, 'drowsy': 1 if max_level > 0 else 0,
            'eye_closed_frames': eye_closed_consecutive,
            'mouth_open_frames': mouth_open_continuous,
            'head_tilt_frames': head_tilt_continuous,
            'reason': detail
        }
        return frame_copy, fatigue_info


# ========================= 报警与可视化模块 =========================
class AlarmManager:
    """声音与红色闪烁报警管理"""
    def __init__(self):
        self.beep_active = False
        self.red_flash_active = False
        self.red_flash_end_time = 0
        self.last_beep_time = 0
        self.beep_cooldown = 0.8            # 蜂鸣冷却时间（秒）

    def trigger_beep_async(self):
        """异步触发蜂鸣声（避免阻塞主线程）"""
        if not SOUND_AVAILABLE:
            return
        current_time = time.time()
        if current_time - self.last_beep_time < self.beep_cooldown:
            return
        if not self.beep_active:
            self.beep_active = True
            self.last_beep_time = current_time
            threading.Thread(target=self._play_beep, daemon=True).start()

    def _play_beep(self):
        """播放蜂鸣声的实际函数"""
        try:
            winsound.Beep(1000, 300)
        except:
            pass
        finally:
            self.beep_active = False

    def activate_red_flash(self, duration=1.0):
        """激活红色闪烁，持续 duration 秒"""
        self.red_flash_active = True
        self.red_flash_end_time = time.time() + duration

    def update_flash(self):
        """更新闪烁状态，超时后关闭"""
        if self.red_flash_active and time.time() > self.red_flash_end_time:
            self.red_flash_active = False

    def draw_red_flash(self, frame):
        """
        在图像边缘绘制红色闪烁边框（根据时间正弦变化产生呼吸感）
        """
        if not self.red_flash_active:
            return frame
        h, w = frame.shape[:2]
        flash_phase = (time.time() * 8) % 1.0
        thickness = 12 if flash_phase > 0.5 else 6
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), (0, 0, 255), thickness)
        alpha = 0.7 if flash_phase > 0.5 else 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame


def enhance_eye_region(img, face_rect):
    """
    对眼睛区域进行CLAHE增强，改善暗光下的眼部特征（尤其针对戴眼镜时）
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face_rect
    eye_y1 = max(0, y1 + int(0.25 * (y2 - y1)))
    eye_y2 = min(h, y1 + int(0.55 * (y2 - y1)))
    eye_x1 = max(0, x1 + int(0.2 * (x2 - x1)))
    eye_x2 = min(w, x1 + int(0.8 * (x2 - x1)))
    if eye_x2 > eye_x1 and eye_y2 > eye_y1:
        roi = img[eye_y1:eye_y2, eye_x1:eye_x2]
        if roi.size > 0:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
            l_enh = clahe.apply(l)
            enhanced = cv2.merge((l_enh, a, b))
            roi_enh = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            img[eye_y1:eye_y2, eye_x1:eye_x2] = roi_enh
    return img


def draw_chinese_text(img, text, position, font_size=20, color=(255,255,255), bg_color=None):
    """
    在OpenCV图像上绘制中文文本（使用PIL以便支持中文TrueType字体）
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except:
        font = ImageFont.load_default()
    if bg_color:
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle(bbox, fill=bg_color)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def print_performance_summary(total_frames, detected_frames, total_time, avg_alarm_delay, fatigue_frames):
    """打印性能统计信息"""
    fps = total_frames / total_time if total_time > 0 else 0
    fatigue_ratio = (fatigue_frames / detected_frames * 100) if detected_frames > 0 else 0
    print("\n" + "="*28+"性能总结"+"="*28)
    print(f"  总处理帧数: {total_frames}")
    print(f"  有效检测帧数: {detected_frames}")
    print(f"  总耗时: {total_time:.2f} s")
    print(f"  平均帧率: {fps:.2f} fps")
    if avg_alarm_delay >= 0:
        print(f"  平均报警响应延迟: {avg_alarm_delay*1000:.3f} s")
    else:
        print("  报警响应延迟: 未触发报警")
    print(f"  疲劳帧占比: {fatigue_ratio:.2f}% ({fatigue_frames}/{detected_frames})")
    print("="*60)


# ========================= 批量分析辅助模块 =========================
def read_label_file(file_path: str) -> List[int]:
    """
    读取真实疲劳标签文件
    支持格式：每行一个0/1，或者一串"012"的字符串，或者逗号分隔的数字
    返回列表，其中 >0 视为疲劳
    """
    labels = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if all(ch in '012' for ch in line):
                    for ch in line:
                        labels.append(1 if int(ch) > 0 else 0)
                else:
                    parts = line.replace(',', ' ').split()
                    if parts:
                        val = int(parts[-1])
                        labels.append(1 if val > 0 else 0)
    except FileNotFoundError:
        pass
    return labels


def calc_metrics(pred, truth):
    """计算准确率、误判率、漏检率"""
    if not pred or not truth:
        return {"accuracy":0.0, "false_alarm_rate":0.0, "miss_rate":0.0}
    min_len = min(len(pred), len(truth))
    pred = pred[:min_len]
    truth = truth[:min_len]
    tp = sum(1 for p,g in zip(pred,truth) if p==1 and g==1)
    tn = sum(1 for p,g in zip(pred,truth) if p==0 and g==0)
    fp = sum(1 for p,g in zip(pred,truth) if p==1 and g==0)
    fn = sum(1 for p,g in zip(pred,truth) if p==0 and g==1)
    acc = (tp+tn)/min_len if min_len>0 else 0
    far = fp/(fp+tn) if (fp+tn)>0 else 0
    miss = fn/(fn+tp) if (fn+tp)>0 else 0
    return {"accuracy":round(acc,4), "false_alarm_rate":round(far,4), "miss_rate":round(miss,4)}


def analyze_single_video(video_path, yunet_path, predictor_path):
    """
    批量分析单个视频，返回评估指标字典
    包含亮度检测、校准、逐帧预测、与真实标签对比
    """
    brightness_level = check_video_brightness_level(video_path)
    detector = FatigueDetector(yunet_path, predictor_path, scale=0.5)
    detector.set_brightness_level(brightness_level)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video_name": os.path.basename(video_path), "error": "无法打开视频"}
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return {"video_name": os.path.basename(video_path), "error": "视频无内容"}
    h, w = first_frame.shape[:2]
    detector.set_camera_matrix(w, h)

    # 执行校准（直到校准完成）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    calib_complete = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detector.process_frame(frame, record_alarm=False, show_alarm_count=False)
        if not detector.is_calibrating:
            calib_complete = True
            break
    if not calib_complete:
        cap.release()
        return {"video_name": os.path.basename(video_path), "error": "校准失败"}

    # 重置状态，正式分析
    detector.reset_state()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = 0
    pred = []                       # 预测疲劳帧列表（0/1）
    eye_closed = []                 # 记录每帧是否闭眼，用于PERCLOS

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        _, info = detector.process_frame(frame, record_alarm=False, show_alarm_count=False)
        if info is not None:
            pred.append(info['drowsy'])
            eye_closed.append(info['ear'] < detector.EAR_THRESHOLD)
        else:
            pred.append(0)
    cap.release()

    perclos = sum(eye_closed) / len(eye_closed) if eye_closed else 0.0
    base = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    gt = read_label_file(os.path.join(video_dir, f"{base}_drowsiness.txt"))

    metrics = calc_metrics(pred, gt)
    vid_name = os.path.basename(video_path)
    print(f"{vid_name}")
    print(f"  标签对齐: 预测帧数={len(pred)}, 真实帧数={len(gt)}, 差值={len(pred)-len(gt)}")
    print(f"  视频分析完成, PERCLOS={perclos*100:.1f}%")
    print(f"  综合疲劳: 准确率={metrics['accuracy']*100:.2f}%, 误判率={metrics['false_alarm_rate']*100:.2f}%, 漏检率={metrics['miss_rate']*100:.2f}%")

    # 保存中间文件
    det_file = os.path.join(video_dir, f"{base}_detection_results.txt")
    with open(det_file, 'w') as f:
        f.write(''.join(str(d) for d in pred))
    eval_file = os.path.join(video_dir, f"{base}_real.txt")
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(f"视频: {vid_name}\n")
        f.write(f"总帧数: {total_frames}, PERCLOS: {perclos*100:.2f}%\n")
        f.write(f"准确率: {metrics['accuracy']*100:.2f}%, 误判率: {metrics['false_alarm_rate']*100:.2f}%, 漏检率: {metrics['miss_rate']*100:.2f}%\n")
    print(f"  报告已保存: {det_file}, {eval_file}\n")

    return {
        "video_name": vid_name,
        "accuracy": metrics['accuracy'],
        "false_alarm_rate": metrics['false_alarm_rate'],
        "miss_rate": metrics['miss_rate']
    }


def evaluate_folder(folder_path, yunet_path="face_detection_yunet.onnx",
                    predictor_path="shape_predictor_68_face_landmarks.dat"):
    """
    批量分析文件夹内所有视频文件（递归查找 .mp4, .avi, .mov, .mkv）
    生成整体评估报告
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"文件夹 '{folder_path}' 不存在")
        return
    video_files = []
    for ext in ['.mp4','.avi','.mov','.mkv']:
        video_files.extend(folder.rglob(f'*{ext}'))
    video_files = list(set(video_files))
    if not video_files:
        print("未找到视频文件")
        return
    print(f"找到 {len(video_files)} 个视频，开始分析...")
    results = []
    for vf in video_files:
        res = analyze_single_video(str(vf), yunet_path, predictor_path)
        if "error" in res:
            print(f"跳过 {res['video_name']}: {res['error']}")
            continue
        results.append(res)

    if results:
        acc = np.mean([r['accuracy'] for r in results])
        far = np.mean([r['false_alarm_rate'] for r in results])
        miss = np.mean([r['miss_rate'] for r in results])
        report_path = os.path.join(folder_path, "overall_evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("驾驶员疲劳检测整体评估报告\n")
            f.write("="*60 + "\n")
            f.write(f"平均准确率: {acc*100:.2f}%\n")
            f.write(f"平均误判率: {far*100:.2f}%\n")
            f.write(f"平均漏检率: {miss*100:.2f}%\n")
            f.write("\n各视频详情：\n")
            for r in results:
                f.write(f"--{r['video_name']}--\n准确率={r['accuracy']*100:.2f}%, 误判率={r['false_alarm_rate']*100:.2f}%, 漏检率={r['miss_rate']*100:.2f}%\n")
        print(f"\n整体报告已保存至: {report_path}")
    else:
        print("没有成功分析的视频。")


# ========================= 主控制模块（实时/视频/批量） =========================
def run_realtime():
    """实时摄像头检测模式"""
    if not os.path.exists("face_detection_yunet.onnx") or not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("错误: 缺少模型文件")
        return False
    detector = FatigueDetector(enable_glare_removal=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return False
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        detector.set_camera_matrix(w, h)

    # 实时模式亮度检测
    brightness_vals = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(np.mean(gray))
    if brightness_vals:
        avg_bright = np.mean(brightness_vals)
        if avg_bright < 85:
            level = "night"
        elif avg_bright >= 200:
            level = "strong"
        else:
            level = "day"
        detector.set_brightness_level(level)
        print(f"实时模式亮度检测: 平均亮度={avg_bright:.1f}, 场景类型={level}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("实时检测模式... 按 q 退出")
    total_frames = 0
    detected_frames = 0
    fatigue_frames = 0
    start_time = time.time()
    alarm_delays = []          # 存储每次报警的延迟（秒）
    alarm_trigger_time = None  # 当前报警触发的时间戳

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        processed, info = detector.process_frame(frame, record_alarm=True, show_alarm_count=False)
        if info is not None:
            detected_frames += 1
            if info['max_level'] >= 2:
                fatigue_frames += 1
                alarm_trigger_time = time.time()   # 记录报警触发时刻
        cv2.imshow('驾驶员疲劳检测', processed)
        # 计算本次报警延迟（如果本帧触发了报警）
        if alarm_trigger_time is not None:
            delay = time.time() - alarm_trigger_time
            alarm_delays.append(delay)
            alarm_trigger_time = None  # 重置，避免重复记录
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    avg_alarm_delay = np.mean(alarm_delays) if alarm_delays else -1.0
    print_performance_summary(total_frames, detected_frames, total_time, avg_alarm_delay, fatigue_frames)
    return True

def run_video_file():
    """单视频文件检测模式"""
    if not os.path.exists("face_detection_yunet.onnx") or not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("错误: 缺少模型文件")
        return False
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")])
    if not video_path:
        return False

    # 预计算亮度等级
    cap = cv2.VideoCapture(video_path)
    brightness_sum = 0.0
    count = 0
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_sum += np.mean(gray)
        count += 1
    cap.release()
    avg_bright = brightness_sum / count if count>0 else 70
    if avg_bright < 85:
        level = "night"
    elif avg_bright >= 200:
        level = "strong"
    else:
        level = "day"

    detector = FatigueDetector(enable_glare_removal=True)
    detector.set_brightness_level(level)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return False
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        detector.set_camera_matrix(w, h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("正在进行初始化校准，请保持头部基本稳定...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detector.process_frame(frame, record_alarm=False, show_alarm_count=False)
        if not detector.is_calibrating:
            break
    else:
        cap.release()
        print("校准失败")
        return False

    detector.reset_state()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"处理视频: {video_path}")

    total_frames = 0
    detected_frames = 0
    fatigue_frames = 0
    start_time = time.time()
    alarm_delays = []
    alarm_trigger_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        processed, info = detector.process_frame(frame, record_alarm=True, show_alarm_count=False)
        if info is not None:
            detected_frames += 1
            if info['max_level'] >= 2:
                fatigue_frames += 1
                alarm_trigger_time = time.time()
        cv2.imshow('视频检测', processed)
        if alarm_trigger_time is not None:
            delay = time.time() - alarm_trigger_time
            alarm_delays.append(delay)
            alarm_trigger_time = None
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # 空格暂停
            while cv2.waitKey(0) & 0xFF not in [ord(' '), ord('q')]:
                pass
    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    avg_alarm_delay = np.mean(alarm_delays) if alarm_delays else -1.0
    print_performance_summary(total_frames, detected_frames, total_time, avg_alarm_delay, fatigue_frames)
    return True

def run_batch():
    """批量分析模式"""
    folder = input("请输入视频根文件夹路径: ").strip()
    evaluate_folder(folder)
    return True


def main():
    try:
        print("驾驶员疲劳检测算法 (六种独立场景：眼部/嘴部/头部全场景差异化)")
        selector = ModeSelector()
        mode = selector.run()
        if mode == "exit":
            return 0
        elif mode == "realtime":
            success = run_realtime()
        elif mode == "videofile":
            success = run_video_file()
        elif mode == "batch":
            success = run_batch()
        else:
            return 1
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
