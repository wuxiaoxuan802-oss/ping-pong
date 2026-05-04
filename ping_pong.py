# -*- coding: utf-8 -*-
"""
🏓 手勢乒乓球遊戲 — MediaPipe + Pygame
==========================================
使用攝影機即時偵測手部位置，以食指指尖控制球拍，
打擊在畫面中彈跳的球。

渲染引擎使用 Pygame，手部追蹤使用 MediaPipe Task API，
攝影機擷取使用 OpenCV。

操作方式：
  - 伸出手掌，食指指尖的水平位置即為球拍位置
  - 按 [空白鍵] 開始 / 重新開始遊戲
  - 按 [Q] 或 [ESC] 退出遊戲

依賴套件：
  uv add pygame opencv-python mediapipe numpy
"""

# ============================================================
# 抑制無害的警告訊息（必須在匯入 mediapipe 之前設定）
# ============================================================
import os
import warnings

# 抑制 TensorFlow Lite 的內部日誌（0=全部顯示, 1=隱藏 INFO, 2=隱藏 WARNING, 3=只顯示 ERROR）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 抑制 protobuf 的棄用警告
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# ============================================================
# 匯入所需模組
# ============================================================
import pygame               # Pygame：遊戲視窗、繪圖、音效
import cv2                  # OpenCV：攝影機擷取
import ctypes               # ctypes: 修改 MediaPipe Bug 的所需模組

# 猴子補丁 (Monkey Patch) 修復 MediaPipe 在 Windows + Python 3.11 以上版本無法載入 'free' 函數的 AttributeError 問題
try:
    _original_getattr = ctypes.CDLL.__getattr__
    def _patched_getattr(self, name):
        if name == 'free':
            try:
                return _original_getattr(self, name)
            except AttributeError:
                # 遇到 AttributeError 找不到 free 時，回傳一個什麼都不做的 Dummy 函數
                return ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda x: None)
        return _original_getattr(self, name)
    ctypes.CDLL.__getattr__ = _patched_getattr
except Exception as e:
    print("MediaPipe Patch Failed:", e)

import mediapipe as mp       # MediaPipe：手部關鍵點偵測
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np           # NumPy：數值計算、音效波形產生
import time                  # time：MediaPipe 時間戳記
import random                # random：球的初始方向隨機化
import math                  # math：音效波形計算

# ============================================================
# 遊戲常數設定
# ============================================================

# --- 視窗尺寸 ---
WINDOW_WIDTH = 1280      # 視窗寬度（提升為 720p HD）
WINDOW_HEIGHT = 720      # 視窗高度
FPS_TARGET = 30         # 目標 FPS（每秒更新幀數）

# --- 手部抓取與發射設定 ---
GRAB_RADIUS = 90              # 吸附距離閾值
THROW_THRESHOLD_SPEED = 30    # 發射速度閾值（每幀變動像素）
BALL_MAX_THROW_SPEED = 45     # 手拋出球的最大允許速度
HOLD_COOLDOWN_TICK = 15       # 抓到球後的發射防呆冷卻（幀）
RELEASE_COOLDOWN_TICK = 15    # 發射後的防止重複抓取冷卻（幀）

# --- 障礙物 (磚塊) 設定 ---
OBSTACLE_ROWS = 5             # 障礙物列數
OBSTACLE_COLS = 12            # 障礙物行數
OBSTACLE_MARGIN_TOP = 90      # 頂部預留空間
OBSTACLE_GAP = 9              # 障礙物之間的間隙
OBSTACLE_HEIGHT = 30          # 每個障礙物的高度
OBSTACLE_COLORS = [           # 霓虹科幻色系（更豐富的 5 色漸層）
    (255, 80, 180),   # 霓虹粉
    (80, 200, 255),   # 電光藍
    (120, 255, 100),  # 螢光綠
    (255, 180, 50),   # 琥珀金
    (180, 100, 255),  # 幻紫
]
OBSTACLE_SPEED_BASE = 4       # 障礙物基本移動速度

# --- 板子設定 ---
PADDLE_WIDTH = 200            # 接球板寬度
PADDLE_HEIGHT = 24            # 接球板高度
PADDLE_COLOR_BASE = (0, 200, 255)  # 接球板基本顏色
PADDLE_COLOR_GRAB = (255, 120, 50) # 捏合抓取時的顏色

# --- 球設定 ---
BALL_RADIUS = 18              # 球的半徑
BALL_SPEED_INIT = 12          # 球的初始速度（每幀移動像素）
BALL_SPEED_INCREMENT = 0.75   # 每次成功擊球後球速增加量
BALL_MAX_SPEED = 30           # 球速上限，避免過快

# --- 顏色定義（RGB 格式，Pygame 使用 RGB）---
COLOR_GRAB_EFFECT = (0, 255, 170)   # 抓取時的發光效果顏色
COLOR_BALL = (255, 200, 0)          # 球顏色：橘黃色
COLOR_BALL_GLOW = (200, 120, 0)     # 球外發光
COLOR_TEXT = (255, 255, 255)        # 文字顏色：白色
COLOR_SCORE = (255, 255, 0)         # 分數顏色：黃色
COLOR_OVERLAY = (30, 20, 20)        # 半透明覆蓋層顏色
COLOR_WALL_TOP = (255, 100, 100)    # 頂部反彈區域指示色
COLOR_TRAIL = (200, 150, 0)         # 球的拖尾效果顏色
COLOR_FINGER = (0, 255, 0)          # 指尖追蹤圓點顏色

# --- 遊戲狀態常數 ---
STATE_START = 0       # 開始畫面
STATE_PLAYING = 1     # 遊戲進行中
STATE_GAME_OVER = 2   # 遊戲結束

# --- 音效設定 ---
SOUND_SAMPLE_RATE = 44100  # 音效取樣率
SOUND_HIT_FREQ = 800       # 擊球音效頻率（Hz）
SOUND_HIT_DURATION = 0.05  # 擊球音效時長（秒）
SOUND_MISS_FREQ = 300      # 漏球音效頻率
SOUND_MISS_DURATION = 0.2  # 漏球音效時長
SOUND_BREAK_FREQ = 1200    # 打碎怪獸音效頻率
SOUND_BREAK_DURATION = 0.05
SOUND_LEVELUP_FREQ = 1600  # 過關音效頻率
SOUND_LEVELUP_DURATION = 0.2

# --- 粒子爆破系統設定 ---
PARTICLE_COUNT_MIN = 15           # 爆破粒子最少數量
PARTICLE_COUNT_MAX = 25           # 爆破粒子最多數量
PARTICLE_SPEED_MIN = 2            # 粒子最小初速
PARTICLE_SPEED_MAX = 8            # 粒子最大初速
PARTICLE_GRAVITY = 0.15           # 粒子重力加速度
PARTICLE_LIFETIME = 40            # 粒子存活幀數

# --- 星空背景設定 ---
STAR_COUNT = 80                   # 背景星星數量
STAR_TWINKLE_SPEED = 0.05         # 星星閃爍速度
STAR_DRIFT_SPEED = 0.3            # 星星緩慢漂移速度

# --- 螢幕震動設定 ---
SHAKE_INTENSITY = 12              # 震動強度（像素）
SHAKE_DURATION = 15               # 震動持續幀數

# --- 動畫計時設定 ---
PULSE_SPEED = 0.06                # 脈動呼吸速度
BLINK_SPEED = 0.08                # 文字閃爍速度

# --- MediaPipe 手部偵測設定 ---
HAND_MAX_NUM = 1                # 最多偵測幾隻手
HAND_DETECTION_CONFIDENCE = 0.7  # 手部偵測信心度閾值
HAND_TRACKING_CONFIDENCE = 0.5   # 手部追蹤信心度閾值
INDEX_FINGER_TIP = 8             # 食指指尖的 Landmark 編號
THUMB_TIP = 4                    # 拇指指尖的 Landmark 編號
PINCH_THRESHOLD = 40             # 食指與拇指捏合的距離閾值（像素）

# --- 手部骨架連線定義 ---
# 定義 21 個手部特徵點之間的連線關係
# 每個 tuple 代表兩個特徵點的編號，連起來形成手部骨架
HAND_CONNECTIONS = [
    # 拇指：手腕(0) → 拇指尖(4)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指：手腕(0) → 食指尖(8)
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指：食指根(5) → 中指尖(12)
    (5, 9), (9, 10), (10, 11), (11, 12),
    # 無名指：中指根(9) → 無名指尖(16)
    (9, 13), (13, 14), (14, 15), (15, 16),
    # 小指：無名指根(13) → 小指根(17)，手腕(0) → 小指尖(20)
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# 手部骨架繪圖顏色（RGB）
COLOR_LANDMARK_POINT = (255, 220, 0)    # 特徵點顏色：亮黃色
COLOR_LANDMARK_LINE = (200, 200, 200)   # 連線顏色：淺灰色


# ============================================================
# 粒子爆破系統
# ============================================================
class Particle:
    """
    粒子物件 — 用於怪獸被擊碎時的視覺爆破特效。

    每個粒子有獨立的位置、速度、顏色、大小和壽命。
    粒子會受到重力影響，隨時間逐漸變透明並縮小。
    """

    def __init__(self, x, y, color):
        """
        初始化粒子。

        參數：
            x (float): 爆發原點 X 座標
            y (float): 爆發原點 Y 座標
            color (tuple): 粒子基底顏色 (R, G, B)
        """
        self.x = x
        self.y = y
        # 隨機方向的速度分量
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(PARTICLE_SPEED_MIN, PARTICLE_SPEED_MAX)
        self.dx = math.cos(angle) * speed
        self.dy = math.sin(angle) * speed
        # 粒子大小隨機（2~6 像素）
        self.radius = random.uniform(2, 6)
        # 在基底色上加入隨機偏移，讓每顆粒子略有不同
        self.color = (
            min(255, color[0] + random.randint(-30, 30)),
            min(255, color[1] + random.randint(-30, 30)),
            min(255, color[2] + random.randint(-30, 30)),
        )
        # 壽命計時器
        self.lifetime = PARTICLE_LIFETIME
        self.max_lifetime = PARTICLE_LIFETIME

    def update(self):
        """
        更新粒子狀態：移動、施加重力、遞減壽命。

        回傳：
            bool: True 表示粒子仍存活，False 表示應移除
        """
        self.x += self.dx
        self.y += self.dy
        self.dy += PARTICLE_GRAVITY  # 重力向下加速
        self.lifetime -= 1
        # 隨時間縮小
        self.radius *= 0.97
        return self.lifetime > 0 and self.radius > 0.5

    def draw(self, surface):
        """
        繪製粒子到指定 Surface。

        透明度隨壽命遞減，營造漸隱效果。
        """
        # 依剩餘壽命比例計算透明度
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        r = max(0, min(255, self.color[0]))
        g = max(0, min(255, self.color[1]))
        b = max(0, min(255, self.color[2]))
        rad = max(1, int(self.radius))
        # 使用帶 Alpha 的 Surface 實現透明效果
        particle_surf = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surf, (r, g, b, alpha), (rad, rad), rad)
        surface.blit(particle_surf, (int(self.x) - rad, int(self.y) - rad))


# ============================================================
# 星空背景系統
# ============================================================
class StarField:
    """
    動態星空背景管理器。

    管理多顆「星星」粒子，模擬太空中閃爍漂浮的效果，
    疊加在攝影機畫面上方，增加科幻氛圍。
    """

    def __init__(self, count=STAR_COUNT):
        """
        初始化星空。

        參數：
            count (int): 星星數量
        """
        self.stars = []
        for _ in range(count):
            self.stars.append({
                'x': random.uniform(0, WINDOW_WIDTH),
                'y': random.uniform(0, WINDOW_HEIGHT),
                'size': random.uniform(1, 3),         # 星星大小
                'brightness': random.uniform(0, 1),   # 目前亮度（0~1）
                'phase': random.uniform(0, 2 * math.pi),  # 閃爍相位偏移
                'drift_dx': random.uniform(-STAR_DRIFT_SPEED, STAR_DRIFT_SPEED),
                'drift_dy': random.uniform(-STAR_DRIFT_SPEED * 0.5, STAR_DRIFT_SPEED * 0.5),
            })

    def update(self, frame_count):
        """
        更新星空狀態：閃爍亮度與緩慢飄移。

        參數：
            frame_count (int): 目前全域幀計數（用於正弦波計算閃爍）
        """
        for star in self.stars:
            # 正弦波閃爍：亮度在 0.2 ~ 1.0 之間波動
            star['brightness'] = 0.6 + 0.4 * math.sin(
                frame_count * STAR_TWINKLE_SPEED + star['phase']
            )
            # 緩慢漂移
            star['x'] += star['drift_dx']
            star['y'] += star['drift_dy']
            # 超出邊界時環繞 (wrap around)
            if star['x'] < 0:
                star['x'] = WINDOW_WIDTH
            elif star['x'] > WINDOW_WIDTH:
                star['x'] = 0
            if star['y'] < 0:
                star['y'] = WINDOW_HEIGHT
            elif star['y'] > WINDOW_HEIGHT:
                star['y'] = 0

    def draw(self, surface):
        """
        繪製所有星星到指定 Surface。

        每顆星星會根據當前亮度調整透明度與大小。
        """
        for star in self.stars:
            alpha = int(star['brightness'] * 200)
            size = max(1, int(star['size'] * star['brightness']))
            x, y = int(star['x']), int(star['y'])
            # 外層光暈（較淡）
            if size >= 2:
                glow_surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (200, 220, 255, alpha // 3),
                                   (size * 2, size * 2), size * 2)
                surface.blit(glow_surf, (x - size * 2, y - size * 2))
            # 核心亮點
            star_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(star_surf, (255, 255, 255, alpha),
                               (size, size), size)
            surface.blit(star_surf, (x - size, y - size))


# ============================================================
# 音效產生函式
# ============================================================
def generate_beep_sound(frequency, duration, sample_rate=SOUND_SAMPLE_RATE):
    """
    使用 NumPy 產生正弦波音效，並轉為 Pygame Sound 物件。

    參數：
        frequency (float): 音效頻率，單位 Hz
        duration (float): 音效持續時間，單位秒
        sample_rate (int): 取樣率

    回傳：
        pygame.mixer.Sound: 可播放的音效物件
    """
    # 產生時間軸：從 0 到 duration，共 sample_rate * duration 個取樣點
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 產生正弦波（振幅 0.3，避免太大聲）
    wave = 0.3 * np.sin(2 * math.pi * frequency * t)

    # 套用淡出效果（最後 20% 的音量漸減），避免喀喀聲
    fade_samples = int(num_samples * 0.2)
    if fade_samples > 0:
        fade = np.linspace(1.0, 0.0, fade_samples)
        wave[-fade_samples:] *= fade

    # 轉為 16-bit 整數格式（Pygame mixer 需要的格式）
    wave_int16 = (wave * 32767).astype(np.int16)

    # 建立立體聲（左右聲道相同）
    stereo = np.column_stack((wave_int16, wave_int16))

    # 轉為 Pygame Sound 物件
    return pygame.mixer.Sound(buffer=stereo.tobytes())


# ============================================================
# 繪圖輔助函式
# ============================================================
def draw_glowing_circle(surface, center, radius, color, glow_color, glow_layers=3):
    """
    繪製帶有發光效果的圓形（多層半透明同心圓模擬）。

    參數：
        surface: Pygame 繪圖表面
        center: 圓心座標 (x, y)
        radius: 圓半徑
        color: 主體顏色 (R, G, B)
        glow_color: 發光層顏色 (R, G, B)
        glow_layers: 發光層數
    """
    # 由外到內繪製漸層發光效果
    for i in range(glow_layers, 0, -1):
        glow_radius = radius + i * 5
        # 建立帶 alpha 通道的暫存 Surface
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        # 透明度隨層數遞減（越外層越透明）
        alpha = max(10, int(40 / i))
        glow_with_alpha = (*glow_color, alpha)
        pygame.draw.circle(glow_surf, glow_with_alpha,
                           (glow_radius, glow_radius), glow_radius)
        # 將發光層繪製到主畫面
        surface.blit(glow_surf,
                     (center[0] - glow_radius, center[1] - glow_radius))

    # 繪製主體圓形
    pygame.draw.circle(surface, color, center, radius)

    # 內部高光效果（模擬光澤感）
    highlight_x = center[0] - radius // 3
    highlight_y = center[1] - radius // 3
    pygame.draw.circle(surface, (255, 255, 255), (highlight_x, highlight_y), radius // 3)


def draw_text_with_shadow(surface, text, position, font, color=(255, 255, 255),
                           shadow_offset=2):
    """
    繪製帶陰影的文字，增加可讀性。

    參數：
        surface: Pygame 繪圖表面
        text: 要顯示的文字字串
        position: 文字左上角位置 (x, y)
        font: Pygame 字型物件
        color: 文字顏色 (R, G, B)
        shadow_offset: 陰影偏移量（像素）
    """
    # 先繪製黑色陰影
    shadow_surface = font.render(text, True, (0, 0, 0))
    surface.blit(shadow_surface,
                 (position[0] + shadow_offset, position[1] + shadow_offset))
    # 再繪製主體文字
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)


def draw_centered_text(surface, text, center_x, y, font, color=(255, 255, 255)):
    """
    繪製水平置中的文字。

    參數：
        surface: Pygame 繪圖表面
        text: 要顯示的文字字串
        center_x: 水平置中參考 X 座標
        y: 文字頂部 Y 座標
        font: Pygame 字型物件
        color: 文字顏色 (R, G, B)
    """
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(midtop=(center_x, y))
    # 先畫陰影
    shadow_surface = font.render(text, True, (0, 0, 0))
    shadow_rect = shadow_surface.get_rect(midtop=(center_x + 2, y + 2))
    surface.blit(shadow_surface, shadow_rect)
    # 再畫主體
    surface.blit(text_surface, text_rect)


def draw_monster(surface, obs, frame_count=0):
    """
    繪製一隻外星小怪獸作為障礙物。
    具有半圓弧頂、鋸齒波浪腳、漸變色身體以及發光雙眼。
    
    參數:
        surface: Pygame 繪圖表面
        obs: 障礙物字典，包含 rect, color 與 dx
        frame_count: 全域幀計數，用於動畫效果
    """
    rect = obs['rect']
    color = obs['color']
    dx = obs.get('dx', 0)
    
    x, y, w, h = rect.left, rect.top, rect.width, rect.height
    
    # 設計怪獸身體的點位 (多邊形)
    body_points = [
        (x, y + h),                      # 左下底點
        (x + w * 0.16, y + h - h * 0.2), # 腳凹槽 1
        (x + w * 0.33, y + h),           # 第一隻腳尖
        (x + w * 0.5, y + h - h * 0.2),  # 腳凹槽 2
        (x + w * 0.66, y + h),           # 第二隻腳尖
        (x + w * 0.84, y + h - h * 0.2), # 腳凹槽 3
        (x + w, y + h),                  # 右下角 (第三隻腳)
        (x + w, y + h * 0.4),            # 右側緣
    ]
    
    # 使用迴圈逼近頂部的圓弧 (半圓)
    steps = 15
    for i in range(steps, -1, -1):
        angle = math.pi * (i / steps)
        arc_x = x + w / 2 + math.cos(angle) * (w / 2)
        arc_y = y + h * 0.4 - math.sin(angle) * (h * 0.4)
        body_points.append((arc_x, arc_y))
        
    body_points.append((x, y + h * 0.4)) # 接回左側緣
    
    # --- 漸變填充效果 (從上到下由亮到暗) ---
    # 先繪製較暗的底色本體
    dark_color = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
    pygame.draw.polygon(surface, dark_color, body_points)
    
    # 在上半部疊加一層亮色（模擬光從頂部照射的漸變感）
    highlight_color = (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40))
    # 建立帶 alpha 的上半部漸層
    grad_surf = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
    grad_h = int(h * 0.6)  # 漸層覆蓋上方 60%
    for row in range(grad_h):
        # alpha 從上到下遞減
        alpha = int(160 * (1 - row / grad_h))
        pygame.draw.line(grad_surf, (*highlight_color, alpha), (0, row), (w + 3, row))
    surface.blit(grad_surf, (x - 2, y - 2))
    
    # 畫白色邊框高光增加立體感
    pygame.draw.polygon(surface, (255, 255, 255, 180), body_points, width=2)
    
    # --- 繪製發光雙眼 ---
    eye_rad = max(4, int(h * 0.18))
    eye_y = int(y + h * 0.4)
    left_eye_x = int(x + w * 0.3)
    right_eye_x = int(x + w * 0.7)
    
    # 眼睛外發光效果（霓虹光暈）
    eye_glow_alpha = int(80 + 40 * math.sin(frame_count * 0.1))
    glow_r = eye_rad + 4
    for ex in [left_eye_x, right_eye_x]:
        glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 255, 200, eye_glow_alpha),
                           (glow_r, glow_r), glow_r)
        surface.blit(glow_surf, (ex - glow_r, eye_y - glow_r))
    
    # 眼白
    pygame.draw.circle(surface, (255, 255, 255), (left_eye_x, eye_y), eye_rad)
    pygame.draw.circle(surface, (255, 255, 255), (right_eye_x, eye_y), eye_rad)
    
    # 黑瞳孔 (依據移動速度 dx 看向左或右)
    pupil_rad = max(2, int(eye_rad * 0.5))
    offset = 2 if dx > 0 else (-2 if dx < 0 else 0)
    
    pygame.draw.circle(surface, (0, 0, 0), (left_eye_x + offset, eye_y + 1), pupil_rad)
    pygame.draw.circle(surface, (0, 0, 0), (right_eye_x + offset, eye_y + 1), pupil_rad)
    
    # 瞳孔內的高光（小白點，增加生動感）
    hl_r = max(1, pupil_rad // 2)
    pygame.draw.circle(surface, (255, 255, 255),
                       (left_eye_x + offset - 1, eye_y - 1), hl_r)
    pygame.draw.circle(surface, (255, 255, 255),
                       (right_eye_x + offset - 1, eye_y - 1), hl_r)



def draw_hand_skeleton(surface, landmarks, width, height):
    """
    繪製手部骨架，使用 Pygame 繪圖函式。

    參數：
        surface: Pygame 繪圖表面
        landmarks: MediaPipe Task API 回傳的手部特徵點列表
        width: 影像寬度（用於將正規化座標轉為像素座標）
        height: 影像高度
    """
    # 將所有正規化座標 (0~1) 轉換為像素座標
    points = []
    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        points.append((px, py))

    # 繪製連線（先畫線再畫點，讓點覆蓋在線上方）
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            pygame.draw.line(surface, COLOR_LANDMARK_LINE,
                             points[start_idx], points[end_idx], 2)

    # 繪製每個特徵點
    for i, pt in enumerate(points):
        # 指尖（4, 8, 12, 16, 20）畫大一點
        radius = 5 if i in (4, 8, 12, 16, 20) else 3
        pygame.draw.circle(surface, COLOR_LANDMARK_POINT, pt, radius)
        # 外圈白色輪廓
        pygame.draw.circle(surface, (255, 255, 255), pt, radius, 1)


def opencv_frame_to_pygame_surface(frame):
    """
    將 OpenCV 的影像 (NumPy BGR 陣列) 轉換為 Pygame Surface。

    步驟：
    1. BGR → RGB 色彩空間轉換
    2. 旋轉與翻轉：OpenCV 與 Pygame 的座標系不同，需要轉置
    3. 建立 Pygame Surface

    參數：
        frame: OpenCV 影像（NumPy 陣列，BGR 格式）

    回傳：
        pygame.Surface: 可繪製到 Pygame 視窗的 Surface
    """
    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # NumPy 陣列在 Pygame 中需要轉置軸（行列交換），因為
    # OpenCV 以 (高, 寬, 通道) 存放，Pygame 需要 (寬, 高, 通道) 順序
    transposed = np.transpose(rgb_frame, (1, 0, 2))
    # 建立 Pygame Surface
    surface = pygame.surfarray.make_surface(transposed)
    return surface


# ============================================================
# 遊戲主類別
# ============================================================
class PingPongGame:
    """
    乒乓球遊戲類別，負責管理所有遊戲邏輯：
    - 球的移動與反彈
    - 球拍位置更新
    - 碰撞偵測
    - 分數計算
    - 畫面渲染
    """

    def __init__(self, hit_sound, miss_sound, break_sound, levelup_sound, fonts):
        """
        初始化遊戲狀態。

        參數：
            hit_sound: 板子擊球音效
            miss_sound: 漏球音效
            break_sound: 擊碎怪獸音效
            levelup_sound: 關卡過關音效
            fonts: 字型字典
        """
        self.state = STATE_START       # 遊戲初始狀態：開始畫面
        self.score = 0                  # 目前分數
        self.high_score = 0             # 最高分數
        # 手的追蹤資訊
        self.hand_x = -1
        self.hand_y = -1
        self.hand_dx = 0.0
        self.hand_dy = 0.0
        self.is_pinching = False  # 記錄手指是否處於捏合狀態
        
        # 抓取狀態機制
        self.is_held = False
        self.cooldown_timer = 0  # 釋放後或抓取後的冷卻計時

        self.ball_speed = BALL_SPEED_INIT  # 目前球速（保留作為介面顯示或發射保底）
        self.hit_count = 0              # 連續擊球次數
        self.ball_trail = []            # 球的軌跡（用於拖尾效果）
        self.obstacles = []             # 目前存活的障礙物陣列
        self.level = 1                  # 關卡層級

        # === 技能與道具系統 ===
        self.powerups = []                          # 場上掉落中的道具列表
        self.current_paddle_width = PADDLE_WIDTH    # 目前板子寬度（受 buff 影響）
        self.current_ball_radius = BALL_RADIUS      # 目前球半徑（受 buff 影響）
        self.buff_paddle_timer = 0                  # 加長板子的剩餘幀數
        self.buff_ball_timer = 0                    # 巨球的剩餘幀數
        self.buff_score_x2_timer = 0                # 雙倍得分的剩餘幀數
        self.buff_shield_timer = 0                  # 底部護盾的剩餘幀數
        self.extra_balls = []                       # 分裂球列表 [{x, y, dx, dy}]

        # 音效物件
        self.hit_sound = hit_sound
        self.miss_sound = miss_sound
        self.break_sound = break_sound
        self.levelup_sound = levelup_sound

        # 字型物件
        self.fonts = fonts

        # === 視覺特效系統 ===
        self.particles = []            # 粒子爆破效果列表
        self.star_field = StarField()  # 動態星空背景
        self.frame_count = 0           # 全域動畫幀計數（用於脈動、閃爍等效果）
        
        # 螢幕震動效果
        self.shake_timer = 0           # 震動剩餘幀數
        self.shake_offset_x = 0        # X 方向偏移量
        self.shake_offset_y = 0        # Y 方向偏移量
        
        # 擊球閃光效果
        self.hit_flash_timer = 0       # 板子擊球閃光剩餘幀數
        
        # 分數彈跳動畫
        self.score_bounce = 0          # 分數文字彈跳量（像素）

        # === 拍照系統 ===
        self.captured_photos = []       # 儲存拍下的照片 (Pygame Surfaces)
        self.photo_cooldown = 0.0       # 拍照冷卻時間 (秒)
        self.take_photo_next_frame = False # 標記是否在下一幀拍照

        # 初始化球的位置與速度
        self.reset_ball()

    def reset_ball(self):
        """
        重置球的位置到畫面，並且保持靜止等待自定落下。
        每次漏球或遊戲重新開始時呼叫。
        """
        # 球初始靜置於畫面中央偏下區域
        self.ball_x = WINDOW_WIDTH // 2
        self.ball_y = WINDOW_HEIGHT // 2 + 80

        # 將初始發球的速度設為0，等待自動落下
        self.ball_dx = 0
        self.ball_dy = 0

        # 遊戲開始後的自動落下計時器（預設為 60 幀，以 FPS_TARGET 為基準，約 1 秒）
        self.auto_drop_timer = FPS_TARGET

        # 清空拖尾軌跡
        self.ball_trail = []

    def start_game(self):
        """開始新遊戲：重置分數、關卡、產出障礙物、各種技能狀態並重置球。"""
        self.state = STATE_PLAYING
        self.score = 0
        self.hit_count = 0
        self.ball_speed = BALL_SPEED_INIT
        self.level = 1
        # 重置所有技能道具狀態
        self.powerups = []
        self.current_paddle_width = PADDLE_WIDTH
        self.current_ball_radius = BALL_RADIUS
        self.buff_paddle_timer = 0
        self.buff_ball_timer = 0
        self.buff_score_x2_timer = 0
        self.buff_shield_timer = 0
        self.extra_balls = []
        
        # 重置拍照系統
        self.captured_photos = []
        self.photo_cooldown = 2.0  # 開始後 2 秒內不拍照
        self.take_photo_next_frame = False
        
        self.spawn_obstacles()
        self.reset_ball()

    def spawn_obstacles(self):
        """
        依據目前關卡等級，動態產出怪獸障礙物陣列。
        
        等級越高，怪獸的「列數」與「行數」都會增加：
          - 列數(rows)：基礎 3 列，每升一級 +1，最多 10 列
          - 行數(cols)：基礎 10 行，每升一級 +1，最多 16 行
        怪獸高度會依據列數自動計算，確保不會超出可用區域。
        """
        self.obstacles = []

        # --- 根據等級計算實際列數與行數 ---
        # 基礎列數 3，每升一級 +1，上限 10 列
        rows = min(3 + (self.level - 1), 10)
        # 基礎行數 10，每升一級 +1，上限 16 行
        cols = min(10 + (self.level - 1), 16)

        # --- 動態計算每個怪獸的寬度與高度 ---
        total_gap_width = OBSTACLE_GAP * (cols + 1)
        obs_width = (WINDOW_WIDTH - total_gap_width) // cols

        # 可用的垂直區域：從頂部留白到畫面 40% 處（避免怪獸太低擋住球）
        available_height = int(WINDOW_HEIGHT * 0.40) - OBSTACLE_MARGIN_TOP
        obs_height = max(20, (available_height - OBSTACLE_GAP * (rows + 1)) // rows)

        for row in range(rows):
            color = OBSTACLE_COLORS[row % len(OBSTACLE_COLORS)]
            # 依據列數與關卡決定移動速度方向（奇偶列交替左右移動）
            direction = 1 if row % 2 == 0 else -1
            speed = OBSTACLE_SPEED_BASE + (self.level - 1) * 0.5
            dx = direction * speed

            for col in range(cols):
                x = OBSTACLE_GAP + col * (obs_width + OBSTACLE_GAP)
                y = OBSTACLE_MARGIN_TOP + row * (obs_height + OBSTACLE_GAP)
                rect = pygame.Rect(x, y, obs_width, obs_height)
                self.obstacles.append({'rect': rect, 'color': color, 'dx': dx})

    def update_hand(self, x, y, is_pinching=False):
        """
        更新手部實時狀態與速度，以及捏合狀態。
        由主迴圈每幀呼叫，帶入食指與拇指是否捏合的資訊。
        """
        if self.state != STATE_PLAYING:
            return

        self.is_pinching = is_pinching

        if self.hand_x == -1 and self.hand_y == -1:
            self.hand_x, self.hand_y = x, y
            return

        # 計算瞬間速度（加入平滑處理避免雜訊，0.5 為平滑系數）
        new_dx = x - self.hand_x
        new_dy = y - self.hand_y
        self.hand_dx = self.hand_dx * 0.5 + new_dx * 0.5
        self.hand_dy = self.hand_dy * 0.5 + new_dy * 0.5

        self.hand_x = x
        self.hand_y = y

    def update(self, dt=None):
        """
        更新遊戲狀態（每幀呼叫一次）。
        包含：球的移動、牆壁反彈、抓取與發射邏輯、漏球偵測。

        參數：
            dt (float): 距離上一幀的時間差（秒）。
                        用於計算 time_scale，讓物理移動與實際幀率脫鉤。
                        若未提供則假設為標準幀率 (1/FPS_TARGET)。
        """
        # --- 計算時間縮放因子 ---
        # time_scale = 1.0 代表在目標 FPS 下的正常速度
        # 若實際 FPS 較低（dt 較大），time_scale > 1.0 → 每幀移動更多像素以補償
        # 若實際 FPS 較高（dt 較小），time_scale < 1.0 → 每幀移動較少像素
        if dt is None:
            dt = 1.0 / FPS_TARGET
        # 限制 dt 範圍，避免極端值（如視窗被拖動時的超長幀）
        dt = min(dt, 0.1)  # 最大允許 100ms（相當於 10 FPS）
        time_scale = dt * FPS_TARGET  # 在 30 FPS 時 = 1.0
        # --- 更新全域動畫計數器（無論遊戲狀態都需要，供開始/結束畫面動畫使用）---
        self.frame_count += 1
        self.star_field.update(self.frame_count)

        # --- 更新粒子系統（即使非 PLAYING 狀態也要更新，確保爆破效果能播完）---
        self.particles = [p for p in self.particles if p.update()]

        # --- 更新螢幕震動 ---
        if self.shake_timer > 0:
            self.shake_timer -= 1
            intensity = SHAKE_INTENSITY * (self.shake_timer / SHAKE_DURATION)
            self.shake_offset_x = random.uniform(-intensity, intensity)
            self.shake_offset_y = random.uniform(-intensity, intensity)
        else:
            self.shake_offset_x = 0
            self.shake_offset_y = 0

        # --- 更新擊球閃光倒數 ---
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1

        # --- 更新分數彈跳動畫 ---
        if self.score_bounce > 0:
            self.score_bounce *= 0.85  # 衰減彈跳
            if self.score_bounce < 0.5:
                self.score_bounce = 0

        # --- 更新拍照冷卻與隨機觸發 ---
        if self.photo_cooldown > 0:
            self.photo_cooldown -= dt
        elif len(self.captured_photos) < 3:
            # 每幀 1% 機率 (在 30FPS 下約每 3.3 秒觸發一次)
            if random.random() < 0.01:
                self.take_photo_next_frame = True
                self.photo_cooldown = 3.0  # 拍完至少等 3 秒

        if self.state != STATE_PLAYING:
            return  # 非遊戲進行中則不更新

        # --- 更新障礙物的移動（乘以 time_scale 實現幀率無關）---
        for obs in self.obstacles:
            obs['rect'].x += obs['dx'] * time_scale
            # 若碰到畫面左右邊界則反向彈回
            if obs['rect'].left <= 0:
                obs['rect'].left = 0
                obs['dx'] = abs(obs['dx'])
            elif obs['rect'].right >= WINDOW_WIDTH:
                obs['rect'].right = WINDOW_WIDTH
                obs['dx'] = -abs(obs['dx'])

        # --- 儲存球的軌跡（用於拖尾效果）---
        self.ball_trail.append((int(self.ball_x), int(self.ball_y)))
        # 保留最近 10 個位置
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)

        # 更新冷卻計時器 (用於其它需要防呆的場合)
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        # --- 判斷球是否要自動落下 ---
        # 如果目前球完全沒有速度，則進行倒數，倒數完畢後賦予初速
        if self.ball_dx == 0 and self.ball_dy == 0:
            if self.auto_drop_timer > 0:
                self.auto_drop_timer -= 1
            elif self.auto_drop_timer == 0:
                self.ball_dy = BALL_SPEED_INIT
                # 給一個隨機的微小 x 方向速度，增加反彈的不可預測性
                self.ball_dx = random.choice([-3, 3])
                # 標記為已觸發，避免重複執行
                self.auto_drop_timer = -1
                
        # --- 更新技能道具的冷卻與特效倒數 ---
        if self.buff_paddle_timer > 0:
            self.buff_paddle_timer -= 1
            if self.buff_paddle_timer == 0:
                self.current_paddle_width = PADDLE_WIDTH
                
        if self.buff_ball_timer > 0:
            self.buff_ball_timer -= 1
            if self.buff_ball_timer == 0:
                self.current_ball_radius = BALL_RADIUS

        if self.buff_score_x2_timer > 0:
            self.buff_score_x2_timer -= 1

        if self.buff_shield_timer > 0:
            self.buff_shield_timer -= 1

        # --- 建立板子的碰撞範圍 ---
        paddle_rect = pygame.Rect(0, 0, self.current_paddle_width, PADDLE_HEIGHT)
        if self.hand_x != -1:
            paddle_rect.center = (self.hand_x, WINDOW_HEIGHT - 60)
            
        current_ball_rect = pygame.Rect(int(self.ball_x - self.current_ball_radius), int(self.ball_y - self.current_ball_radius),
                                        self.current_ball_radius*2, self.current_ball_radius*2)

        # --- 更新掉落技能道具與觸發判定（乘以 time_scale）---
        for p in self.powerups[:]:
            p['rect'].y += p['dy'] * time_scale
            # 落出畫面則移除
            if p['rect'].top > WINDOW_HEIGHT:
                self.powerups.remove(p)
                continue
            
            hit_paddle = (self.hand_x != -1) and p['rect'].colliderect(paddle_rect)
            hit_ball = p['rect'].colliderect(current_ball_rect)
            
            if hit_paddle or hit_ball:
                if p['type'] == 'enlarge_ball':
                    # 巨球：球半徑放大 2 倍，持續 6 秒
                    self.current_ball_radius = int(BALL_RADIUS * 2)
                    self.buff_ball_timer = FPS_TARGET * 6
                elif p['type'] == 'enlarge_paddle':
                    # 長板：板子加寬 1.5 倍，持續 6 秒
                    self.current_paddle_width = int(PADDLE_WIDTH * 1.5)
                    self.buff_paddle_timer = FPS_TARGET * 6
                elif p['type'] == 'slow_down':
                    # 減速：球速降低 30%（瞬間生效）
                    self.ball_dx *= 0.7
                    self.ball_dy *= 0.7
                elif p['type'] == 'multi_ball':
                    # 分裂球：產生 2 顆額外的球，各自獨立彈跳可破壞怪獸
                    for angle_offset in [-0.5, 0.5]:
                        eb = {
                            'x': float(self.ball_x),
                            'y': float(self.ball_y),
                            'dx': self.ball_dx * math.cos(angle_offset) - self.ball_dy * math.sin(angle_offset),
                            'dy': self.ball_dx * math.sin(angle_offset) + self.ball_dy * math.cos(angle_offset),
                            'life': FPS_TARGET * 8,  # 存活 8 秒後自動消失
                        }
                        self.extra_balls.append(eb)
                elif p['type'] == 'score_x2':
                    # 雙倍得分：持續 8 秒內所有得分 ×2
                    self.buff_score_x2_timer = FPS_TARGET * 8
                elif p['type'] == 'shield':
                    # 底部護盾：持續 5 秒，球碰到底部會反彈而非失分
                    self.buff_shield_timer = FPS_TARGET * 5
                elif p['type'] == 'speed_up':
                    # 加速衝刺：球速瞬間提升 40%（雙刃劍，更刺激但也更難控制）
                    self.ball_dx *= 1.4
                    self.ball_dy *= 1.4
                    # 限制最高速
                    spd = math.hypot(self.ball_dx, self.ball_dy)
                    if spd > BALL_MAX_SPEED:
                        ratio = BALL_MAX_SPEED / spd
                        self.ball_dx *= ratio
                        self.ball_dy *= ratio
                self.powerups.remove(p)
                self.levelup_sound.play()  # 播個音效回饋

        # --- 物理移動球（乘以 time_scale，讓球速與幀率脫鉤）---
        self.ball_x += self.ball_dx * time_scale
        self.ball_y += self.ball_dy * time_scale

        # --- 板子反彈碰撞偵測 ---
        if self.hand_x != -1:
            new_ball_rect = pygame.Rect(int(self.ball_x - self.current_ball_radius), int(self.ball_y - self.current_ball_radius), self.current_ball_radius*2, self.current_ball_radius*2)
            # 當球往下墜落 (ball_dy > 0) 且碰到板子
            if new_ball_rect.colliderect(paddle_rect) and self.ball_dy > 0:
                # 將球反彈
                self.ball_dy = -abs(self.ball_dy)  # 確保一定是往上
                # 避免球卡在板子裡面
                self.ball_y = paddle_rect.top - self.current_ball_radius
                
                # 依據擊球位置給予橫向力道，賦予切球的手感 (離中心越遠角度越大)
                hit_offset = (self.ball_x - paddle_rect.centerx) / (self.current_paddle_width / 2)
                self.ball_dx += hit_offset * 4.0
                
                # 計算目前的速度大小
                current_speed = math.hypot(self.ball_dx, self.ball_dy)
                
                # 防止除以零（理論上 ball_dy > 0 已保證，但加保護更安全）
                if current_speed > 0.01:
                    # 每次接到球後將速度向上提升
                    new_speed = current_speed + BALL_SPEED_INCREMENT
                    
                    # 限制最高球速避免過於暴走
                    if new_speed > BALL_MAX_SPEED:
                        new_speed = BALL_MAX_SPEED
                        
                    # 套用新速度
                    ratio = new_speed / current_speed
                    self.ball_dx *= ratio
                    self.ball_dy *= ratio
                    
                    # 更新顯示用的球速變數
                    self.ball_speed = new_speed

                # 觸發擊球閃光效果
                self.hit_flash_timer = 6
                self.hit_sound.play()

            # --- 障礙物碰撞與破壞偵測 ---
            ball_rect = pygame.Rect(int(self.ball_x - self.current_ball_radius), int(self.ball_y - self.current_ball_radius),
                                    self.current_ball_radius * 2, self.current_ball_radius * 2)
            
            for obs in self.obstacles[:]:
                if ball_rect.colliderect(obs['rect']):
                    # 碰到障礙物，將其移除
                    self.obstacles.remove(obs)
                    # 碰撞反彈 (AABB判定撞擊面)
                    dx_overlap = min(ball_rect.right - obs['rect'].left, obs['rect'].right - ball_rect.left)
                    dy_overlap = min(ball_rect.bottom - obs['rect'].top, obs['rect'].bottom - ball_rect.top)
                    
                    if dx_overlap < dy_overlap:
                        self.ball_dx = -self.ball_dx
                        self.ball_x += self.ball_dx
                    else:
                        self.ball_dy = -self.ball_dy
                        self.ball_y += self.ball_dy
                        
                    # 計算得分（雙倍得分 buff 生效時 ×2）
                    base_pts = 5
                    if self.buff_score_x2_timer > 0:
                        base_pts *= 2
                    self.score += base_pts
                    self.score_bounce = 8  # 觸發分數彈跳動畫
                    if self.score > self.high_score:
                        self.high_score = self.score
                    self.break_sound.play()

                    # --- 產生粒子爆破特效 ---
                    num_particles = random.randint(PARTICLE_COUNT_MIN, PARTICLE_COUNT_MAX)
                    cx, cy = obs['rect'].centerx, obs['rect'].centery
                    for _ in range(num_particles):
                        self.particles.append(Particle(cx, cy, obs['color']))

                    # 破壞怪獸時，隨機掉落技能道具 (約 25% 機率)
                    if random.random() < 0.25:
                        # 道具池：原 3 種 + 新增 4 種
                        p_type = random.choice([
                            'enlarge_ball',    # 巨球
                            'enlarge_paddle',  # 長板
                            'slow_down',       # 減速
                            'multi_ball',      # 分裂球
                            'score_x2',        # 雙倍得分
                            'shield',          # 底部護盾
                            'speed_up',        # 加速衝刺
                        ])
                        sz = 30   # 道具尺寸（配合 1280×720 解析度）
                        px = obs['rect'].centerx - sz // 2
                        py = obs['rect'].centery - sz // 2
                        self.powerups.append({'rect': pygame.Rect(px, py, sz, sz), 'type': p_type, 'dy': 5})

                    break # 保證一次更新最多撞碎一顆磚塊，避免卡在深處

            if not self.obstacles:
                # 如果所有障礙物都被清除，推進關卡並重新生成
                self.levelup_sound.play()
                self.level += 1
                self.spawn_obstacles()

        # --- 左右牆壁反彈（移出 hand_x 條件外，確保即使偵測不到手也能正常反彈）---
        if self.ball_x - self.current_ball_radius <= 0:
            self.ball_x = self.current_ball_radius
            self.ball_dx = abs(self.ball_dx)
        elif self.ball_x + self.current_ball_radius >= WINDOW_WIDTH:
            self.ball_x = WINDOW_WIDTH - self.current_ball_radius
            self.ball_dx = -abs(self.ball_dx)

        # --- 頂部反彈 ---
        if self.ball_y - self.current_ball_radius <= 0:
            self.ball_y = self.current_ball_radius
            self.ball_dy = abs(self.ball_dy)

        # --- 漏球偵測（包含護盾判定）---
        if self.ball_y - self.current_ball_radius > WINDOW_HEIGHT:
            if self.buff_shield_timer > 0:
                # 護盾生效中：球從底部反彈回來
                self.ball_y = WINDOW_HEIGHT - self.current_ball_radius
                self.ball_dy = -abs(self.ball_dy)
                # 消耗一部分護盾時間作為代價
                self.buff_shield_timer = max(0, self.buff_shield_timer - FPS_TARGET)
                self.hit_sound.play()
            else:
                # 無護盾，遊戲結束
                self.miss_sound.play()
                self.shake_timer = SHAKE_DURATION
                self.state = STATE_GAME_OVER

        # --- 更新分裂球（extra_balls）的物理與障礙物碰撞 ---
        for eb in self.extra_balls[:]:
            eb['x'] += eb['dx'] * time_scale
            eb['y'] += eb['dy'] * time_scale
            eb['life'] -= 1
            # 左右牆壁反彈
            if eb['x'] - BALL_RADIUS <= 0:
                eb['x'] = BALL_RADIUS
                eb['dx'] = abs(eb['dx'])
            elif eb['x'] + BALL_RADIUS >= WINDOW_WIDTH:
                eb['x'] = WINDOW_WIDTH - BALL_RADIUS
                eb['dx'] = -abs(eb['dx'])
            # 頂部反彈
            if eb['y'] - BALL_RADIUS <= 0:
                eb['y'] = BALL_RADIUS
                eb['dy'] = abs(eb['dy'])
            # 底部消失或壽命耗盡
            if eb['y'] > WINDOW_HEIGHT + 50 or eb['life'] <= 0:
                self.extra_balls.remove(eb)
                continue
            # 分裂球碰撞障礙物
            eb_rect = pygame.Rect(int(eb['x'] - BALL_RADIUS), int(eb['y'] - BALL_RADIUS),
                                  BALL_RADIUS * 2, BALL_RADIUS * 2)
            for obs in self.obstacles[:]:
                if eb_rect.colliderect(obs['rect']):
                    self.obstacles.remove(obs)
                    eb['dy'] = -eb['dy']
                    base_pts = 5
                    if self.buff_score_x2_timer > 0:
                        base_pts *= 2
                    self.score += base_pts
                    self.score_bounce = 8
                    if self.score > self.high_score:
                        self.high_score = self.score
                    self.break_sound.play()
                    num_p = random.randint(PARTICLE_COUNT_MIN, PARTICLE_COUNT_MAX)
                    cx, cy = obs['rect'].centerx, obs['rect'].centery
                    for _ in range(num_p):
                        self.particles.append(Particle(cx, cy, obs['color']))
                    break
        # 分裂球也可能清除最後的障礙物，觸發過關
        if not self.obstacles and self.state == STATE_PLAYING:
            self.levelup_sound.play()
            self.level += 1
            self.spawn_obstacles()

    def draw(self, surface):
        """
        在 Pygame Surface 上繪製所有遊戲進行中的元素。

        參數：
            surface: Pygame 繪圖表面
        """
        # --- 繪製星空背景 ---
        self.star_field.draw(surface)

        # --- 繪製頂部反彈區域指示線（漸變霓虹色）---
        # 計算隨時間變化的彩虹色相
        hue_shift = (self.frame_count * 2) % 360
        top_line_color = pygame.Color(0)
        top_line_color.hsla = (hue_shift, 80, 60, 100)
        pygame.draw.line(surface, top_line_color, (0, 3), (WINDOW_WIDTH, 3), 3)
        # 上方漸層光暈
        glow_surf = pygame.Surface((WINDOW_WIDTH, 8), pygame.SRCALPHA)
        glow_surf.fill((*top_line_color[:3], 40))
        surface.blit(glow_surf, (0, 0))

        # --- 繪製球的拖尾效果（彩虹漸層版）---
        if self.ball_trail:
            trail_len = len(self.ball_trail)
            for i, pos in enumerate(self.ball_trail):
                # 進度 0~1（越新的越大）
                progress = (i + 1) / trail_len
                trail_radius = max(2, int(self.current_ball_radius * progress * 0.6))
                trail_alpha = int(progress * 120)
                # 根據球速計算色相：慢→藍(200°)，快→紅(0°)
                speed_ratio = min(1.0, self.ball_speed / BALL_MAX_SPEED)
                hue = int(200 - speed_ratio * 200 + i * 8) % 360
                trail_color = pygame.Color(0)
                trail_color.hsla = (hue, 90, 60, 100)
                # 繪製拖尾圓點
                ts = pygame.Surface((trail_radius * 2, trail_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(ts, (*trail_color[:3], trail_alpha),
                                   (trail_radius, trail_radius), trail_radius)
                surface.blit(ts, (pos[0] - trail_radius, pos[1] - trail_radius))

        # --- 繪製小怪獸障礙物（帶動畫幀）---
        for obs in self.obstacles:
            draw_monster(surface, obs, self.frame_count)

        # --- 繪製技能道具（帶旋轉光環）---
        # 道具類型對應的顏色與圖示字母
        powerup_styles = {
            'enlarge_ball':   ((255, 120, 0),   "B"),  # 橘色 - 巨球
            'enlarge_paddle': ((0, 180, 255),   "P"),  # 藍色 - 長板
            'slow_down':      ((100, 255, 100), "S"),  # 綠色 - 減速
            'multi_ball':     ((255, 255, 0),   "M"),  # 黃色 - 分裂球
            'score_x2':       ((255, 50, 255),  "2"),  # 深紫 - 雙倍得分
            'shield':         ((0, 255, 200),   "H"),  # 青綠 - 護盾
            'speed_up':       ((255, 60, 60),   "!"),  # 紅色 - 加速
        }
        for p in self.powerups:
            pr = p['rect']
            style = powerup_styles.get(p['type'], ((200, 200, 200), "?"))
            c, text = style
            # 外層旋轉光環
            ring_alpha = int(120 + 60 * math.sin(self.frame_count * 0.15))
            ring_r = pr.width // 2 + 4
            ring_surf = pygame.Surface((ring_r * 2, ring_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (*c, ring_alpha), (ring_r, ring_r), ring_r, 2)
            surface.blit(ring_surf, (pr.centerx - ring_r, pr.centery - ring_r))
            # 核心圓
            pygame.draw.circle(surface, c, pr.center, pr.width // 2)
            pygame.draw.circle(surface, (255, 255, 255), pr.center, pr.width // 2, 2)
            draw_text_with_shadow(surface, text, (pr.left + 5, pr.top + 2),
                                  self.fonts['small'], (255, 255, 255), 1)

        # --- 繪製分裂球（紅色半透明球）---
        for eb in self.extra_balls:
            eb_center = (int(eb['x']), int(eb['y']))
            # 分裂球外光暈
            glow_r = BALL_RADIUS + 6
            gs = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (255, 100, 0, 60), (glow_r, glow_r), glow_r)
            surface.blit(gs, (eb_center[0] - glow_r, eb_center[1] - glow_r))
            # 分裂球本體
            pygame.draw.circle(surface, (255, 160, 50), eb_center, BALL_RADIUS)
            pygame.draw.circle(surface, (255, 255, 200), eb_center, BALL_RADIUS // 2)

        # --- 繪製底部護盾光條 ---
        if self.buff_shield_timer > 0:
            shield_alpha = int(100 + 80 * math.sin(self.frame_count * 0.12))
            shield_surf = pygame.Surface((WINDOW_WIDTH, 6), pygame.SRCALPHA)
            shield_surf.fill((0, 255, 200, shield_alpha))
            surface.blit(shield_surf, (0, WINDOW_HEIGHT - 6))
            # 護盾光暈
            shield_glow = pygame.Surface((WINDOW_WIDTH, 16), pygame.SRCALPHA)
            shield_glow.fill((0, 255, 200, shield_alpha // 3))
            surface.blit(shield_glow, (0, WINDOW_HEIGHT - 16))

        # --- 繪製球（帶脈動呼吸效果）---
        pulse = math.sin(self.frame_count * PULSE_SPEED) * 2  # ±2 像素微動
        display_radius = int(self.current_ball_radius + pulse)
        ball_center = (int(self.ball_x), int(self.ball_y))
        draw_glowing_circle(surface, ball_center, display_radius,
                            COLOR_BALL, COLOR_BALL_GLOW, glow_layers=4)

        # --- 繪製接球實體板 (Paddle)（漸變色 + 擊球閃光）---
        if self.hand_x != -1:
            paddle_x = int(self.hand_x - self.current_paddle_width / 2)
            paddle_y = int((WINDOW_HEIGHT - 60) - PADDLE_HEIGHT / 2)
            paddle_rect = pygame.Rect(paddle_x, paddle_y, self.current_paddle_width, PADDLE_HEIGHT)
            
            # 建立半透明的發光光暈（加大範圍）
            glow_radius = 18
            pad_glow = pygame.Surface((self.current_paddle_width + glow_radius * 2,
                                       PADDLE_HEIGHT + glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.rect(pad_glow, (*PADDLE_COLOR_BASE, 60),
                             (0, 0, self.current_paddle_width + glow_radius * 2,
                              PADDLE_HEIGHT + glow_radius * 2),
                             border_radius=18)
            surface.blit(pad_glow, (paddle_x - glow_radius, paddle_y - glow_radius))
            
            # 漸變色板子本體（中心亮、邊緣暗）
            paddle_surf = pygame.Surface((self.current_paddle_width, PADDLE_HEIGHT), pygame.SRCALPHA)
            for col in range(self.current_paddle_width):
                # 離中心越遠越暗
                dist = abs(col - self.current_paddle_width / 2) / (self.current_paddle_width / 2)
                factor = 1.0 - dist * 0.4
                c = (int(PADDLE_COLOR_BASE[0] * factor),
                     int(PADDLE_COLOR_BASE[1] * factor),
                     int(PADDLE_COLOR_BASE[2] * factor))
                pygame.draw.line(paddle_surf, c, (col, 0), (col, PADDLE_HEIGHT - 1))
            surface.blit(paddle_surf, (paddle_x, paddle_y))
            # 高光內框
            pygame.draw.rect(surface, (255, 255, 255, 120), paddle_rect.inflate(-4, -4), 2, border_radius=6)
            
            # 擊球閃光效果
            if self.hit_flash_timer > 0:
                flash_alpha = int(200 * (self.hit_flash_timer / 6))
                flash_surf = pygame.Surface((self.current_paddle_width + 20, PADDLE_HEIGHT + 20), pygame.SRCALPHA)
                flash_surf.fill((255, 255, 255, flash_alpha))
                surface.blit(flash_surf, (paddle_x - 10, paddle_y - 10))

            # 板子下方陰影
            shadow_surf = pygame.Surface((self.current_paddle_width, 6), pygame.SRCALPHA)
            shadow_surf.fill((0, 0, 0, 50))
            surface.blit(shadow_surf, (paddle_x, paddle_y + PADDLE_HEIGHT + 2))

        # --- 繪製粒子爆破效果 ---
        for particle in self.particles:
            particle.draw(surface)

        # --- 繪製分數（帶彈跳動畫）---
        score_y = int(20 - self.score_bounce)
        draw_text_with_shadow(surface, f"Score: {self.score}", (20, score_y),
                               self.fonts['medium'], COLOR_SCORE)
        draw_text_with_shadow(surface, f"Best: {self.high_score}", (20, 55),
                               self.fonts['small'], (180, 180, 180))

        # --- 繪製關卡指示器 ---
        level_text = f"Level {self.level}"
        draw_text_with_shadow(surface, level_text, (WINDOW_WIDTH - 120, 20),
                               self.fonts['small'], (180, 100, 255))

        # --- 繪製球速進度條 ---
        bar_x = WINDOW_WIDTH - 180
        bar_y = 48
        bar_w = 160
        bar_h = 10
        speed_ratio = min(1.0, self.ball_speed / BALL_MAX_SPEED)
        # 進度條底框
        pygame.draw.rect(surface, (50, 50, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        # 進度條填充（顏色隨速度從綠→黃→紅漸變）
        if speed_ratio < 0.5:
            fill_color = (int(speed_ratio * 2 * 255), 255, 50)
        else:
            fill_color = (255, int((1 - speed_ratio) * 2 * 255), 50)
        fill_w = int(bar_w * speed_ratio)
        if fill_w > 0:
            pygame.draw.rect(surface, fill_color, (bar_x, bar_y, fill_w, bar_h), border_radius=5)
        # 進度條外框
        pygame.draw.rect(surface, (150, 150, 180), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=5)
        # 標籤
        draw_text_with_shadow(surface, f"Speed", (bar_x, bar_y - 18),
                               self.fonts['fps'], (150, 200, 255))

        # --- 繪製技能效果提示（左下角，顯示當前生效中的所有 buff）---
        buff_y = WINDOW_HEIGHT - 40
        blink = int(180 + 75 * math.sin(self.frame_count * 0.2))
        if self.buff_ball_timer > 0:
            draw_text_with_shadow(surface, "🔴 巨球!", (20, buff_y),
                                   self.fonts['small'], (255, blink, 50))
            buff_y -= 25
        if self.buff_paddle_timer > 0:
            draw_text_with_shadow(surface, "🔵 長板!", (20, buff_y),
                                   self.fonts['small'], (50, blink, 255))
            buff_y -= 25
        if self.buff_score_x2_timer > 0:
            draw_text_with_shadow(surface, "💰 雙倍得分!", (20, buff_y),
                                   self.fonts['small'], (255, blink, 255))
            buff_y -= 25
        if self.buff_shield_timer > 0:
            draw_text_with_shadow(surface, "🛡 護盾!", (20, buff_y),
                                   self.fonts['small'], (0, 255, blink))
            buff_y -= 25
        if self.extra_balls:
            draw_text_with_shadow(surface, f"☄ 分裂球 x{len(self.extra_balls)}", (20, buff_y),
                                   self.fonts['small'], (255, 255, blink))

    def draw_start_screen(self, surface):
        """
        繪製遊戲開始畫面（帶脈動標題、旋轉光環、閃爍提示）。

        參數：
            surface: Pygame 繪圖表面
        """
        # 半透明暗色覆蓋層
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 10, 20, 190))
        surface.blit(overlay, (0, 0))

        # 繪製星空背景
        self.star_field.draw(surface)

        center_x = WINDOW_WIDTH // 2

        # --- 旋轉光環裝飾 ---
        ring_angle = self.frame_count * 0.02
        ring_radius = 100
        ring_cx, ring_cy = center_x, 140
        ring_surf = pygame.Surface((ring_radius * 2 + 20, ring_radius * 2 + 20), pygame.SRCALPHA)
        for i in range(3):
            r = ring_radius - i * 20
            alpha = int(40 + 20 * math.sin(ring_angle + i * 1.5))
            hue = (self.frame_count * 2 + i * 60) % 360
            c = pygame.Color(0)
            c.hsla = (hue, 80, 60, 100)
            pygame.draw.circle(ring_surf, (*c[:3], alpha),
                               (ring_radius + 10, ring_radius + 10), r, 2)
        surface.blit(ring_surf, (ring_cx - ring_radius - 10, ring_cy - ring_radius - 10))

        # --- 脈動標題 ---
        pulse_val = math.sin(self.frame_count * PULSE_SPEED) * 0.15 + 1.0
        title_brightness = int(min(255, 255 * pulse_val))
        title_color = (0, title_brightness, title_brightness)
        draw_centered_text(surface, "PING PONG", center_x, 100,
                           self.fonts['title'], title_color)

        # 副標題
        draw_centered_text(surface, "Hand Tracking Game", center_x, 170,
                           self.fonts['medium'], (200, 200, 200))

        # 操作說明
        instructions = [
            ("用食指指尖控制球拍", (180, 180, 180)),
            ("反彈球來擊碎怪獸得分！", (180, 180, 180)),
        ]
        y_start = 215
        for i, (line, color) in enumerate(instructions):
            draw_centered_text(surface, line, center_x, y_start + i * 28,
                               self.fonts['small'], color)

        # --- 技能道具圖鑑（兩欄排版）---
        # 小標題
        draw_centered_text(surface, "- 道具一覽 -", center_x, 280,
                           self.fonts['small'], (255, 220, 100))

        # 道具資料：(圖示字母, 顏色, 名稱, 說明)
        powerup_info = [
            ("B", (255, 120, 0),   "巨球",     "球變大，更容易擊中"),
            ("P", (0, 180, 255),   "長板",     "板子加寬 1.5 倍"),
            ("S", (100, 255, 100), "減速",     "球速降低 30%"),
            ("M", (255, 255, 0),   "分裂球",   "產生 2 顆額外的球"),
            ("2", (255, 50, 255),  "雙倍得分", "所有得分 ×2"),
            ("H", (0, 255, 200),   "護盾",     "底部反彈，不怕漏球"),
            ("!", (255, 60, 60),   "加速衝刺", "球速提升 40%"),
        ]

        # 兩欄排版參數
        col_width = 280                    # 每欄寬度
        col_gap = 40                       # 欄間距
        total_w = col_width * 2 + col_gap  # 兩欄總寬度
        left_x = center_x - total_w // 2   # 左欄起始 X
        right_x = left_x + col_width + col_gap  # 右欄起始 X
        item_h = 36                        # 每個道具項目的高度
        start_y = 310                      # 道具列表起始 Y

        for idx, (letter, color, name, desc) in enumerate(powerup_info):
            # 前 4 個放左欄，後 3 個放右欄
            if idx < 4:
                ix = left_x
                iy = start_y + idx * item_h
            else:
                ix = right_x
                iy = start_y + (idx - 4) * item_h

            # 繪製道具圓形圖示
            icon_cx = ix + 14
            icon_cy = iy + 12
            icon_r = 12
            # 圖示外光暈
            glow_alpha = int(60 + 30 * math.sin(self.frame_count * 0.1 + idx))
            gs = pygame.Surface((icon_r * 3, icon_r * 3), pygame.SRCALPHA)
            pygame.draw.circle(gs, (*color, glow_alpha),
                               (icon_r * 3 // 2, icon_r * 3 // 2), icon_r + 4)
            surface.blit(gs, (icon_cx - icon_r * 3 // 2, icon_cy - icon_r * 3 // 2))
            # 圖示本體
            pygame.draw.circle(surface, color, (icon_cx, icon_cy), icon_r)
            pygame.draw.circle(surface, (255, 255, 255), (icon_cx, icon_cy), icon_r, 2)
            # 圖示字母
            letter_surf = self.fonts['fps'].render(letter, True, (255, 255, 255))
            lr = letter_surf.get_rect(center=(icon_cx, icon_cy))
            surface.blit(letter_surf, lr)

            # 繪製道具名稱與說明文字
            draw_text_with_shadow(surface, name, (ix + 32, iy),
                                  self.fonts['small'], color, 1)
            draw_text_with_shadow(surface, desc, (ix + 32, iy + 16),
                                  self.fonts['fps'], (160, 160, 170), 1)

        # --- 底部操作提示（閃爍效果）---
        bottom_y = 470
        blink_alpha = int(180 + 75 * math.sin(self.frame_count * BLINK_SPEED))
        blink_color = (0, blink_alpha, int(blink_alpha * 0.66))
        draw_centered_text(surface, "比出V勢或按 [空白鍵] 開始", center_x, bottom_y,
                           self.fonts['small'], blink_color)
        draw_centered_text(surface, "按 [Q] 退出", center_x, bottom_y + 30,
                           self.fonts['small'], (150, 150, 150))

        # 繪製殘留粒子效果
        for particle in self.particles:
            particle.draw(surface)

    def draw_game_over_screen(self, surface):
        """
        繪製遊戲結束畫面（帶震動效果、脈動標題）。

        參數：
            surface: Pygame 繪圖表面
        """
        # 半透明暗色覆蓋層（偏紅色調）
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((30, 10, 10, 200))
        surface.blit(overlay, (0, 0))

        # 繪製星空背景
        self.star_field.draw(surface)

        # 繪製殘留粒子效果
        for particle in self.particles:
            particle.draw(surface)

        center_x = WINDOW_WIDTH // 2

        # GAME OVER 標題（帶脈動效果）
        pulse_val = math.sin(self.frame_count * PULSE_SPEED * 1.5) * 0.2 + 1.0
        title_brightness = int(min(255, 255 * pulse_val))
        draw_centered_text(surface, "GAME OVER", center_x, 140,
                           self.fonts['title'], (title_brightness, int(title_brightness * 0.4), 0))

        # 分數顯示
        draw_centered_text(surface, f"Score: {self.score}", center_x, 230,
                           self.fonts['large'], COLOR_SCORE)

        # 最高分
        draw_centered_text(surface, f"Best: {self.high_score}", center_x, 280,
                           self.fonts['medium'], (200, 200, 200))

        # 關卡資訊
        draw_centered_text(surface, f"到達 Level {self.level}", center_x, 320,
                           self.fonts['small'], (180, 100, 255))

        # --- 繪製遊戲過程中拍下的照片 ---
        if self.captured_photos:
            photo_width = 300
            photo_height = int(photo_width * (WINDOW_HEIGHT / WINDOW_WIDTH))
            padding = 40
            total_width = len(self.captured_photos) * photo_width + (len(self.captured_photos) - 1) * padding
            start_x = center_x - total_width // 2
            photo_y = 480
            
            for i, photo in enumerate(self.captured_photos):
                # 縮小照片
                small_photo = pygame.transform.scale(photo, (photo_width, photo_height))
                
                # 建立相框 (拍立得風格：白色背景，下方留白)
                frame_surf = pygame.Surface((photo_width + 20, photo_height + 40))
                frame_surf.fill((255, 255, 255))
                frame_surf.blit(small_photo, (10, 10))
                
                # 加點隨機旋轉角度讓照片看起來比較自然
                angle = [5, -3, 4][i] if i < 3 else 0
                rotated_frame = pygame.transform.rotate(frame_surf, angle)
                
                rx = start_x + i * (photo_width + padding) - (rotated_frame.get_width() - photo_width) // 2
                ry = photo_y - (rotated_frame.get_height() - photo_height) // 2
                surface.blit(rotated_frame, (rx, ry))

        # 重新開始提示（往下移）
        blink_alpha = int(180 + 75 * math.sin(self.frame_count * BLINK_SPEED))
        draw_centered_text(surface, "比出V手勢或按 [空白鍵] 重新開始", center_x, 650,
                           self.fonts['small'], (0, blink_alpha, int(blink_alpha * 0.66)))
        draw_centered_text(surface, "按 [Q] 退出", center_x, 685,
                           self.fonts['small'], (150, 150, 150))


# ============================================================
# 主程式入口
# ============================================================
def main():
    """
    遊戲主函式：
    1. 初始化 Pygame、攝影機與 MediaPipe
    2. 進入主迴圈：處理事件 → 讀取影格 → 手部偵測 → 更新遊戲 → 繪製畫面
    3. 清理資源
    """

    # --- 初始化 Pygame ---
    pygame.init()
    # 初始化混音器（音效系統），設定取樣率與緩衝區大小
    pygame.mixer.init(frequency=SOUND_SAMPLE_RATE, size=-16, channels=2, buffer=512)

    # --- 載入並播放背景音樂 ---
    # 支援 bgm.wav, bgm.mp3, bgm.ogg 等格式，若無法載入則略過
    bgm_path = "bgm.wav"  # 可以自行替換為您的 .mp3 或其他音樂檔名稱
    if os.path.exists(bgm_path):
        try:
            pygame.mixer.music.load(bgm_path)
            pygame.mixer.music.set_volume(0.4) # 設定背景音樂音量大小
            pygame.mixer.music.play(-1)        # -1 代表背景音樂無限循環播放
            print(f"成功載入背景音樂：{bgm_path}")
        except Exception as e:
            print(f"背景音樂 {bgm_path} 載入失敗：{e}")
    else:
        print(f"提示：未找到背景音樂 {bgm_path}。若想加入音樂，請將音檔放入相同資料夾並命名為 {bgm_path}，或更新程式碼中的檔名。")

    # 建立遊戲視窗
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Ping Pong - Hand Tracking")

    # 建立時鐘物件，用於控制 FPS
    clock = pygame.time.Clock()

    # --- 載入字型 ---
    # 載入專案目錄下的 msjh.ttc 字型檔
    font_path = "msjh.ttc"
    try:
        title_font = pygame.font.Font(font_path, 52)
        title_font.set_bold(True)
        large_font = pygame.font.Font(font_path, 36)
        large_font.set_bold(True)
        fonts = {
            'title': title_font,
            'large': large_font,
            'medium': pygame.font.Font(font_path, 24),
            'small': pygame.font.Font(font_path, 18),
            'fps': pygame.font.Font(font_path, 16),
        }
    except Exception:
        # 若找不到中文字型，使用預設字型
        fonts = {
            'title': pygame.font.Font(None, 64),
            'large': pygame.font.Font(None, 48),
            'medium': pygame.font.Font(None, 32),
            'small': pygame.font.Font(None, 24),
            'fps': pygame.font.Font(None, 20),
        }

    # --- 產生音效 ---
    hit_sound = generate_beep_sound(SOUND_HIT_FREQ, SOUND_HIT_DURATION)
    miss_sound = generate_beep_sound(SOUND_MISS_FREQ, SOUND_MISS_DURATION)
    break_sound = generate_beep_sound(SOUND_BREAK_FREQ, SOUND_BREAK_DURATION)
    levelup_sound = generate_beep_sound(SOUND_LEVELUP_FREQ, SOUND_LEVELUP_DURATION)

    # --- 初始化攝影機 ---
    # VideoCapture(0) 開啟預設攝影機（通常是內建 Webcam）
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    # 確認攝影機是否成功開啟
    if not cap.isOpened():
        print("錯誤：無法開啟攝影機！請確認攝影機是否已連接。")
        pygame.quit()
        return

    # --- 初始化 MediaPipe Hand Landmarker (Task API) ---
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=HAND_MAX_NUM,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=HAND_TRACKING_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        running_mode=vision.RunningMode.VIDEO
    )
    hands_detector = vision.HandLandmarker.create_from_options(options)

    # --- 初始化遊戲 ---
    game = PingPongGame(hit_sound, miss_sound, break_sound, levelup_sound, fonts)

    print("=" * 50)
    print("  [Ping Pong] 手勢乒乓球遊戲已啟動！（Pygame 版）")
    print("  - 請將手掌對準攝影機")
    print("  - 按 [空白鍵] 開始遊戲")
    print("  - 按 [Q] 或 [ESC] 退出")
    print("=" * 50)

    # MediaPipe 偵測用的時間起點
    start_timestamp = time.time()

    # ========================================================
    # 主遊戲迴圈
    # ========================================================
    running = True
    while running:
        # --- 處理 Pygame 事件 ---
        for event in pygame.event.get():
            # 視窗關閉事件（按 X 按鈕）
            if event.type == pygame.QUIT:
                running = False

            # 鍵盤按下事件
            if event.type == pygame.KEYDOWN:
                # 按 Q 或 ESC 退出遊戲
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                # 按空白鍵開始 / 重新開始遊戲
                if event.key == pygame.K_SPACE:
                    if game.state in (STATE_START, STATE_GAME_OVER):
                        game.start_game()

        # --- 讀取攝影機影格 ---
        success, frame = cap.read()
        if not success:
            print("警告：無法讀取攝影機影格")
            break

        # --- 水平翻轉影像（鏡像效果，讓操作更直覺）---
        frame = cv2.flip(frame, 1)

        # --- 確保影像尺寸正確 ---
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        # --- MediaPipe 手部偵測 ---
        # 轉為 RGB 格式（MediaPipe 需要 RGB 輸入）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 取得相對時間戳記（毫秒），確保單調遞增
        frame_timestamp_ms = int((time.time() - start_timestamp) * 1000)

        # 執行推論
        try:
            results = hands_detector.detect_for_video(mp_image, frame_timestamp_ms)
        except Exception as e:
            results = None
            print(f"手部推論錯誤: {e}")

        # --- 將攝影機畫面轉為 Pygame Surface ---
        cam_surface = opencv_frame_to_pygame_surface(frame)

        # --- 使用臨時 Surface 繪製所有內容（支援螢幕震動）---
        render_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        render_surface.blit(cam_surface, (0, 0))

        # --- 繪製半透明暗色覆蓋層（讓遊戲元素更清晰）---
        if game.state == STATE_PLAYING:
            dark_overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            dark_overlay.fill((20, 20, 30, 150))
            render_surface.blit(dark_overlay, (0, 0))

        # --- 處理手部偵測結果 ---
        if results and results.hand_landmarks:
            for hand_landmarks_data in results.hand_landmarks:
                # 取得食指指尖 (Landmark #8) 與拇指指尖 (Landmark #4)的座標（用於判斷捏合）
                index_tip = hand_landmarks_data[INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks_data[THUMB_TIP]
                
                finger_x = int(index_tip.x * WINDOW_WIDTH)
                finger_y = int(index_tip.y * WINDOW_HEIGHT)
                thumb_x = int(thumb_tip.x * WINDOW_WIDTH)
                thumb_y = int(thumb_tip.y * WINDOW_HEIGHT)
                
                # 判斷是否捏合
                pinch_dist = math.hypot(finger_x - thumb_x, finger_y - thumb_y)
                is_pinching = pinch_dist < PINCH_THRESHOLD

                # 取五隻手指指尖 (4, 8, 12, 16, 20) 的平均座標作為板子控制中心
                sum_x = 0
                sum_y = 0
                for tip_idx in [4, 8, 12, 16, 20]:
                    lm = hand_landmarks_data[tip_idx]
                    sum_x += lm.x * WINDOW_WIDTH
                    sum_y += lm.y * WINDOW_HEIGHT
                
                center_x = int(sum_x / 5)
                center_y = int(sum_y / 5)

                # 更新手的跟蹤坐標（使用五指平均點）與狀態
                game.update_hand(center_x, center_y, is_pinching)

                # 判斷是否比出「耶」手勢 (✌️) 以開始遊戲
                if game.state in (STATE_START, STATE_GAME_OVER):
                    # 耶手勢特徵：食指與中指伸直（y 較小），無名指與小指彎曲（y 較大）
                    index_up = hand_landmarks_data[8].y < hand_landmarks_data[6].y
                    middle_up = hand_landmarks_data[12].y < hand_landmarks_data[10].y
                    ring_down = hand_landmarks_data[16].y > hand_landmarks_data[14].y
                    pinky_down = hand_landmarks_data[20].y > hand_landmarks_data[18].y
                    
                    if index_up and middle_up and ring_down and pinky_down:
                        game.start_game()

                # 在控制中心繪製追蹤圓點（捏合時轉換顏色，視覺回饋）
                finger_color = (255, 100, 50) if is_pinching else COLOR_FINGER
                pygame.draw.circle(render_surface, finger_color,
                                   (center_x, center_y), 8)
                pygame.draw.circle(render_surface, finger_color,
                                   (center_x, center_y), 12, 2)

                # 繪製手部骨架
                draw_hand_skeleton(render_surface, hand_landmarks_data,
                                   WINDOW_WIDTH, WINDOW_HEIGHT)

        # --- 更新遊戲物理（傳入 delta time 實現幀率無關）---
        # clock.tick() 回傳的毫秒數除以 1000 得到秒
        dt = clock.get_time() / 1000.0  # 上一幀實際花費的時間（秒）
        game.update(dt)

        # --- 繪製遊戲畫面 ---
        if game.state == STATE_START:
            game.draw_start_screen(render_surface)
        elif game.state == STATE_PLAYING:
            game.draw(render_surface)
            
            # --- 處理拍照 ---
            if game.take_photo_next_frame:
                # 複製目前的 render_surface
                photo = render_surface.copy()
                game.captured_photos.append(photo)
                game.take_photo_next_frame = False
                
        elif game.state == STATE_GAME_OVER:
            game.draw_game_over_screen(render_surface)

        # --- 顯示 FPS ---
        current_fps = int(clock.get_fps())
        draw_text_with_shadow(render_surface, f"FPS: {current_fps}",
                               (WINDOW_WIDTH - 100, 68),
                               fonts['fps'], (100, 255, 100))

        # --- 將渲染結果 blit 到螢幕（套用震動偏移）---
        screen.fill((0, 0, 0))  # 先清黑（震動時邊緣才不會殘影）
        shake_x = int(game.shake_offset_x)
        shake_y = int(game.shake_offset_y)
        screen.blit(render_surface, (shake_x, shake_y))

        # --- 更新螢幕（雙緩衝翻轉）---
        pygame.display.flip()

        # --- 控制 FPS ---
        clock.tick(FPS_TARGET)

    # ========================================================
    # 清理資源
    # ========================================================
    print(f"\n遊戲結束！感謝遊玩！")
    print(f"最高分數：{game.high_score}")

    # 釋放攝影機
    cap.release()
    # 釋放 MediaPipe 資源
    hands_detector.close()
    # 關閉 Pygame
    pygame.quit()


# ============================================================
# 程式進入點
# ============================================================
if __name__ == "__main__":
    main()
