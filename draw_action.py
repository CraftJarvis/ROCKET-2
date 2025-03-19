'''
Date: 2025-01-19 10:13:12
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-02-21 17:00:52
FilePath: /ROCKET-2/draw_action.py
'''

import cv2
import numpy as np
from rich import print
from rich.console import Console
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal


key_file_mapping = {
    "forward": {
        "release": "w-key.png", 
        "pressed": "w-key-pressed.png", 
        "bias": (0, 1),  # 第 1 行，第 2 列
    },
    "back": {
        "release": "s-key.png", 
        "pressed": "s-key-pressed.png", 
        "bias": (1, 1),  # 第 2 行，第 2 列
    }, 
    "left": {
        "release": "a-key.png", 
        "pressed": "a-key-pressed.png", 
        "bias": (1, 0),  # 第 2 行，第 1 列
    },
    "right": {
        "release": "d-key.png", 
        "pressed": "d-key-pressed.png", 
        "bias": (1, 2),  # 第 2 行，第 3 列
    },
    "inventory": {
        "release": "e-key.png", 
        "pressed": "e-key-pressed.png", 
        "bias": (0, 4.2),  # 第 1 行，第 5 列
    }, 
    "sprint": {
        "release": "shift-key.png", 
        "pressed": "shift-key-pressed.png", 
        "bias": (0, 3.2),  # 第 1 行，第 4 列
    }, 
    "sneak": {
        "release": "ctrl-key.png", 
        "pressed": "ctrl-key-pressed.png", 
        "bias": (1, 3.2),  # 第 2 行，第 4 列
    },
    "jump": {
        "release": "space-key.png", 
        "pressed": "space-key-pressed.png", 
        "bias": (1, 4.2),  # 第 2 行，第 5 列
    },
    "mouse": {
        "release": "mouse.png",
        "left-click": "left-click.png",
        "right-click": "right-click.png",
        "bias": (0.7, 5.4), 
    }
}

loaded_icon_files = None
asset_dir = Path(__file__).parent / "keyboard-assets"

def draw_action(frame: np.ndarray, action: Dict, start_point: Tuple=(350, 285)):

    global loaded_icon_files
    if loaded_icon_files is None:
        loaded_icon_files = {}
        for key, value in key_file_mapping.items():
            loaded_icon_files[key] = {
                png_key: Image.open(str(asset_dir / value[png_key])).convert("RGBA") \
                    for png_key in value.keys() \
                        if png_key in ["release", "pressed", "left-click", "right-click"]
            }
        # print(loaded_icon_files)
    
    # start_point = (350, 285)
    space = 2
    icon_size = 30
    
    image = Image.new("RGBA", frame.shape[:2][::-1], (255, 255, 255, 0))  # 初始完全透明
    
    # 从 start_point 开始绘制一个白色的透明底幕布， 透明度调节至 70%
    draw = ImageDraw.Draw(image)
    draw.rectangle([start_point[0] - 5, start_point[1] - 5,
                    start_point[0] + 280, 
                    start_point[1] + 65], 
                    fill=(255, 255, 255, 100))
    
    for key, value in action.items():
        if key not in key_file_mapping:
            continue
        if value == 1:
            icon = loaded_icon_files[key]["pressed"].resize((icon_size, icon_size))
        else:
            icon = loaded_icon_files[key]["release"].resize((icon_size, icon_size))
        bias = key_file_mapping[key]["bias"]
        start_x = int(start_point[0] + bias[1] * (icon_size + space))
        start_y = int(start_point[1] + bias[0] * (icon_size + space))
        image.paste(icon, (start_x, start_y), icon)
    
    # draw mouse 
    mouse_scale = 1.2
    mouse_size = (int(icon_size * mouse_scale), int(icon_size * mouse_scale))
    start_x = int(start_point[0] + key_file_mapping["mouse"]["bias"][1] * (icon_size + space))
    start_y = int(start_point[1] + key_file_mapping["mouse"]["bias"][0] * (icon_size + space))
    if action["attack"] == 1:
        left_click_icon = loaded_icon_files["mouse"]["left-click"].resize(mouse_size)
        image.paste(left_click_icon, (start_x, start_y), left_click_icon)
    elif action["use"] == 1:
        right_click_icon = loaded_icon_files["mouse"]["right-click"].resize(mouse_size)
        image.paste(right_click_icon, (start_x, start_y), right_click_icon)
    else:
        mouse_icon = loaded_icon_files["mouse"]["release"].resize(mouse_size)
        image.paste(mouse_icon, (start_x, start_y), mouse_icon)
    
    # draw circle
    circle_radius = 25
    circle_color = (0, 0, 0)
    circle_center = (int(start_point[0] + 242), 
                    int(start_point[1] + 30))
    draw = ImageDraw.Draw(image)
    draw.ellipse([circle_center[0] - circle_radius, circle_center[1] - circle_radius, 
                  circle_center[0] + circle_radius, circle_center[1] + circle_radius], 
                  outline=circle_color, width=3)
    # 经过圆心，绘制一横一竖两条两条线段, 代表 x, y 轴，使用点划线
    draw.line([circle_center[0] - circle_radius, circle_center[1], 
               circle_center[0] + circle_radius, circle_center[1]], 
               fill=circle_color, width=1, joint="curve")
    draw.line([circle_center[0], circle_center[1] - circle_radius,
                circle_center[0], circle_center[1] + circle_radius],
                fill=circle_color, width=1, joint="curve")
    
    camera = action["camera"]
    camera = np.clip(camera, -10, 10) / 10
    # 将 camera 所表示的圆映射到圆内，半径为 5
    draw.ellipse([circle_center[0] + int(camera[1] * circle_radius),
                  circle_center[1] + int(camera[0] * circle_radius),
                  circle_center[0] + int(camera[1] * circle_radius) + 5,
                  circle_center[1] + int(camera[0] * circle_radius) + 5],
                  fill=(0, 0, 200))
    
    ori_image = Image.fromarray(frame).convert("RGBA")
    image = Image.alpha_composite(ori_image, image)
    
    frame = np.array(image.convert("RGB"))
    return frame

if __name__ == '__main__':
    
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[..., 0] = 255
    dummy_action = {
        "forward": 1, 
        "back": 0, 
        "left": 0, 
        "right": 0, 
        "inventory": 0, 
        "sprint": 0, 
        "sneak": 0, 
        "jump": 0,
        "attack": 1,
        "use": 0, 
        "camera": np.array([5.0, -3.2])
    }
    frame = draw_action(frame, dummy_action)
    cv2.imwrite("action.png", frame)
    