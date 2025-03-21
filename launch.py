'''
Date: 2025-03-18 20:04:14
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-03-21 14:27:01
FilePath: /ROCKET2-OSS/launch.py
'''
import os
import cv2
import time
import yaml
import torch
import requests
import gradio as gr
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

from sam2.build_sam import build_sam2_camera_predictor
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    load_callbacks_from_config, MaskActionsCallback, PrevActionCallback, InitInventoryCallback, CommandsCallback
)
from model import CrossViewRocket, load_cross_view_rocket
from cfg_wrapper import CFGWrapper
from draw_action import draw_action


DEFAULT_IMAGE_URL = "https://pic1.imgdb.cn/item/67d973a688c538a9b5c048fd.png"
start_image = Image.open(requests.get(DEFAULT_IMAGE_URL, stream=True).raw)
SCALE = 360 / 640
PAGE_WIDTH = 800
PAGE_HEIGHT = int(PAGE_WIDTH * SCALE)
custom_css = """
#custom-container {
    width: 800px;
    margin: 0 auto; /* 居中 */
}
"""

TUTORIAL = """
## General Guidelines To Launch ROCKET-2

1. Customize the environment by filling the yaml block in the `Customize Environment` tab. 
+ Remember to click the **Set** button for updating the environment config. 

2. Reset the environment and ROCKET-2 model by clicking the **Reset** button in the `Launch Rocket` tab. 

3. Select the cross-view image and segment the object in `Specify Goal` tab.
+ We provide three ways to select the cross-view image: upload, gallery, and history observations. 
+ You can generate the point prompt for SAM-2 by directly clicking on the cross-view image. 
+ Click the **Segment** button to segment the object in the cross-view image. 

4. Get back to the ``Launch Rocket`` tab and click the **Go** button to start the ROCKET-2 rollout. 
+ You can change the rollout steps by adjusting the **Steps** number.

5. Record the rollout trajectory in the `Record Video` tab. 

## Important Tips

1. If ROCKET-2 is stuck, please click the **Clear Memory** button to clear the agent memory.

2. Minimizing the temporal interval between current agent view and the cross view image leading to better performance.
+ So, please try to change the cross-view image more frequently. 
"""

DEFAULT_CODE = """
spawn_positions:
  - 
    seed: 19961103
    position: [-79, 64, -512]
  - 
    seed: 19961103
    position: [-145, 67, -495]
  - 
    seed: 19961103
    position: [-312, 69, -509]

init_inventory: 
  -
    slot: 0
    type: iron_axe
    quantity: 1

masked_actions: 
  hotbar.1: 0
  hotbar.2: 0
  hotbar.3: 0
  hotbar.4: 0
  hotbar.5: 0
  hotbar.6: 0
  hotbar.7: 0
  hotbar.8: 0
  hotbar.9: 0
"""

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

SEGMENT_MAPPING = {
    "Hunt": 0, "Use": 3, "Mine": 2, "Interact": 3, "Craft": 4, "Switch": 5, "Approach": 6, "None": -1
}

def apply_two_images(bg_image, fg_image):
    scale = 0.33
    fg_image = cv2.copyMakeBorder(fg_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    fg_image = cv2.resize(fg_image, (int(640*scale), int(360*scale)), interpolation=cv2.INTER_LINEAR)
    sx, sy, ex, ey = bg_image.shape[1]-5-fg_image.shape[1], 5, bg_image.shape[1]-5, 5+fg_image.shape[0]
    bg_image[sy:ey, sx:ex] = bg_image[sy:ey, sx:ex] * 0.0 + 1.0 * fg_image
    return bg_image

class CrossViewRocketSession:
    
    def __init__(self, sam_path: str):
        start_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.current_image = np.array(start_image)
        self.sam_path = sam_path
        self.clear_points()
        
        self.sam_choice = 'base'
        self.load_sam()
        
        self.tracking_flag = True
        self.points = []
        self.points_label = []
        self.able_to_track = False
        self.segment_type = "None"
        self.obj_mask = np.zeros((360, 640), dtype=np.float32)
        self.cross_view_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.calling_rocket = False
        self.num_steps = 0
        self.env = None
        self.env_conf = None
    
    def clear_points(self):
        self.points = []
        self.points_label = []
    
    def clear_obj_mask(self):
        self.obj_mask = np.zeros((360, 640), dtype=np.float32)
        self.cross_view_image = np.zeros((360, 640, 3), dtype=np.uint8)
    
    def clear_agent_memory(self):
        self.num_steps = 0
        if hasattr(self, "agent"):
            self.state = self.agent.initial_state()

    def load_sam(self):
        
        ckpt_mapping = {
            'large': [os.path.join(self.sam_path, "sam2_hiera_large.pt"), "sam2_hiera_l.yaml"],
            'base': [os.path.join(self.sam_path, "sam2_hiera_base_plus.pt"), "sam2_hiera_b+.yaml"],
            'small': [os.path.join(self.sam_path, "sam2_hiera_small.pt"), "sam2_hiera_s.yaml"], 
            'tiny': [os.path.join(self.sam_path, "sam2_hiera_tiny.pt"), "sam2_hiera_t.yaml"]
        }
        sam_ckpt, model_cfg = ckpt_mapping[self.sam_choice]
        # first realease the old predictor
        if hasattr(self, "predictor"):
            del self.predictor
        self.predictor = build_sam2_camera_predictor(model_cfg, sam_ckpt)
        print(f"Successfully loaded SAM2 from {sam_ckpt}")
        self.able_to_track = False

    def load_rocket(self, ckpt_path, cfg_coef=1.0):
        if ckpt_path.startswith("hf:"):
            ckpt_path = ckpt_path.split(":")[-1]
            agent = CrossViewRocket.from_pretrained(ckpt_path).to("cuda")
        else:
            assert os.path.exists(ckpt_path), f"Model path {ckpt_path} not found."
            agent = load_cross_view_rocket(ckpt_path).to("cuda")
        agent.eval()
        self.agent = CFGWrapper(agent, k=cfg_coef)
        self.clear_agent_memory()

    def segment(self):
        self.predictor.load_first_frame(self.cross_view_image)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=0, 
            obj_id=0,
            points=self.points,
            labels=self.points_label,
        )
        self.obj_mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy() # 360, 640
        self.clear_points()
        return self.obj_mask

    def reset(self):
        self.obs_history = []
        self.image_history = []
        callbacks = load_callbacks_from_config(self.env_conf)
        callbacks.extend([
            PrevActionCallback(), 
            # MaskActionsCallback(**{"hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0}),
        ])
        self.env = MinecraftSim(callbacks=callbacks)
        self.obs, self.info = self.env.reset()
        for i in range(30): #! better init
            time.sleep(0.1)
            noop_action = self.env.noop_action()
            self.obs, self.reward, terminated, truncated, self.info = self.env.step(noop_action)
        self.reward = 0
        self.current_image = self.info["pov"]
        # save_image = np.concatenate([self.current_image, self.current_image], axis=0)
        self.obs_history.append(self.current_image)
        self.image_history.append(self.current_image)
        return self.current_image
    
    def apply_mask(self):   
        image = self.cross_view_image.copy()
        color = COLORS[ SEGMENT_MAPPING[self.segment_type] ]
        color = np.array(color).reshape(1, 1, 3)[:, :, ::-1]
        obj_mask = (self.obj_mask[..., None] * color).astype(np.uint8)
        image = cv2.addWeighted(image, 1.0, obj_mask, 0.5, 0.0)
        return image
    
    def step(self, input_action=None):
        pred_point = None
        pred_bbox  = None
        pred_exist = None
        if input_action is not None:
            action = input_action
        else:
            obj_id = torch.tensor( SEGMENT_MAPPING[self.segment_type] )
            obj_mask = self.obj_mask.astype(np.uint8)
            obj_mask = cv2.resize(obj_mask, (224, 224), interpolation=cv2.INTER_LINEAR)
            obj_mask = torch.tensor(obj_mask, dtype=torch.uint8)
            cross_view_image = cv2.resize(self.cross_view_image, (224, 224), interpolation=cv2.INTER_LINEAR)
            obs = {
                'image': self.obs['image'], 
                'env_prev_action': self.obs['env_prev_action'],
                'cross_view': {
                    'cross_view_image': cross_view_image, 
                    'cross_view_obj_id': obj_id, 
                    'cross_view_obj_mask': obj_mask, 
                }
            }
            action, self.state = self.agent.get_action(obs, self.state, input_shape="*")
            if 'bbox' in self.agent.cache_latents:
                pred_bbox = self.agent.cache_latents['bbox'].cpu().numpy()
            pred_point = self.agent.cache_latents['point'].cpu().numpy()
            pred_exist = self.agent.cache_latents['exist'].sigmoid().cpu().item()
        
        self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.current_image = self.info["pov"]
        image = self.current_image.copy()
        
        if self.able_to_track and self.tracking_flag:
            self.segment()
            image = self.apply_mask()
        else:
            time.sleep(0.001)

        if "buttons" in action:
            image = draw_action(image, self.env.agent_action_to_env_action(action))
        else:
            image = draw_action(image, action)

        if pred_point is not None:
            point = (pred_point * np.array([image.shape[0], image.shape[1]])).astype(np.int32)
            # cv2.circle(image, tuple(point[::-1]), 10, (162, 210, 255), -1)
            center_y, center_x = point
            cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), (255, 255, 255), 1)
            cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), (255, 255, 255), 1)

        if pred_exist is not None:
            image = cv2.rectangle(image, (10, 40), (10 + 250, 50), (100, 100, 100), -1)
            image = cv2.rectangle(image, (10, 40), (10 + int(250 * pred_exist), 50), (255, 255, 255), -1)

        if pred_bbox is not None:
            bbox_scale = np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            x_min, y_min, x_max, y_max = (pred_bbox * bbox_scale).astype(np.int32)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        self.num_steps += 1
        # get cross_view image with segmentation mask
        cross_view_image_copy = self.cross_view_image.copy()
        color = COLORS[ SEGMENT_MAPPING[self.segment_type] ]
        color = np.array(color).reshape(1, 1, 3)[:, :, ::-1]
        obj_mask = (self.obj_mask[..., None] * color).astype(np.uint8)
        cross_view_image_with_mask = cv2.addWeighted(cross_view_image_copy, 1.0, obj_mask, 0.5, 0.0)
        binary_mask = (self.obj_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_color = (255, 0, 0)
        line_thickness = 2
        cv2.drawContours(cross_view_image_with_mask, contours, -1, contour_color, line_thickness)
        
        save_image = apply_two_images(bg_image=image, fg_image=cross_view_image_with_mask)
        
        # save_image = np.concatenate([image, cross_view_image_with_mask], axis=0)
        # cv2.putText(save_image, f"Rollout by Gradio", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.obs_history.append(self.current_image)
        self.image_history.append(save_image)
        info = {
            'visibility': pred_exist, 
            'cross_view_image_with_mask': cross_view_image_with_mask,
        }
        return save_image, info


def draw_components(args):

    with gr.Blocks(theme="shivi/calm_seafoam", css=custom_css) as demo:
        links = gr.HTML(
            '''
            <div align="center">
                <h1>ROCKET-2 Interactive Demo</h1>
            </div>
            <br/>
            <div style="text-align: center;"> Paper Authors: 
            [<a href="https://phython96.github.io/" style="text-decoration: none;">Shaofei Cai</a>] 
            [<a href="https://muzhancun.github.io/" style="text-decoration: none;">Zhancun Mu</a>] 
            [<a href="https://liuanji.github.io/" style="text-decoration: none;">Anji Liu</a>] 
            [<a href="https://scholar.google.com/citations?user=KVzR1XEAAAAJ&hl=zh-CN" style="text-decoration: none;">Yitao Liang</a>] 
            </div>
            <br/>
            <div style="text-align: center;"> About ROCKET-2: 
            [<a href="https://arxiv.org/abs/2503.02505" style="text-decoration: none;">Paper</a>] 
            [<a href="https://craftjarvis.github.io/ROCKET-2" style="text-decoration: none;">Website</a>] 
            [<a href="https://github.com/CraftJarvis/ROCKET-2" style="text-decoration: none;">GitHub</a>] | Related Links: 
            [<a href="https://craftjarvis.github.io/" style="text-decoration: none;">CraftJarvis</a>] 
            [<a href="https://github.com/CraftJarvis/MineStudio" style="text-decoration: none;">MineStudio</a>] 
            [<a href="https://craftjarvis.github.io/ROCKET-1" style="text-decoration: none;">ROCKET-1</a>]
            </div>
            ''', elem_id="custom-container"
        )
        # with gr.Row(elem_id="custom-container"):
        #     upload_button = gr.UploadButton(label="Upload Cross-View Image", file_types=["image"])
        
        rocket_session = gr.State(CrossViewRocketSession(sam_path=args.sam_path))
        
        with gr.Row(elem_id="custom-container"):

            with gr.Tabs():
                with gr.TabItem("Tutorial"):
                    gr.Markdown(TUTORIAL)
                
                with gr.TabItem("Customize Environment"):
                    set_status = gr.Textbox(show_label=False, placeholder="Setting status will display here. ", interactive=False)
                    config_code = gr.Code(
                        value=DEFAULT_CODE, 
                        language="yaml", label="Environment Config", interactive=True
                    )

                    with gr.Row(equal_height=True):
                        env_files = [x for x in Path(args.env_conf).glob("*.yaml")]
                        # prompt = gr.Textbox(value="Config List", show_label=False, interactive=False, scale=1)
                        choices = sorted([str(x) for x in env_files])
                        default = [x for x in choices if 'wood' in x][0]
                        conf_source = gr.Dropdown(
                            value=default, 
                            choices=choices,
                            show_label=False, interactive=True, scale=4
                        )
                        load_conf_btn = gr.Button("Load", scale=1)
                        set_conf_btn = gr.Button("Set", variant="primary", scale=1)

                with gr.TabItem("Launch Rocket"):

                    with gr.Row(equal_height=True):
                        status = gr.Textbox(show_label=False, placeholder="Running status will display here. ", interactive=False, scale=2)
                        visibility = gr.Label({"Visibility": 0.00}, show_label=False, show_heading=False, scale=1)
                    
                    display = gr.Image(
                        value=start_image, 
                        interactive=False, 
                        show_label=False, 
                        show_download_button=False, 
                        show_fullscreen_button=False,
                        streaming=True,     
                        height=int(PAGE_WIDTH*SCALE), 
                        width=PAGE_WIDTH
                    )

                    with gr.Tabs():

                        with gr.TabItem("Control Panel"):
                            with gr.Row(equal_height=True):
                                reset_btn = gr.Button("Reset", scale=1)
                                clr_mem_btn = gr.Button("Clear Memory", scale=1)
                                steps = gr.Number(
                                    value=30, minimum=1, maximum=1200,
                                    show_label=False, interactive=True, scale=1
                                )
                                launch_btn = gr.Button("Go", variant="primary", interactive=False, scale=1)

                        with gr.TabItem("Commands Panel"):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=4):
                                    command = gr.Dropdown(
                                        choices=[
                                            '/setblock ~0 ~0 ~4 minecraft:diamond_block', 
                                        ],
                                        show_label=False, interactive=True, allow_custom_value=True
                                    )
                                with gr.Column(scale=1):
                                    send_cmd_btn = gr.Button("Send Command", scale=1)

                        with gr.TabItem("Setting Panel"):
                            with gr.Row(equal_height=True):
                                ckpt_path = gr.Dropdown(
                                    label="Chekpoint Path",
                                    choices=[
                                        "hf:phython96/ROCKET-2-1.5x-17w", 
                                        "hf:phython96/ROCKET-2-1x-22w", 
                                    ], 
                                    show_label=True, interactive=True, allow_custom_value=True, scale=4
                                )
                                cfg_coef = gr.Number(
                                    label="Classifier-Free", 
                                    value=1.5, minimum=0.0, maximum=10.0, step=0.1,
                                    show_label=True, interactive=True, scale=1
                                )
                                set_rocket_btn = gr.Button("Set", scale=1)

                with gr.TabItem("Specify Goal"):
                    
                    with gr.Row(equal_height=True):
                        upload_button = gr.UploadButton(label="Upload Cross-View Image", file_types=["image"], scale=1)
                        cross_view_slider = gr.Slider(label="Select Cross-View Image from History Observations", minimum=0, maximum=0, value=0, step=1, interactive=False, scale=2)
                    
                    cross_view_display = gr.Image(
                        value=start_image, 
                        interactive=False, 
                        show_label=False, 
                        show_download_button=False, 
                        show_fullscreen_button=False,
                        streaming=True,     
                        height=int(PAGE_WIDTH*SCALE), 
                        width=PAGE_WIDTH
                    )
                    
                    with gr.Row(equal_height=True):
                        sam_choice = gr.Radio(
                            choices=["large", "base", "small", "tiny"],
                            value="base",
                            label="Select SAM2 Checkpoint",
                            interactive=True, scale=5,
                        )
                        add_or_remove = gr.Radio(
                            choices=["Pos", "Neg"], value="Pos", 
                            label="Add Point Prompt",
                            interactive=True, scale=3
                        )
                        clr_point_btn = gr.Button("Clear Points", scale=1)
                    
                    with gr.Row(equal_height=True):
                        prompt = gr.Textbox(value="Interaction Type", show_label=False, interactive=False, scale=1)
                        segment_type = gr.Dropdown(
                            choices=["Approach", "Interact", "Hunt", "Mine", "Craft", "None"],
                            show_label=False, interactive=True, scale=7
                        )
                        segment_btn = gr.Button("Segment", scale=1, variant="primary")

                    images = [image_path for image_path in Path("./gallery").glob("*.png")]
                    gallery = gr.Gallery(
                        images, 
                        label="Select Cross-Episode Images in Gallery",
                        columns=[5], rows=[1], object_fit="contain", height=150, 
                        show_share_button=False, show_download_button=False, interactive=True, show_fullscreen_button=False
                    )

                with gr.TabItem("Record Video"):
                    # add video component
                    record_video = gr.Video(show_label=False, interactive=False)
                    with gr.Row(equal_height=True):
                        make_video_btn = gr.Button("Make Video")
                        download_btn = gr.DownloadButton("Download!", variant='primary', interactive=False)

            # process: load config
            def load_conf_btn_fn(conf_source):
                config_code = open(conf_source).read()
                return config_code
            load_conf_btn.click(load_conf_btn_fn, inputs=[conf_source], outputs=[config_code])

            # process: set config
            def set_conf_btn_fn(config_code, rocket_session):
                try:
                    env_conf = yaml.safe_load(config_code)
                    rocket_session.env_conf = env_conf
                    return "Environment Config Set Successfully."
                except:
                    import traceback
                    return traceback.format_exc()
            set_conf_btn.click(set_conf_btn_fn, inputs=[config_code, rocket_session], outputs=[set_status])
            
            # process: reset
            def reset_btn_fn(ckpt_path, cfg_coef, rocket_session):
                if rocket_session.env_conf is None:
                    return start_image, gr.update(), "Please set environment config first.", None, gr.update(interactive=False)
                rocket_session.load_rocket(ckpt_path, cfg_coef)
                image = rocket_session.reset()
                rocket_session.cross_view_image = image
                cross_view_slider = gr.update(value=0, maximum=len(rocket_session.obs_history)-1, interactive=True)
                return image, image, "Environment Reset Successfully.", cross_view_slider, gr.update(interactive=True)
            reset_btn.click(reset_btn_fn, inputs=[ckpt_path, cfg_coef, rocket_session], outputs=[display, cross_view_display, status, cross_view_slider, launch_btn])
            
            # process: set checkpoint
            def set_rocket_btn_fn(ckpt_path, cfg_coef, rocket_session):
                rocket_session.load_rocket(ckpt_path, cfg_coef)
                return f"Rocket Model with ckpt {ckpt_path}, CFG {cfg_coef};"
            set_rocket_btn.click(set_rocket_btn_fn, inputs=[ckpt_path, cfg_coef, rocket_session], outputs=[status])
            
            # process: upload cross-view image
            def upload_image_fn(image_file, rocket_session):
                image = Image.open(image_file).convert("RGB")
                image = np.array(image)
                image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
                rocket_session.cross_view_image = image
                return image
            upload_button.upload(upload_image_fn, inputs=[upload_button, rocket_session], outputs=[cross_view_display])
            
            # process: select history observation with slider
            def select_obs_fn(cross_view_slider, rocket_session):
                image = rocket_session.obs_history[cross_view_slider]
                rocket_session.cross_view_image = image
                return image
            cross_view_slider.release(select_obs_fn, inputs=[cross_view_slider, rocket_session], outputs=[cross_view_display])

            # process: select from gallery
            def select_gallery_fn(image_path, rocket_session, evt: gr.SelectData):
                # import ipdb; ipdb.set_trace()
                image = Image.open(evt.value['image']['path']).convert("RGB")
                image = np.array(image)
                image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
                rocket_session.cross_view_image = image
                return image
            gallery.select(select_gallery_fn, inputs=[gallery, rocket_session], outputs=[cross_view_display])

            # process: select SAM2 checkpoint
            def select_sam2(sam_choice, rocket_session):
                rocket_session.sam_choice = sam_choice
                rocket_session.load_sam()
            sam_choice.select(fn=select_sam2, inputs=[sam_choice, rocket_session], outputs=[], show_progress=True)
            
            # process: draw points (positive and negtive)
            def draw_points_fn(image, label, rocket_session, evt: gr.SelectData):
                points = rocket_session.points
                point_label = rocket_session.points_label
                x, y = evt.index[0], evt.index[1]
                rx, ry = evt.index[0] / image.shape[1], evt.index[1] / image.shape[0]
                bx = (PAGE_WIDTH - image.shape[1]) // 2
                by = (PAGE_HEIGHT - image.shape[0]) // 2
                x = np.clip(int(rx * PAGE_WIDTH - bx), a_min=0, a_max=image.shape[1])
                y = np.clip(int(ry * PAGE_HEIGHT - by), a_min=0, a_max=image.shape[0])
                point_radius, point_color = 3, (0, 255, 0) if label == 'Pos' else (255, 0, 0)
                points.append([x, y])
                point_label.append(1 if label == 'Pos' else 0)
                image = np.copy(image)
                cv2.circle(image, (x, y), point_radius, point_color, -1)
                return image
            cross_view_display.select(draw_points_fn, inputs=[cross_view_display, add_or_remove, rocket_session], outputs=[cross_view_display])
            
            # process: clear points 
            def clear_points_fn(rocket_session):
                rocket_session.clear_points()
                return rocket_session.cross_view_image
            clr_point_btn.click(clear_points_fn, inputs=[rocket_session], outputs=[cross_view_display], show_progress=True)
            
            # process: change segment-type
            def change_segment_fn(segment_type, rocket_session):
                rocket_session.segment_type = segment_type
            segment_type.change(change_segment_fn, inputs=[segment_type, rocket_session], outputs=[], show_progress=True)
            
            # process: do segmenting
            def segment_fn(segment_type, rocket_session):
                rocket_session.segment_type = segment_type
                if len(rocket_session.points) == 0:
                    return rocket_session.cross_view_image
                rocket_session.segment()    
                image = rocket_session.apply_mask()
                return image
            segment_btn.click(segment_fn, inputs=[segment_type, rocket_session], outputs=[cross_view_display], show_progress=True)
            
            # process: do clear memory
            def clear_memory_fn(rocket_session):
                rocket_session.clear_agent_memory()
                return "Agent Memory Cleared."
            clr_mem_btn.click(clear_memory_fn, inputs=[rocket_session], outputs=[status])
            
            # process: launch rocket
            def launch_fn(steps, rocket_session):
                for i in range(steps):
                    display_image, info = rocket_session.step()
                    # display_image = apply_two_images(bg_image=image, fg_image=info['cross_view_image_with_mask'])
                    status = f"Current Step: {i+1}/{steps}, Memory Length: {rocket_session.num_steps}. "
                    cross_view_slider = gr.update(maximum=len(rocket_session.obs_history)-1, interactive=True)
                    visibility = {'visibility': info['visibility']}
                    yield display_image, status, visibility, cross_view_slider, 
            launch_btn.click(launch_fn, inputs=[steps, rocket_session], outputs=[display, status, visibility, cross_view_slider], show_progress=False)
            
            # process: send cheat command
            def send_command_fn(command, rocket_session):
                try:
                    rocket_session.env.env.execute_cmd(command)
                    for i in range(10): 
                        time.sleep(0.1)
                        noop_action = rocket_session.env.noop_action()
                        obs, reward, terminated, truncated, info = rocket_session.env.step(noop_action)
                    return info["pov"], f"Command Sent: {command}"
                except:
                    return gr.update(), "Please reset the environment first."
            send_cmd_btn.click(send_command_fn, inputs=[command, rocket_session], outputs=[display, status])
            
            # process: make video
            def make_video_fn(rocket_session, progress=gr.Progress()):
                images = rocket_session.image_history
                if len(images) == 0:
                    return gr.update(interactive=False), gr.update(interactive=True), None
                filepath = "rocket.mp4"
                h, w = images[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(filepath, fourcc, 25.0, (w, h))
                for image in progress.tqdm(images):
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    video.write(image)
                video.release()
                rocket_session.image_history = []
                return gr.update(interactive=False), gr.update(value=filepath, interactive=True), filepath
            make_video_btn.click(make_video_fn, inputs=[rocket_session,], outputs=[make_video_btn, download_btn, record_video], show_progress=True)
            
            # process: download
            def save_video_fn():
                return gr.update(interactive=True), gr.update(interactive=False), None
            download_btn.click(save_video_fn, inputs=[], outputs=[make_video_btn, download_btn, record_video], show_progress=False)

        demo.queue()
        demo.launch(share=False, server_name='0.0.0.0', server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--env-conf", type=str, default=None)
    parser.add_argument("--sam-path", type=str)
    args = parser.parse_args()
    draw_components(args)
