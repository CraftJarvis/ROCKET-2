'''
Date: 2025-01-10 05:33:50
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-02-21 15:00:30
FilePath: /ROCKET-2/cross_view_dataset.py
'''
import math
import random
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np

from rich import print
from rich.console import Console
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.data.minecraft import RawDataset, RawDataModule
from minestudio.data.minecraft.callbacks import SegmentationDrawFrameCallback

import numpy as np

def mask_to_bounding_box_batch(masks):
    """
    Convert a batch of 2D binary masks to bounding boxes using numpy vectorized operations.
    
    Args:
    - masks (np.ndarray): A numpy array of shape (b, 224, 224), where b is the batch size.
    
    Returns:
    - bounding_boxes (np.ndarray): A numpy array of shape (b, 4), where each row is a bounding box
      (x_min, y_min, x_max, y_max).
    """
    n_rows = masks.shape[1]
    n_cols = masks.shape[2]
    # Find which rows and columns contain any non-zero values (i.e., 1s)
    rows = np.any(masks, axis=2)  # (b, 224)
    cols = np.any(masks, axis=1)  # (b, 224)
    
    # Get the first and last row with non-zero values
    y_min = np.argmax(rows, axis=1)  # (b,)
    y_max = n_rows - np.argmax(np.flip(rows, axis=1), axis=1) - 1  # (b,)
    
    # Get the first and last column with non-zero values
    x_min = np.argmax(cols, axis=1)  # (b,)
    x_max = n_cols - np.argmax(np.flip(cols, axis=1), axis=1) - 1  # (b,)
    
    # Normalize to [0, 1]
    x_min = x_min / n_cols
    x_max = x_max / n_cols
    y_min = y_min / n_rows
    y_max = y_max / n_rows
    
    # Combine the coordinates into a single array of shape (b, 4)
    bounding_boxes = np.stack((x_min, y_min, x_max, y_max), axis=1)
    
    return bounding_boxes

class CrossViewDataset(RawDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_cross_view(self, episode: str, frame_range: Tuple[int, int], max_retries: int=3, event_constrain=None) -> Tuple[int, int]:
        candidate_choices = list(range(frame_range[0], frame_range[1]+1))
        for i in range(max_retries):
            frame_id = random.choice(candidate_choices)
            candidate_choices.remove(frame_id)
            frame = self.kernel_manager.read(episode, frame_id, 1, 1, event_constrain=event_constrain)
            if frame['segmentation']['obj_mask'][0].sum() > 0 or len(candidate_choices) == 0:
                return frame_id, frame
        return frame_id, frame

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert idx < len(self), f"Index <{idx}> out of range <{len(self)}>"
        episode, relative_idx = self.locate_item(idx)
        start = max(0, relative_idx * self.win_len) # if start > 0 is the prequest for previous action
        item = self.kernel_manager.read(episode, start, self.win_len, self.skip_frame)

        # === apply the cross view for each frame === #
        cross_view = {
            "cross_view_image": [], 
            "cross_view_obj_id": [],
            "cross_view_obj_mask": [],
            "cross_view_point": [],
            "cross_view_event": [],
            "cross_view_frame_id": [],
        }
        cross_view_mapping = {}
        for wid, frame_range in enumerate(item['segmentation']['frame_range']):
            frame_range_key = f"{frame_range[0]}_{frame_range[1]}"
            if frame_range[0] == -1:
                cross_view['cross_view_image'].append(np.zeros_like(item['image'][wid]))
                cross_view['cross_view_obj_id'].append(-1)
                cross_view['cross_view_obj_mask'].append(np.zeros_like(item['segmentation']['obj_mask'][wid]))
                cross_view['cross_view_point'].append((-1, -1))
                cross_view['cross_view_event'].append("")
                cross_view['cross_view_frame_id'].append(-1)
                continue
            if frame_range_key not in cross_view_mapping:
                cross_view_frame_range = (frame_range[0], min(frame_range[1], self.episodes_with_length[episode]-1))
                current_event = item['segmentation']['event'][wid]
                cross_view_frame_id, single_frame_item = self.sample_cross_view(episode, cross_view_frame_range, max_retries=5, event_constrain=current_event)
                cross_view_mapping[frame_range_key] = {
                    'cross_view_image': single_frame_item['image'][0],
                    'cross_view_obj_id': single_frame_item['segmentation']['obj_id'][0],
                    'cross_view_obj_mask': single_frame_item['segmentation']['obj_mask'][0],
                    'cross_view_point': single_frame_item['segmentation']['point'][0],
                    'cross_view_event': single_frame_item['segmentation']['event'][0],
                    'cross_view_frame_id': single_frame_item['segmentation']['frame_id'][0],
                }
            cross_view['cross_view_image'].append(cross_view_mapping[frame_range_key]['cross_view_image'])
            cross_view['cross_view_obj_id'].append(cross_view_mapping[frame_range_key]['cross_view_obj_id'])
            cross_view['cross_view_obj_mask'].append(cross_view_mapping[frame_range_key]['cross_view_obj_mask'])
            cross_view['cross_view_point'].append(cross_view_mapping[frame_range_key]['cross_view_point'])
            cross_view['cross_view_event'].append(cross_view_mapping[frame_range_key]['cross_view_event'])
            cross_view['cross_view_frame_id'].append(cross_view_mapping[frame_range_key]['cross_view_frame_id'])

        for key in cross_view:
            if not isinstance(cross_view[key][0], str):
                cross_view[key] = np.stack(cross_view[key], axis=0)
        item["cross_view"] = cross_view
        # === apply the cross view for each frame === #

        for key in list(item.keys()):
            if key.endswith('mask'):
                mask = item.pop(key)
        item["mask"] = mask
        # item["point_dropout"] = (np.random.uniform(0, 1, item["mask"].size) > 0.9).astype(np.float32)
        item["prev_action_dropout"] = (np.random.uniform(0, 1, item["mask"].size) > 0.75).astype(np.float32)
        item['segmentation']['bbox'] = mask_to_bounding_box_batch(item["segmentation"]['obj_mask'])

        item['text'] = 'raw'
        item['timestamp'] = np.arange(start, start+self.win_len, self.skip_frame)
        item['episode'] = episode
        episode_samples = math.ceil(self.episodes_with_length[episode] / self.win_len)
        item['progress'] = f"{relative_idx}/{episode_samples}"
        item = self.to_tensor(item)
        return item


class CrossViewDrawFrameCallback(SegmentationDrawFrameCallback):

    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int) -> np.ndarray:
        cache_frames = []
        for frame_idx, frame in enumerate(frames):
            frame_up = frame.copy()
            cross_view_info = infos['cross_view']
            frame_down = cross_view_info['cross_view_image'][sample_idx][frame_idx].numpy()
            obj_id = cross_view_info['cross_view_obj_id'][sample_idx][frame_idx].item()
            obj_mask = cross_view_info['cross_view_obj_mask'][sample_idx][frame_idx].numpy()
            point = (cross_view_info['cross_view_point'][sample_idx][frame_idx][1].item(), 
                     cross_view_info['cross_view_point'][sample_idx][frame_idx][0].item())
            event = cross_view_info['cross_view_event'][sample_idx][frame_idx]
            cross_view_frame_id = cross_view_info['cross_view_frame_id'][sample_idx][frame_idx].item()
            frame_down = self.draw_frame(frame_down, point, obj_mask, obj_id, event, cross_view_frame_id)
            
            frame = np.concatenate([frame_up, frame_down], axis=0)
            cache_frames.append(frame)
        return cache_frames

class CrossViewDataModule(RawDataModule):
    
    def setup(self, stage: Optional[str]=None):
        self.train_dataset = CrossViewDataset(split='train', **self.data_params)
        self.val_dataset = CrossViewDataset(split='val', **self.data_params)
