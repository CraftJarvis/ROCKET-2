'''
Date: 2025-01-14 09:36:12
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-03-16 15:09:09
FilePath: /ROCKET-2/train.py
'''

import hydra
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from einops import rearrange
from typing import Dict, Any, Tuple

from minestudio.offline import MineLightning
from minestudio.offline.utils import convert_to_normal
from minestudio.offline.mine_callbacks import BehaviorCloneCallback
from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback, EMA
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, ActionKernelCallback, SegmentationKernelCallback
)

from model import CrossViewRocket
from model2 import CrossViewRocket2
from model3 import CrossViewRocket3
from model4 import CrossViewRocket4
from model5 import CrossViewRocket5
from model6 import CrossViewRocket6
from loss import PointPredictionCallback, PrePointPredictionCallback
from cross_view_dataset import CrossViewDataModule

logger = WandbLogger(project="minestudio")
# logger = None
@hydra.main(config_path='.', config_name='config')
def main(args):

    rocket_policy = CrossViewRocket(
        view_backbone=args.model.view_backbone,
        mask_backbone=args.model.mask_backbone,
        hiddim=args.model.hiddim,
        num_heads=args.model.num_heads,
        num_layers=args.model.num_layers,
        timesteps=args.model.timesteps,
        mem_len=args.model.mem_len, 
        use_prev_action=args.model.use_prev_action,
        num_view_tokens=args.model.num_view_tokens,
    )

    mine_lightning = MineLightning(
        mine_policy=rocket_policy, 
        log_freq=20,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=[
            BehaviorCloneCallback(weight=args.objective_weight),
            PointPredictionCallback(point_weight=0.1, bbox_weight=0.1, exist_weight=0.01),
        ], 
        hyperparameters=convert_to_normal(args),
    )

    mine_data = CrossViewDataModule(
        data_params=dict(
            dataset_dirs=args.dataset_dirs,
            modal_kernel_callbacks=[
                ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
                ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
                SegmentationKernelCallback(frame_width=224, frame_height=224), 
            ],
            win_len=128,
            split_ratio=args.split_ratio, 
            shuffle_episodes=args.shuffle_episodes,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        episode_continuous_batch=args.episode_continuous_batch,
    )

    callbacks=[
        LearningRateMonitor(logging_interval='step'), 
        SpeedMonitorCallback(),
        SmartCheckpointCallback(
            dirpath='./weights', filename='weight-{epoch}-{step}', save_top_k=-1, 
            every_n_train_steps=args.save_freq, save_weights_only=True,
        ), 
        SmartCheckpointCallback(
            dirpath='./checkpoints', filename='ckpt-{epoch}-{step}', save_top_k=1, 
            every_n_train_steps=args.save_freq+1, save_weights_only=False,
        ), 
        # EMA(
        #     decay=args.ema.decay, 
        #     validate_original_weights=args.ema.validate_original_weights, 
        #     every_n_steps=args.ema.every_n_steps, 
        #     cpu_offload=args.ema.cpu_offload, 
        # )
    ]

    L.Trainer(
        logger=logger, 
        devices=args.devices, 
        precision='bf16', 
        strategy='ddp_find_unused_parameters_true', 
        use_distributed_sampler=not args.episode_continuous_batch,
        callbacks=callbacks, 
        gradient_clip_val=1.0, 
        accumulate_grad_batches=args.accumulate_grad_batches, 
    ).fit(
        model=mine_lightning, 
        datamodule=mine_data, 
        ckpt_path=args.ckpt_path,
    )

if __name__ == '__main__':
    main()