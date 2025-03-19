'''
Date: 2025-03-19 20:24:46
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-03-19 20:27:55
FilePath: /ROCKET2-OSS/model.py
'''

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from einops import rearrange, repeat
from typing import List, Dict, Any, Tuple, Optional
from rich import print

import timm
from minestudio.models.base_policy import MinePolicy
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.utils.register import Registers

BINARY_KEYS = [
    "forward", "back", "left", "right", "inventory", "sprint", "sneak", "jump", "attack", "use", 
    "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5", "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9"
]

class ActionEmbeddingLayer(nn.Module):
    
    def __init__(self, hiddim: int):
        super().__init__()
        self.camera_layer = nn.Linear(2, hiddim)
        self.binary_layers = nn.ModuleDict({
            f"act_{key}": nn.Embedding(2, hiddim) for key in BINARY_KEYS
        })
    
    def forward(self, action: Dict) -> torch.Tensor:
        x = self.camera_layer(action['camera'].float())
        for key in BINARY_KEYS:
            x += self.binary_layers[f"act_{key}"](action[key.replace("_", ".")])
        return x

class CrossViewRocket(MinePolicy):
    
    def __init__(self, 
        view_backbone: str = 'timm/vit_base_patch16_224.dino', 
        mask_backbone: str = 'timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k', 
        hiddim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        timesteps: int = 128,
        mem_len: int = 128,
        use_prev_action: bool = False,
        num_view_tokens: int = 1,
        action_space = None,
        **kwargs,
    ):
        super().__init__(hiddim=hiddim, action_space=action_space) 
        # super().__init__(hiddim=hiddim, action_space=action_space, nucleus_prob=0.85)
        self.view_backbone = timm.create_model(view_backbone, pretrained=True, features_only=True)
        data_config = timm.data.resolve_model_data_config(self.view_backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        self.mask_backbone = timm.create_model(mask_backbone, pretrained=True, features_only=True, in_chans=1)
        self.updim_obs = nn.Conv2d(self.view_backbone.feature_info[-1]['num_chs'], hiddim, kernel_size=1, bias=False)
        vision_dim = self.view_backbone.feature_info[-1]['num_chs'] + self.mask_backbone.feature_info[-1]['num_chs']
        self.updim_cross = nn.Conv2d(vision_dim, hiddim, kernel_size=1, bias=False)
        self.num_view_tokens = num_view_tokens
        self.view_cls_tokens = nn.Parameter(torch.randn(1, self.num_view_tokens, hiddim) * 1e-3)
        self.view_resampler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim, 
                nhead=num_heads, 
                dim_feedforward=hiddim*2, 
                dropout=0.1,
                batch_first=True
            ), 
            num_layers=num_layers,
        )
        self.interaction = nn.Embedding(10, hiddim) # denotes the number of interaction types
        self.num_step_tokens = self.num_view_tokens + 1
        
        self.index_bias = -2
        self.use_prev_action = use_prev_action
        if self.use_prev_action:
            self.action_embedding_layer = ActionEmbeddingLayer(hiddim)
            self.num_step_tokens += 1
            self.index_bias -= 1

        self.dropout_embedding = nn.Parameter(torch.randn(1, 1, hiddim) * 1e-3)

        print(f"number of tokens per timestep is {self.num_step_tokens}")
        self.recurrent = ResidualRecurrentBlocks(
            hidsize=hiddim,
            timesteps=timesteps*self.num_step_tokens, 
            recurrence_type="transformer", 
            is_residual=True,
            use_pointwise_layer=True,
            pointwise_ratio=4, 
            pointwise_use_activation=False, 
            attention_mask_style="clipped_causal", 
            attention_heads=num_heads,
            attention_memory_size=(mem_len+timesteps)*self.num_step_tokens,
            n_block=num_layers,
            inject_condition=False, # inject obj_embedding as the condition
        )
        self.lastlayer = FanInInitReLULayer(hiddim, hiddim, layer_type="linear", batch_norm=False, layer_norm=True)
        self.final_ln = nn.LayerNorm(hiddim)
        
        self.aux_vis_head = nn.Linear(hiddim, 1+2+4) # exist, point, bbox
        # self.pre_aux_vis_head = nn.Linear(hiddim, 1+2+4) # exist, point, bbox

        #! view backbone is frozen during training
        for param in self.view_backbone.parameters():
            param.requires_grad = False

    def encode_view_tokens(self, agent_view: torch.Tensor, cross_view: Dict) -> torch.Tensor:
        b, t = agent_view.shape[:2]

        # 1. encode observation with view_backbone
        obs_rgb = rearrange(agent_view, 'b t h w c -> (b t) c h w')
        obs_rgb = self.transforms(obs_rgb)
        x_obs = self.view_backbone(obs_rgb)[-1]
        x_obs = self.updim_obs(x_obs)
        x_obs = rearrange(x_obs, 'b c h w -> b (h w) c')

        # 2. encode cross-view image with view_backbone
        cross_view_rgb = rearrange(cross_view['cross_view_image'], 'b t h w c -> (b t) c h w')
        cross_view_rgb = self.transforms(cross_view_rgb)
        x_cross_image = self.view_backbone(cross_view_rgb)[-1]

        # 3. encode cross-view object mask with mask_backbone
        cross_view_mask = cross_view['cross_view_obj_mask']
        cross_view_mask = rearrange(cross_view_mask, 'b t h w -> (b t) 1 h w') * 1.0
        x_cross_mask = self.mask_backbone(cross_view_mask)[-1]

        # 4. fuse x_cross image and x_cross mask in the feature dimension
        x_cross = torch.cat([x_cross_image, x_cross_mask], dim=1)
        x_cross = self.updim_cross(x_cross)
        x_cross = rearrange(x_cross, 'b c h w -> b (h w) c')

        # 5. fuse x_obs and x_cross in the spatial dimension
        # generate view tokens
        x_cls = self.view_cls_tokens.expand(x_obs.shape[0], -1, -1)
        x_view = torch.cat([x_cls, x_obs, x_cross], dim=1)
        x_view = self.view_resampler(x_view)[:, :self.num_view_tokens, :]
        x_view = rearrange(x_view, "(b t) n c -> b t n c", b=b)

        return x_view

    def temporal_reason(self, x: torch.Tensor, memory: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        b, t = x.shape[:2]
        if not hasattr(self, 'first') or self.first.shape[:2] != (b, t):
            self.first = torch.tensor([[False]], device=x.device).repeat(b, t)
        if memory is None:
            memory = [state.to(x.device) for state in self.recurrent.initial_state(b)]
        z, memory = self.recurrent(x, self.first, memory)
        z = F.relu(z, inplace=False)
        z = self.lastlayer(z)
        z = self.final_ln(z)
        return z, memory

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
    
        b, t = input['image'].shape[:2]

        x_view = self.encode_view_tokens(input['image'], input['cross_view'])
        x = x_view

        # generate interaction tokens
        x_cond = self.interaction(input['cross_view']['cross_view_obj_id'] + 1)
        x_cond = rearrange(x_cond, "b t c -> b t 1 c")
        x = torch.cat([x, x_cond], dim=-2)

        # generate prev_action tokens
        if self.use_prev_action:
            x_prev_a = self.action_embedding_layer(input['env_prev_action'])
            if 'prev_action_dropout' in input:
                dropout_mask = input['prev_action_dropout'][..., None]
                dropout_embedding = repeat(self.dropout_embedding, "1 1 c -> b t c", b=b, t=t)
                x_prev_a = x_prev_a * dropout_mask + dropout_embedding * (1 - dropout_mask)
            
            # dropout_embedding = repeat(self.dropout_embedding, "1 1 c -> b t c", b=b, t=t) #! only for debug
            # x_prev_a = dropout_embedding #! only for debug
            
            x_prev_a = rearrange(x_prev_a, "b t c -> b t 1 c")
            x = torch.cat([x, x_prev_a], dim=-2) 
        
        x = rearrange(x, "b t n c -> b (t n) c", b=b)
        z, memory = self.temporal_reason(x, memory)
        z = rearrange(z, "b (t n) c -> b t n c", t=t)

        aux_vis_logits = self.aux_vis_head(z[:, :, self.index_bias, :])
        exist = aux_vis_logits[:, :, 0:1]
        point = aux_vis_logits[:, :, 1:3]
        bbox = aux_vis_logits[:, :, 3:7]
        
        pi_logits = self.pi_head(z[:, :, -1, :])
        vpred =  self.value_head(z[:, :, -1, :])
        latents = {
            "pi_logits": pi_logits, 
            "vpred": vpred, 
            "exist": exist, 
            "point": point, 
            "bbox": bbox,
        }
        return latents, memory


    def initial_state(self, batch_size: int = None) -> List[torch.Tensor]:
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

def load_cross_view_rocket(ckpt_path: Optional[str] = None):
    ckpt = torch.load(ckpt_path)
    model = CrossViewRocket(**ckpt['hyper_parameters']['model'])
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    return model

if __name__ == '__main__':
    model = CrossViewRocket(
        # view_backbone='timm/vit_base_patch16_224.dino', 
        view_backbone='timm/vit_small_patch16_224.dino', 
        mask_backbone='timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k', 
        hiddim=1024, 
        num_layers=4,
        num_view_tokens=1, 
    ).to("cuda")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Params (MB): {num_params / 1e6 :.2f}")
    
    for key in ["view_backbone", "mask_backbone", "view_resampler", "interaction", "recurrent", "lastlayer", "final_ln"]:
        num_params = sum(p.numel() for p in getattr(model, key).parameters())
        print(f"{key} Params (MB): {num_params / 1e6 :.2f}")

    print("Debug Training mode: True")
    output, memory = model(
        input={
            'image': torch.zeros(1, 64, 224, 224, 3).to("cuda"), 
            'segmentation': {
                'point': torch.zeros(1, 64, 2).to("cuda"),
                'bbox': torch.zeros(1, 64, 4).to("cuda"),
            }, 
            'cross_view': {
                'cross_view_image': torch.zeros(1, 64, 224, 224, 3).to("cuda"),
                'cross_view_obj_id': torch.zeros(1, 64, dtype=torch.long).to("cuda"),
                'cross_view_obj_mask': torch.zeros(1, 64, 224, 224).to("cuda"),
            }
        }
    )
    print("Debug Infering mode: True")
    model.eval()
    output, memory = model(
        input={
            'image': torch.zeros(1, 1, 224, 224, 3).to("cuda"), 
            'cross_view': {
                'cross_view_image': torch.zeros(1, 1, 224, 224, 3).to("cuda"),
                'cross_view_obj_id': torch.zeros(1, 1, dtype=torch.long).to("cuda"),
                'cross_view_obj_mask': torch.zeros(1, 1, 224, 224).to("cuda"),
            }
        }
    )
    print(output.keys())