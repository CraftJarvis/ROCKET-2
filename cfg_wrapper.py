'''
Date: 2025-02-23 22:18:55
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-03-19 13:36:18
FilePath: /ROCKET-2/cfg_wrapper.py
'''

import numpy as np
import torch
import typing
from typing import Dict, List, Optional, Tuple, Any, Union
from minestudio.models.base_policy import recursive_tensor_op, dict_map


class CFGWrapper:
    
    def __init__(self, model, k: float=1.0):
        self.model = model
        self.k = k
    
    def initial_state(self):
        return (self.model.initial_state(), self.model.initial_state())
    
    @torch.inference_mode()
    def get_action(self,
                   input: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
                   **kwargs, 
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        if state_in is None:
            state_in = (None, None)
        cond_state_in, base_state_in = state_in
        cond_input = input.copy()
        cond_latents, cond_state_out = self.get_action_once(cond_input, cond_state_in, deterministic, input_shape, **kwargs)
        self.cache_latents = dict_map(lambda tensor: tensor[0][0], cond_latents)

        base_input = input.copy()
        base_input['cross_view']['cross_view_image'] = np.zeros_like(base_input['cross_view']['cross_view_image'])
        base_input['cross_view']['cross_view_obj_id'] = torch.zeros_like(base_input['cross_view']['cross_view_obj_id']) - 1
        base_input['cross_view']['cross_view_obj_mask'] = torch.zeros_like(base_input['cross_view']['cross_view_obj_mask'])
        base_latents, base_state_out = self.get_action_once(base_input, base_state_in, deterministic, input_shape, **kwargs)
        # import ipdb; ipdb.set_trace()
        pi_logits = {}
        pi_logits["buttons"] = (1+self.k) * cond_latents['pi_logits']["buttons"] - self.k*base_latents['pi_logits']["buttons"]
        pi_logits["camera"] = (1+self.k) * cond_latents['pi_logits']["camera"] - self.k*base_latents['pi_logits']["camera"]


        action = self.model.pi_head.sample(pi_logits, deterministic)
        
        state_out = (recursive_tensor_op(lambda x: x[0], cond_state_out), 
                    recursive_tensor_op(lambda x: x[0], base_state_out))
        return dict_map(lambda tensor: tensor[0][0], action), state_out
    
    @torch.inference_mode()
    def get_action_once(self,
                   input: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
                   **kwargs, 
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        assert input_shape == '*'
        # import ipdb; ipdb.set_trace()
        input = dict_map(self.model._batchify, input)
        if state_in is not None:
            state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
        latents, state_out = self.model.forward(input, state_in, **kwargs)
        return latents, state_out