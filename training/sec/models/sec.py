from typing import Literal, Tuple, Type

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from .lisa import LisaModel

from xtuner.model.utils import guess_load_checkpoint
import copy

from training.sam2.sam2.modeling.sam.transformer import Attention
from training.sam2.sam2.modeling.sam2_utils import MLP

class SeCModel(LisaModel):
    def __init__(
        self,
        mllm,
        tokenizer,
        grounding_encoder,
        loss_fns,
        pretrained_pth=None,
        special_tokens=None,
        # for arch selection
        arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
        # for grad freeze
        frozen_sam2_decoder=True,
        frozen_token_attn=False,
    ):
        super(LisaModel, self).__init__()
        self.arch_type = arch_type
        self.mllm = BUILDER.build(mllm)

        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl' or self.arch_type == 'llava':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type == 'qwen':
                self.mllm.model.model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.get_output_embeddings().weight.requires_grad_(True)
        
        self.mllm.model.language_model.print_trainable_parameters()

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(special_tokens)
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)

        # Add new modules
        self.grounding_encoder.sam2_model.token_attn = copy.deepcopy(self.grounding_encoder.sam2_model.memory_attention)

        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
        if not frozen_token_attn:
            self.grounding_encoder.sam2_model.token_attn.requires_grad_(True)
        
        if self.arch_type == 'intern_vl':
            _in_dim = self.mllm.model.config.llm_config.hidden_size
        elif self.arch_type == 'qwen':
            _in_dim = self.mllm.model.config.hidden_size
        elif self.arch_type == 'llava':
            _in_dim = self.mllm.model.language_model.config.hidden_size

        in_dim = _in_dim
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        # import pdb; pdb.set_trace()
        self.loss_fns = BUILDER.build(loss_fns)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

    
    def _merge_lora(self):
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")

    def activation_checkpointing_disable(self):
        if self.arch_type == 'qwen':
            self.mllm.model.model.gradient_checkpointing_disable()
        else:
            self.mllm.model.language_model.gradient_checkpointing_disable()

    def _add_special_tokens(self, special_tokens):
        if special_tokens is None: 
            special_tokens = ['[SEG]']
        
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))
        
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def all_state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl' or self.arch_type == 'llava':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
            elif self.arch_type == 'qwen':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
                )
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
            raise NotImplementedError
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})
        # Step 4. mask decoder of grounding model (SAM/SAM2)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mask_decoder' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'token_attn' in k})
        # Step 5. others (fcs)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k or 'output' in k and 'sam2_model' not in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k or 'tok_embeddings' in k})
        return to_return

class SeCModel_zero3(SeCModel):
    def __init__(
        self,
        mllm,
        tokenizer,
        grounding_encoder,
        loss_fns,
        pretrained_pth=None,
        special_tokens=['[SEG]'],
        arch_type='intern_vl',
        num_seg_tokens=1,
        # zero3
        frozen_sam2_decoder=True,
        frozen_token_attn=True,
        bs=1,
    ):
        super(SeCModel_zero3, self).__init__(
            mllm=mllm,
            tokenizer=tokenizer,
            grounding_encoder=grounding_encoder,
            loss_fns=loss_fns,
            pretrained_pth=pretrained_pth,
            special_tokens=special_tokens,
            arch_type=arch_type,
            frozen_sam2_decoder=frozen_sam2_decoder,
            frozen_token_attn=frozen_token_attn,
        )
        self.bs = bs
        self.num_seg_tokens = num_seg_tokens

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((1, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        
        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True
        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"

        # ------ MLLM Part ------
        output = self.mllm(data, data_samples, mode)
        
        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :1].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 1

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        # -----------------------
        
        # ------ SAM2 Part ------
        # num_objs = pred_embeddings_list[0].shape[0]
        num_objs = pred_embeddings_list[0].shape[0] // self.num_seg_tokens
        language_embeddings = torch.cat(pred_embeddings_list, dim=0)[:, None]
        
        # for multi seg token for one object
        language_embeddings = language_embeddings.reshape(self.bs, -1, language_embeddings.shape[-1])
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])                                                                          # (F, C, H, W)
        
        # get image embeddings
        tgt_states = self.grounding_encoder.get_sam2_embeddings(
            g_pixel_values, frames_per_batch, expand_size=num_objs, choices='tgt'
        )
        
        # divide into batch
        gt_masks = torch.cat([
            F.interpolate(gt_mask.unsqueeze(0), size=g_pixel_values.shape[-2:], mode='nearest').squeeze(0) 
            for gt_mask in gt_masks
        ], dim=0).unsqueeze(1)                                                      # (F, 1, H, W)
        split_gt_masks = torch.split(gt_masks, frames_per_batch, dim=0)
        tgt_gt_masks = torch.cat([item[-1:] for item in split_gt_masks], dim=0)     # (B, 1, H, W)
        
        # get language conditioned features
        pix_feats_with_language = self.grounding_encoder.prepare_sam2_language_conditioned_features(
            language_embeddings, tgt_states, frames_per_batch
        )
        
        sam_outputs = self.grounding_encoder.forward_sam_heads(tgt_states, pix_feats_with_language)
        _, high_res_multimasks, ious, _, _, _, object_score_logits = sam_outputs
        # -----------------------

        # ------ Loss Part ------
        current_output = {
            "multistep_pred_multimasks_high_res": [high_res_multimasks],
            "multistep_pred_ious": [ious],
            "multistep_object_score_logits": [object_score_logits],
        }
        outs_batch = [current_output]                     # List[Dict]
        targets_batch = tgt_gt_masks.permute(1, 0, 2, 3)  # torch.Tensor [1, B, H, W]
        loss_dict = self.loss_fns(outs_batch, targets_batch)

        loss_dict["llm_loss"] = output.loss
        # -----------------------
        return loss_dict
