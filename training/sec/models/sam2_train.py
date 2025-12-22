import os.path

import torch
import random
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mmengine.model import BaseModule

from vlm.utils import load_checkpoint_with_prefix, load_state_dict_to_model
import copy

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed

class SAM2TrainRunner(BaseModule):
    def __init__(
        self,
        cfg_path: str = "sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path: str = "sam2.1_hiera_large.pt",
        hydra_overrides_extra=None,
        apply_postprocessing=True,
    ):
        super().__init__(init_cfg=None)

        import training.sam2 # noqa: F401

        if hydra_overrides_extra is None:
            hydra_overrides_extra = []
        hydra_overrides = [
            ## Extension: LLM prompt
            "++model._target_=training.sam2.sam2.modeling.sam2_base.SAM2Base",
        ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                # "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                # "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)

        # Read config and init model
        cfg = compose(config_name=cfg_path, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        sam2_model = instantiate(cfg.model, _recursive_=True)
        state_dict = load_checkpoint_with_prefix(os.path.join(PROJECT_DIR, ckpt_path))
        load_state_dict_to_model(sam2_model, state_dict)

        self.sam2_model = sam2_model

        self.hidden_dim = self.sam2_model.hidden_dim
        self.num_maskmem = self.sam2_model.num_maskmem
        self.max_obj_ptrs_in_encoder = self.sam2_model.max_obj_ptrs_in_encoder
        self.mem_dim = self.sam2_model.mem_dim

        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std
        return image

    def forward_sam_heads(self, sam_states, pix_feat_with_mem):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sam_outputs = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=None),
            )

        return sam_outputs
        
    def prepare_sam2_language_conditioned_features(
        self, language_embeddings, tgt_states, frames_per_batch
    ):
        """Fuse the current frame's visual feature map with language embeddings."""
        B = len(frames_per_batch)
        C = self.hidden_dim
        H, W = tgt_states['feat_sizes'][-1]
        
        current_vision_feats = tgt_states["current_vision_feats"][-1]               # [H * W, B, C]
        current_vision_pos_embeds = tgt_states["current_vision_pos_embeds"][-1]     # [H * W, B, C]
        if self.mem_dim < C:                                                        # [B, N, C] -> [N, B, C // mem_dim, mem_dim]
            language_embd = language_embeddings.permute(1, 0, 2)
            _language_embd = language_embd.reshape(                
                -1, B, C // self.mem_dim, self.mem_dim
            )
            _language_embd = _language_embd.permute(0, 2, 1, 3).flatten(0, 1)       # [L, B, C]
            _language_embd_pos = _language_embd.new_zeros(_language_embd.size(0), 1, self.mem_dim)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pix_feat_with_language = self.sam2_model.token_attn(
                curr=current_vision_feats,
                curr_pos=current_vision_pos_embeds,
                memory=_language_embd,
                memory_pos=_language_embd_pos,
                num_obj_ptr_tokens=_language_embd.shape[0],
            )
        pix_feat_with_language = pix_feat_with_language.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_language
            
               
    def get_sam2_embeddings(self, images, frames_per_batch, expand_size=1, choices='tgt'):
        assert choices in ['tgt', 'mem']
        def _expand(feats, expand_size):
            if expand_size > 1:
                # feats['vision_features'] = feats['vision_features'][:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                for i, feat in enumerate(feats["backbone_fpn"]):
                    feats["backbone_fpn"][i] = feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                for i, pos in enumerate(feats["vision_pos_enc"]):
                    pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                    feats["vision_pos_enc"][i] = pos
            return feats
        
        images_split = torch.split(images, frames_per_batch, dim=0)
        tgt_images = torch.cat([x[-1:] for x in images_split], dim=0)
        mem_images = torch.cat([x[:-1] for x in images_split], dim=0)

        rt_images = tgt_images if choices == 'tgt' else mem_images
        # Step 1: inference the backbone with the images
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rt_feats = self.sam2_model.forward_image(rt_images)
        rt_feats = _expand(rt_feats, expand_size)

        # Step 2: Process the features to output
        _, rt_vision_feats, rt_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(rt_feats)

        rt_states = {
            "current_vision_feats": rt_vision_feats,
            "current_vision_pos_embeds": rt_vision_pos_embeds,
            "feat_sizes": feat_sizes
        }

        return rt_states

    def forward(self, batch):
        raise NotImplementedError
