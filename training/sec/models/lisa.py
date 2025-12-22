
import torch
import torch.nn as nn

from mmengine.model import BaseModel

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict


class LisaModel(BaseModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,):
        super(LisaModel, self).__init__()
        self.mllm = BUILDER.build(mllm)

        if self.mllm.use_llm_lora:
            self.mllm.model.language_model.base_model.model.lm_head.requires_grad_(True)
            self.mllm.model.language_model.base_model.model.model.embed_tokens.requires_grad_(True)

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        self.grounding_encoder.mask_decoder.requires_grad_(True)
        
        in_dim = self.mllm.model.config.llm_config.hidden_size
        out_dim = self.grounding_encoder.mask_decoder.transformer_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        
    def _add_special_tokens(self):
        special_tokens = ['[SEG]']
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))

        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list=None, orig_size_list=None):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            
            pred_mask = self.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size_list[i], )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks
    
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)
    
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict))
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'grounding_encoder.mask_decoder.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k})
        return to_return
    
    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data)
        elif mode == 'predict':
            return self.predict(data)
        elif mode == 'tensor':
            return self._forward(data)
        else:
            raise NotImplementedError
        
    def compute_loss(self,data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            g_pixel_values = [
                torch.randn(3, 512, 1024).to(output.hidden_states[-1]) 
                    for _ in range(len(input_ids))]
            ori_size_list = [(512, 1024) for _ in range(len(input_ids))]
            seg_token_mask = torch.zeros_like(input_ids).bool()
            seg_token_mask[:, -2] = True
        else:
            ori_size_list = [mask.shape[-2:] for mask in gt_masks]
            seg_token_mask = input_ids == self.seg_token_idx

        resize_list = [pixel.shape[-2:] for pixel in g_pixel_values]
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess(pixel) for pixel in g_pixel_values
        ])
        image_embeddings = self.grounding_encoder.image_encoder(g_pixel_values)

        seg_token_mask = seg_token_mask[:, 1:]
        seg_token_mask = torch.cat([
            seg_token_mask,
            seg_token_mask.new_zeros(seg_token_mask.shape[0], 1)], dim=-1)

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        pred_embeddings = hidden_states[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        
        pred_masks = self._generate_and_postprocess_masks(
            pred_embeddings_list, image_embeddings, resize_list, ori_size_list)
        
        if gt_masks is None:
            return {
                'loss_mask': pred_masks[0].sum() * 0.0,
                'loss_dice': pred_masks[0].sum() * 0.0,
                'llm_loss': output.loss,
            }
        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        for i in range(bs):
            pred_mask = pred_masks[i]
            gt_mask = gt_masks[i]

            sam_loss_mask = self.loss_mask(pred_mask, gt_mask)
            sam_loss_dice = self.loss_dice(pred_mask, gt_mask)
            accuracy = torch.eq((pred_mask.sigmoid() > 0.5), gt_mask).to(pred_mask).mean()
            loss_mask += sam_loss_mask
            loss_dice += sam_loss_dice

        loss_dict = {
            'loss_mask': loss_mask / bs,
            'loss_dice': loss_dice / bs,
            'llm_loss': output.loss,
        }
        return loss_dict

    def predict(self, data):
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|end|>')
        generation_config['eos_token_id'] = eos_token_id
        pixel_values = data.pop('pixel_values')
        attention_mask = data.pop('attention_mask', None)
        input_ids = data['input_ids']
        generate_output = self.mllm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generation_config,
        )
        device = self.mllm.model.device

        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1] for item in hidden_states[1:]] # remove input_ids
        last_hidden_states = torch.cat(last_hidden_states, dim=1)
        last_hidden_states = last_hidden_states[0] # remove batch dim
        output_ids = generate_output.sequences[0][:-1] # remove batch dim and eos token
        output_text = self.tokenizer.decode(output_ids)
        seg_mask = output_ids == self.seg_token_idx
        if seg_mask.sum() == 0:
            return dict(
                pred_mask_logits=None,
                output_text=output_text,
            )
        seg_embeds = self.text_hidden_fcs(last_hidden_states[seg_mask])
        
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data['masks']
        
        ori_size_list = [mask.shape[-2:] for mask in gt_masks]
        resize_list = [pixel.shape[-2:] for pixel in g_pixel_values]
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess(pixel.to(device)) for pixel in g_pixel_values
        ])
        image_embeddings = self.grounding_encoder.image_encoder(g_pixel_values)
        pred_masks = self._generate_and_postprocess_masks(
            [seg_embeds], image_embeddings, resize_list, ori_size_list)
        
        return dict(
            pred_mask_logits=pred_masks[0], # remove batch dim
            output_text=output_text,
        )

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.mllm.model.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.mllm.model.language_model.gradient_checkpointing_disable()
