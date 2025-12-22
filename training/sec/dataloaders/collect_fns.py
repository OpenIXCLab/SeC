from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def video_lisa_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):
    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_pe = any(inst.get('image_grid_thw', None) is not None for inst in instances)
    has_fast_image = any(inst.get('fast_pixel_values', None) is not None for inst in instances)
    has_grounding_image = any(inst.get('g_pixel_values') is not None for inst in instances)
    has_mask = any(inst.get('masks') is not None for inst in instances)
    has_bboxes = any(inst.get('bboxes') is not None for inst in instances)
    has_points = any(inst.get('points') is not None for inst in instances)
    has_fast_exists = any(inst.get('fast_exists') is not None for inst in instances)

    has_vp = any(inst.get('vp_overall_mask') is not None for inst in instances)
    has_prompt_mask = any(inst.get('prompt_masks') is not None for inst in instances)

    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        pixel_values = []
        frames_per_batch = []
        image_grid_thw = []
    if has_grounding_image:
        grounding_pixel_values = []
    if has_mask:
        object_masks = []
    if has_bboxes:
        object_bboxes = []
    if has_points:
        prompt_points = []
    if has_fast_image:
        fast_pixel_values = []
    if has_fast_exists:
        fast_exists = []
    if has_vp:
        vp_overall_mask = []
    else:
        vp_overall_mask = None

    if has_prompt_mask:
        prompt_masks = []
    else:
        prompt_masks = None

    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))

        if has_image:
            pixel_values.append(example['pixel_values'])
            if has_pe:
                image_grid_thw.append(example['image_grid_thw'])
            if has_vp:
                if 'vp_overall_mask' in example.keys() and example['vp_overall_mask'] is not None:
                    vp_overall_mask.append(example['vp_overall_mask'])
                else:
                    vp_overall_mask.append(torch.Tensor([False] * len(pixel_values[-1])))
        if has_fast_image:
            if 'fast_pixel_values' in example.keys() and example['fast_pixel_values'] is not None:
                fast_pixel_values.append(example['fast_pixel_values'])
        if has_fast_exists:
            if 'fast_exists' in example.keys() and example['fast_exists'] is not None:
                fast_exists.append(example['fast_exists'])
        if has_grounding_image and 'g_pixel_values' in example.keys():
            if isinstance(example['g_pixel_values'], list):
                grounding_pixel_values += example['g_pixel_values']
                frames_per_batch.append(len(example['g_pixel_values']))
            else:
                grounding_pixel_values.append(example['g_pixel_values'])
                frames_per_batch.append(1)

        if has_mask:
            if 'masks' in example.keys() and example['masks'] is not None:
                if isinstance(example['masks'], list):
                    if isinstance(example['masks'][0], np.ndarray):
                        _masks = np.stack(example['masks'], axis=0)
                        _masks = torch.from_numpy(_masks)
                        object_masks.append(_masks)
                    else:
                        object_masks.append(torch.stack(example['masks'], dim=0))
                else:
                    object_masks.append(example['masks'])
        if has_bboxes:
            if 'bboxes' in example.keys() and example['bboxes'] is not None:
                object_bboxes.append(example['bboxes'])
        if has_points:
            if 'points' in example.keys() and example['points'] is not None:
                prompt_points.append(example['points'])

        if has_prompt_mask:
            if 'prompt_masks' in example.keys():
                prompt_masks.append(example['prompt_masks'])

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros_like(input_ids).bool()
        for i, length in enumerate(ori_length):
            attention_mask[i, :length] = True

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)

    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }

    if has_image:
        if all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = torch.stack(pixel_values, dim=0)
        data_dict['frames_per_batch'] = frames_per_batch
        data_dict['pixel_values'] = pixel_values
        if has_pe:
            data_dict['image_grid_thw'] = image_grid_thw

    if has_fast_image:
        if all(x.shape == fast_pixel_values[0].shape for x in fast_pixel_values):
            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)
        data_dict['fast_pixel_values'] = fast_pixel_values

    if has_fast_exists:
        data_dict['fast_exists'] = fast_exists

    if has_vp:
        data_dict['vp_overall_mask'] = torch.cat(vp_overall_mask, dim=0)

    if has_prompt_mask:
        data_dict['prompt_masks'] = prompt_masks

    if has_grounding_image:
        # if all(x.shape == grounding_pixel_values[0].shape for x in grounding_pixel_values):
            # grounding_pixel_values = torch.stack(grounding_pixel_values, dim=0)
        data_dict['g_pixel_values'] = grounding_pixel_values

    if has_mask:
        data_dict['masks'] = object_masks

    if has_bboxes:
        data_dict['bboxes'] = object_bboxes

    if has_points:
        data_dict['points'] = prompt_points

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}