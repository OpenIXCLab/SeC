import logging
import os
import glob
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from iopath.common.file_io import g_pathmgr

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import copy
import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from .encode_fn import video_lisa_encode_fn
from .vos_segment_loader import PalettisedPNGSegmentLoader, MultiplePNGSegmentLoader

SEG_QUESTIONS = [
    "Please segment the object in the last frame based on the object labeled in the first several images.",
    "Using the object labeled in the first several images, please segment the object in the final frame.",
    "Segment the object in the last frame based on the annotations from the first several images.",
    "Please perform object segmentation on the last frame using the labeled object from the previous several frames.",
    "Based on the object marked in the first several images, segment the object in the last frame.",
    "Using the object annotations from the first several images, segment the object in the final frame."
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

# 'conversation', 'input_ids', 'labels', 'pixel_values', 'g_pixel_values', 'masks', 'type'
class SegPNGDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    MODEL_CONFIGS = {
        "internvl": {
            "img_context_token": '<IMG_CONTEXT>',
            "img_start_token": '<img>',
            "img_end_token": '</img>',
        },
        "qwenvl": {
            "img_context_token": '<|image_pad|>',
            "img_start_token": '<|vision_start|>',
            "img_end_token": '<|vision_end|>',
        }
    }

    def __init__(
        self,
        img_folder: str,
        gt_folder: str,
        tokenizer: dict,
        sample_rate: int = 1,
        file_list_txt: str = None,
        excluded_videos_list_txt: str = None,
        is_palette: bool = False,
        single_object_mode: bool = False,
        model_type: str = "internvl",
        extra_image_processor: dict = None,
        template_map_fn: dict = None,
        image_size: int = 448,
        downsample_ratio: float = 0.5,
        patch_size: int = 14,
        max_sam_sampled_frames: int = 16,
        max_sampled_frames: int = 5,
        add_noise: bool = False,
        num_seg_tokens: int = 1,
        select_obj_number: int = 5,
        repeats: int = 1,
        max_length: int = 8196,
        lazy: bool = True,
        special_tokens: list = None,
    ):
        assert lazy, "Currently, only lazy loading is implemented."
        assert tokenizer, "tokenizer is required."
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.add_noise = add_noise

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
        
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []
        
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )
        # self.video_names = ['-klPo9pilc4']
        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        self.max_length = max_length
        self.repeats = repeats
        self.select_obj_number = select_obj_number
        self.max_sampled_frames = max_sampled_frames
        self.max_sam_sampled_frames = max_sam_sampled_frames
        self.image_size = image_size    
        self.num_seg_tokens = num_seg_tokens
        self._system = ''

        config = self.MODEL_CONFIGS[model_type]
        self.IMG_CONTEXT_TOKEN = config['img_context_token']
        self.IMG_START_TOKEN = config['img_start_token']
        self.IMG_END_TOKEN = config['img_end_token']

        if isinstance(template_map_fn, dict):
            _type = template_map_fn['type']
            del template_map_fn['type']
            self.template_map_fn = _type(**template_map_fn)

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        self.patch_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        self.transformer = self._init_image_transformer()
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        # --- For Debugging --- #
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        print(f"Image context dataset initialized with {len(self.video_names)} items.")

    def _init_image_transformer(self) -> T.Compose:
        """封装图像变换器的创建过程"""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.video_names) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.video_names:
            cur_len = 20000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.video_names)

    def dataset_map_fn(self, selected_object_id, n_frames):
        n_objs = len(selected_object_id)
        text_dict = self.prepare_text(n_frames, n_objs, num_image_tokens=self.patch_token)
        ret = {'conversation': text_dict['conversation']}
        return ret

    def prepare_text(self, n_frames, n_objs, num_image_tokens=256):
        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        questions = [random.choice(SEG_QUESTIONS) for _ in range(n_objs)]
        answers = [random.choice(ANSWER_LIST) for _ in range(n_objs)]
        # replace [SEG] with seg tokens
        seg_token_str = '[SEG]' * self.num_seg_tokens
        answers = [ans.replace('[SEG]', seg_token_str) for ans in answers]
        
        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = (frame_token_str + '\n') * n_frames
                qa_list.append({'from': 'human', 'value': frame_tokens.strip() + question})
            else:
                qa_list.append({'from': 'human', 'value': question})
            qa_list.append({'from': 'gpt', 'value': answer})

        conversation = []
        input_text = ''
        for msg in qa_list:
            if msg['from'] == 'human':
                input_text += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input_text, 'output': msg['value']})
                input_text = ''
            else:
                raise NotImplementedError

        # add system information
        conversation[0].update({'system': self._system})
        return {'conversation': conversation}

    def __getitem__(self, index):        
        idx = index % self.real_len()
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(self.img_folder, os.path.dirname(video_name))
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)
        
        video_mask_root = os.path.join(self.gt_folder, video_name)
        
        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(video_mask_root, self.single_object_mode)

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        all_frames = all_frames[:: self.sample_rate]

        def _load_frame_and_masklent(frames, segment_loader, loaded_frames=None):
            video_frames = [] if loaded_frames is None else loaded_frames
            masklents_per_frame, masklents = [], []
            object_ids = set()
            for frame_path in frames:
                if loaded_frames is None:
                    video_frames.append(cv2.imread(frame_path))
                frame_id = int(os.path.splitext(os.path.basename(frame_path))[0])
                binary_segments = segment_loader.load(frame_id)
                masklents_per_frame.append(binary_segments)
                if binary_segments:
                    object_ids.update(binary_segments.keys())
            
            assert object_ids
            rt_object_ids = sorted(list(object_ids))
            mask_h, mask_w = video_frames[0].shape[:2]
            
            for frame_masks_dict in masklents_per_frame:
                masks_for_this_frame = [
                    frame_masks_dict.get(obj_id, np.zeros((mask_h, mask_w), dtype=np.uint8)) 
                    for obj_id in rt_object_ids
                ]
                stacked_masks = np.stack(masks_for_this_frame, axis=-1)
                masklents.append(stacked_masks)
            return video_frames, masklents, rt_object_ids
        
        video_frames, masklents, object_ids = _load_frame_and_masklent(all_frames, segment_loader)
        
        if self.is_palette:
            all_objects_list = object_ids
        else:
            _video_name = os.path.dirname(video_name) if self.single_object_mode else video_name
            video_mask_dirpath = os.path.join(self.gt_folder, _video_name)
            all_objects_list = os.listdir(video_mask_dirpath)
            all_objects_list = sorted([int(x) for x in all_objects_list])

        # Sample main object and prepare frames
        def _prepare_frames(object_ids, masklents, all_objects_list):
            selected_object_id, gt_masklents = self._sample_objects(object_ids, masklents, all_objects_list)
            selected_frame_indexes, selected_g_frame_indexes = self._sample_frames(gt_masklents)
            pixel_values, extra_pixel_values = self._process_frames(video_frames, gt_masklents, selected_frame_indexes, selected_g_frame_indexes)
            
            rt_masklents = [gt_masklents[i] for i in selected_g_frame_indexes]
            rt_masklents = np.stack(rt_masklents, axis=0)                       # (n_frames, h, w, n_obj)
            rt_masklents = torch.from_numpy(rt_masklents).permute(3, 0, 1, 2)   # (n_obj, n_frames, h, w)
            rt_masklents = rt_masklents.flatten(0, 1).to(torch.uint8)           # (n_obj  n_frames, h, w)

            return selected_object_id, pixel_values, extra_pixel_values, rt_masklents
        
        selected_object_id, pixel_values, extra_pixel_values, gt_masklents = _prepare_frames(object_ids, masklents, all_objects_list)

        if self.add_noise:
            # Take some noise frames in pixel_values -- noise object num = 2
            remain_objects = list(set(all_objects_list) - set(selected_object_id))
            if len(remain_objects) > 0:
                num_noise_objects = random.randint(1, min(2, len(remain_objects)))
            else:
                num_noise_objects = 0
            # print(f"num_noise_objects: {num_noise_objects}, length of pixel_values: {len(pixel_values)}")
            if num_noise_objects > 0 and len(pixel_values) > 2:
                noise_object_ids = random.sample(remain_objects, num_noise_objects)
                noise_pixel_values = []
                for noise_object_id in noise_object_ids:
                    if not self.is_palette and self.single_object_mode:
                        _video_mask_root = os.path.join(video_mask_dirpath, f"{noise_object_id:03d}")
                        _segment_loader = MultiplePNGSegmentLoader(_video_mask_root, self.single_object_mode)
                        _, tmp_masklents, tmp_object_ids = _load_frame_and_masklent(all_frames, _segment_loader, loaded_frames=video_frames)
                        _, _noise_pixel_values, _, _ = _prepare_frames(tmp_object_ids, tmp_masklents, tmp_object_ids)
                    else:
                        _, _noise_pixel_values, _, _ = _prepare_frames([noise_object_id], masklents, all_objects_list)
                    noise_pixel_values.extend(_noise_pixel_values[:-1])
                # Replace a random number of frames (0 to less than 50%) with noise_pixel_values
                random.shuffle(noise_pixel_values)
                # print(f"length of noise_pixel_values: {len(noise_pixel_values)}")
                max_replace = max(
                    0, 
                    min((len(pixel_values) - 2) // 2, len(noise_pixel_values))
                )
                num_replace = max_replace if max_replace < 2 else random.randint(1, max_replace)
                replace_indices = sorted(random.sample(range(1, len(pixel_values) - 1), num_replace))

                # print(f"replace indices: {replace_indices}")
                for i, idx in enumerate(replace_indices):
                    pixel_values[idx] = noise_pixel_values[i]

        data_dict = self.dataset_map_fn(selected_object_id, len(pixel_values))
        result = self.template_map_fn(data_dict)
        data_dict.update(result)

        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length, with_image_token=True)

        data_dict.update(result)
        data_dict['pixel_values'] = pixel_values
        if self.extra_image_processor is not None:
            data_dict['g_pixel_values'] = extra_pixel_values
        
        data_dict['masks'] = gt_masklents
        data_dict['type'] = 'video'
        return data_dict

    def visualization_debug(self, data_dict):
        def save_image(image, save_path):
            image = image.permute(1, 2, 0) * 255
            image = image.to(torch.uint8).numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image)

        def save_mask(mask, save_path, base_image, image_size):
            mask_resized = cv2.resize(mask.numpy().astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = mask_resized * 255
            mask_colored = np.stack([mask_resized, np.zeros_like(mask_resized), np.zeros_like(mask_resized)], axis=2)
            combined = base_image * 1 + 0.2 * mask_colored
            combined = np.clip(combined, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, combined.astype(np.uint8))
        
        save_folder = os.path.join(self.save_folder, 'sample_{}'.format(self.cur_number))
        os.makedirs(save_folder, exist_ok=True)
        self.cur_number += 1

        save_folder_image = os.path.join(save_folder, 'image')
        save_folder_mask = os.path.join(save_folder, 'mask')
        os.makedirs(save_folder_image, exist_ok=True)
        os.makedirs(save_folder_mask, exist_ok=True)

        # Images
        pixel_values = data_dict['pixel_values']
        for i_image, image_pixel_value in enumerate(pixel_values):
            image_pixel_value[0] = image_pixel_value[0] * 0.229 + 0.485
            image_pixel_value[1] = image_pixel_value[1] * 0.224 + 0.456
            image_pixel_value[2] = image_pixel_value[2] * 0.225 + 0.406
            save_image(image_pixel_value, os.path.join(save_folder_image, f'{i_image}.jpg'))

        # Text
        input_text = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=False)
        with open(os.path.join(save_folder, 'text.json'), 'w') as f:
            json.dump([input_text], f)

        # Masks
        g_pixel_values = data_dict['g_pixel_values']
        masks = data_dict['masks']
        n_frames = len(g_pixel_values)
        _, h, w = masks.shape
        masks = masks.reshape(-1, n_frames, h, w)

        for i_frame, g_image in enumerate(g_pixel_values):
            g_image = g_image.permute(1, 2, 0)
            g_image = g_image.to(torch.uint8).numpy()
            g_image = cv2.cvtColor(g_image, cv2.COLOR_RGB2BGR)

            for i_obj, obj_masks in enumerate(masks):
                obj_folder = os.path.join(save_folder_mask, f'obj_{i_obj}')
                os.makedirs(obj_folder, exist_ok=True)
                save_mask(obj_masks[i_frame], os.path.join(obj_folder, f'{i_frame}.png'), g_image, 1024)
        
        return
    
    def _process_frames(self, video_frames, masklents, selected_frame_indexes, selected_g_frame_indexes):
        def _preprocess_image(image):
            image = image[:, :, ::-1]
            image = Image.fromarray(image).convert('RGB')
            return image
        
        pixel_values = []
        extra_pixel_values = []

        for i, _idx in enumerate(selected_frame_indexes):
            frame, mask = video_frames[_idx].copy(), masklents[_idx]
            if i < len(selected_frame_indexes) - 1:
                mask = np.uint8(mask)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    frame = cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            frame_image = _preprocess_image(frame)
            frame_image = self.transformer(frame_image)
            pixel_values.append(frame_image)

        for i in selected_g_frame_indexes:
            frame_image = _preprocess_image(video_frames[i].copy())
            if self.extra_image_processor:
                g_image = np.array(frame_image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)
        
        pixel_values = torch.stack(pixel_values, dim=0)
        return pixel_values, extra_pixel_values
        
    def _sample_objects(self, object_ids, masklents, all_objects_list, n_obj=1):
        if not self.is_palette and self.single_object_mode:
            selected_object_ids = [object_ids[0]]
            selected_all_list_indexes = [0]
        else:
            n_objects = len(object_ids)
            selected_indexes = np.random.choice(n_objects, n_obj, replace=n_objects < n_obj)
            selected_object_ids = [object_ids[_idx] for _idx in selected_indexes]
            selected_all_list_indexes = [all_objects_list.index(obj_id) for obj_id in selected_object_ids]

        _masklents = []
        for _mask in masklents:
            _mask_selected = []
            for _idx in selected_all_list_indexes:
                _mask_selected.append(_mask[:, :, int(_idx)])
            _mask_selected = np.stack(_mask_selected, axis=2)
            _masklents.append(_mask_selected)
        rt_masklents = _masklents

        return selected_object_ids, rt_masklents
    
    def _sample_frames(self, masklents):
        n_frames = len(masklents)
        valid_frame_idx = [i for i in range(n_frames) if masklents[i].sum() > 0]

        num_g_memory_frames = np.random.randint(1, self.max_sam_sampled_frames) + 1
        num_memory_frames = np.random.randint(0, self.max_sampled_frames - 1) + 1

        valid_start_frame_idx = [i for i in valid_frame_idx if i < n_frames - num_g_memory_frames]
        if len(valid_start_frame_idx) == 0:
            selected_start_frame = valid_frame_idx[0]
            num_g_memory_frames = n_frames - selected_start_frame
        else:
            selected_start_frame = np.random.choice(valid_start_frame_idx, 1, replace=False)[0]

        # contiguous sample for sam memory, just keep the last frame
        selected_g_frame_indexes = [selected_start_frame + num_g_memory_frames - 1]

        # random sample frames for mllm memory
        ramain_valid_frame_idx = list(set(valid_frame_idx) - set([selected_start_frame, selected_g_frame_indexes[-1]]))
        if len(ramain_valid_frame_idx) <= num_memory_frames:
            selected_frame_indexes = ramain_valid_frame_idx
            num_memory_frames = len(selected_frame_indexes)
        else:
            selected_frame_indexes = np.random.choice(ramain_valid_frame_idx, num_memory_frames, replace=False)

        selected_frame_indexes = list(selected_frame_indexes) + [selected_g_frame_indexes[-1]]
        
        return selected_frame_indexes, selected_g_frame_indexes


if __name__ == '__main__':
    DATA_ROOT = '/Path/to/MOSEv2/train/'
    model_path = '/Path/to/InternVL2_5-4B'
    from transformers import AutoTokenizer
    from xtuner.utils import PROMPT_TEMPLATE
    from xtuner.dataset.map_fns import template_map_fn_factory

    from training.sec.models.preprocess.image_resize import DirectResize

    prompt_template = PROMPT_TEMPLATE.qwen_chat
    max_length = 16384
    special_tokens = ['[SEG]']
    extra_image_processor = dict(
        type=DirectResize,
        target_length=1024,
    )
    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
        padding_side='right'
    )

    img_folder = DATA_ROOT + 'JPEGImages/'
    gt_folder = DATA_ROOT + 'Annotations/'

    video_sam2_dataset = dict(
        img_folder = img_folder,
        gt_folder = gt_folder,
        tokenizer=tokenizer,
        sample_rate=1,
        is_palette=True,
        # single_object_mode=True,
        add_noise=True,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        lazy=True,
        repeats=1,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        max_sampled_frames=8,
        select_obj_number=1,
    )

    # print(video_sam2_dataset)

    dataset = SegPNGDataset(**video_sam2_dataset)

    # print(len(dataset))
    data = dataset[0]
    
    dataset.visualization_debug(dataset[0])
    import pdb; pdb.set_trace()

    