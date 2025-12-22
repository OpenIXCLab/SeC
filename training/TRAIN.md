# Training Code for SeC
This folder contains the training code for SeC, which is trained in two stages.

## Dataset and Model Preparation

### 1. Dataset Preparation
Download the [SA-V](https://ai.meta.com/datasets/segment-anything-video/) dataset and organize it in the following format. Before training, please extract video frames first following the script [here](https://github.com/facebookresearch/sam2/blob/main/training/scripts/sav_frame_extraction_submitit.py)

```
dataset
├── SA-V
│   ├── metavideo                   # the original SA-V dataset
│   │   ├── {video_id}_manual.json  # video annotations
│   │   ├── {video_id}.mp4
│   │   └── ...
│   ├── train                       # extracted video frames for training
│   │   ├── {video_id}
│   │   │   ├── 00000.jpg           # video frame
│   │   │   ├── 00004.jpg           # video frame
│   │   │   └── ...
│   │   ├── {video_id}
│   │   └── ...
├── sav_scene_top2k.txt
└── sav_video_obj_ids.txt
```

### 2. Pretrained Model Preparation
Download the [InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) and [SAM2.1-Hiera-L](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt), and organize them in the following format.

```
pretrained_models
├── InternVL2_5-4B
├── sam2.1_hiera_large.pt
└── reshape.py
```

Then, please run the `reshape.py` script to convert the weights of the SAM model into the format suitable for SeC:

## Training
### Stage 1: Enhanced Pixel-level Association Module
Fine-tune the SAM-based model with memory enhancement:
```bash
export PYTHONPATH=.
python training/sam2/training/train.py \
    -c "sec_sam2.1_hiera_l_finetune_maskmem22.yaml" \
    --use-cluster 0 \
    --num-gpus 8
```

Checkpoints and logs will be saved to `work_dirs/sec_sam2/sec_sam2.1_hiera_l_finetune_maskmem22.yaml/`.

### Stage 2: Concept Guidance Module
Train the Concept Guidance Module using distributed training. Please replace the `model_path` and `sam2_path` in `training/sec/configs/sec-4b.py` with the paths to the pretrained InternVL2_5-4B model and the fine-tuned SAM model from Stage 1, respectively.

```bash
export PYTHONPATH=.
bash tools/dist.sh train "training/sec/configs/sec_4b.py" 8
```

Checkpoints and logs will be saved to `work_dirs/sec_4b/`.