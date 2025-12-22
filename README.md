# SeC: Advancing Complex Video Object Segmentation via Progressive Concept Construction

🚀🚀🚀 Official implementation of **SeC: Advancing Complex Video Object Segmentation via Progressive Concept Construction**

<p align="left" style="font-size: 1 em; margin-top: -1em">
    <a href="https://github.com/rookiexiong7/">Zhixiong Zhang</a><sup>*</sup> ·
    <a href="https://mark12ding.github.io/">Shuangrui Ding</a><sup>*</sup> ·
    <a href="https://lightdxy.github.io/">Xiaoyi Dong</a><sup>✉</sup> ·
    <a href="https://github.com/OpenIXCLab/SeC">Songxin He</a> ·
    <a href="https://github.com/OpenIXCLab/SeC">Jianfan Lin</a> ·
    <a href="https://github.com/OpenIXCLab/SeC">Junsong Tang</a><br/>
    <a href="https://yuhangzang.github.io/">Yuhang Zang</a> ·
    <a href="https://scholar.google.com/citations?user=sJkqsqkAAAAJ">Yuhang Cao</a> ·
    <a href="http://dahua.site/">Dahua Lin</a> ·
    <a href="https://myownskyw7.github.io/">Jiaqi Wang</a><sup>✉</sup>
</p>

<p align="center" style="font-size: 5 em; margin-top: 0.5em">
    <a href="https://arxiv.org/abs/2507.15852" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&style=for-the-badge">
    </a>
    <a href="https://rookiexiong7.github.io/projects/SeC" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Demo-GitHub%20Pages-2f2f2f?logo=github&style=for-the-badge">
    </a>
    <!-- <a href="https://huggingface.co/datasets/your-dataset-name" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Demo-HuggingFace-orange?logo=huggingface&style=for-the-badge">
    </a> -->
    <a href="https://huggingface.co/OpenIXCLab/SeC-4B" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Model-HuggingFace-orange?logo=huggingface&style=for-the-badge">
    </a>
    <a href="https://huggingface.co/datasets/OpenIXCLab/SeCVOS" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface&style=for-the-badge">
    </a>

</p>

## Demo Video
https://github.com/user-attachments/assets/40fbf928-5722-45e1-adae-adb70c1251f7

## 📜 News

🚀 [2025/12/22] Training code has been fully open-sourced.

🚀 [2025/10/14] SeC is cited by [SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) and used as a baseline!

🏆 [2025/8/13] SeC sets a new state-of-the-art on the latest MOSE v2 [leaderboard](https://www.codabench.org/competitions/10062/#/results-tab)! 

🚀 [2025/7/22] The [Paper](https://arxiv.org/abs/2507.15852) and [Project Page](https://rookiexiong7.github.io/projects/SeC/) are released!

## 💡 Highlights

- 🔥We introduce **Segment Concept (SeC)**, a **concept-driven** segmentation framework for **video object segmentation** that integrates **Large Vision-Language Models (LVLMs)** for robust, object-centric representations.
- 🔥SeC dynamically balances **semantic reasoning** with **feature matching**, adaptively adjusting computational efforts based on **scene complexity** for optimal segmentation performance.
- 🔥We propose the **Semantic Complex Scenarios Video Object Segmentation (SeCVOS)** benchmark, designed to evaluate segmentation in challenging scenarios.
<!-- where SeC achieves an **11.8-point improvement** over **SAM 2.1**, setting a new state-of-the-art in concept-aware VOS. -->

## ✨ SeC Performance
| Model | SA-V val | SA-V test | LVOS v2 val | MOSE v1 | MOSE v2 |DAVIS 2017 val | YTVOS 2019 val | SeCVOS |
| :---- | :------: | :-------: | :---------: | :-----: | :-----: | :-----------: | :------------: | :----: |
| SAM 2.1 | 78.6 | 79.6 | 84.1 | 74.5 | 49.5 | 90.6 | 88.7 | 58.2 |
| SAMURAI | 79.8 | 80.0 | 84.2 | 72.6 | 51.1 | 89.9 | 88.3 | 62.2 |
| SAM2.1Long | 81.1 | 81.2 | 85.9 | 75.2 | 51.5 | 91.4 | 88.7 | 62.3 |
| **SeC (Ours)** | **82.7**  | **81.7** | **86.5**  | **75.3**  | **53.8** | **91.3** | **88.6** | **70.0** |

## 👨‍💻 TODO

- [x] Release SeC training code
- [x] Release SeCVOS benchmark annotations
- [x] Release SeC inference code and checkpoints

## 🛠️ Usage

### 1. Install environment and dependencies
Please make sure using the correct versions of transformers and peft.
```bash
conda create -n sec python=3.10
conda activate sec
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Download the Pretrained Checkpoints

Download the SeC checkpoint from [🤗HuggingFace](https://huggingface.co/OpenIXCLab/SeC-4B) and place it in the following directory : 
```
saved_models
  ├── SeC-4B
  │   └── config.json
  │   └── generation_config.json
  ...
```

### 3. Quick Start
If you want to test **SeC** inference on a single video, please refer to `demo.ipynb`.

### 4. Run the inference and evaluate the results
The inference instruction is in [INFERENCE.md](vos_evaluation/INFERENCE.md). 

The evaluation instruction can be found in [EVALUATE.md](vos_evaluation/EVALUATE.md). To evaluate performance on seen and unseen categories in the LVOS dataset, refer to the evaluation code available [here](https://github.com/LingyiHongfd/lvos-evaluation).

### 5. Training SeC
Training instructions can be found in [TRAIN.md](training/TRAIN.md).

To train SeC on your own dataset, simply organize your data following the format described there.

## ❤️ Acknowledgments and License
This repository are licensed under a [Apache License 2.0](LICENSE). 

This repo benefits from [SAM 2](https://github.com/facebookresearch/sam2), [SAM2Long](https://github.com/Mark12Ding/SAM2Long) and [Sa2VA](https://github.com/magic-research/Sa2VA). Thanks for their wonderful works.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@article{zhang2025sec,
    title     = {SeC: Advancing Complex Video Object Segmentation via Progressive Concept Construction},
    author    = {Zhixiong Zhang and Shuangrui Ding and Xiaoyi Dong and Songxin He and Jianfan Lin and Junsong Tang and Yuhang Zang and Yuhang Cao and Dahua Lin and Jiaqi Wang},
    journal   = {arXiv preprint arXiv:2507.15852},
    year      = {2025}
}
```
