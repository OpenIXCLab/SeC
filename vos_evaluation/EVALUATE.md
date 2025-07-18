## Semi-supervised VOS evaluation

We provide an evaluator to compute the common J and F metrics on SeCVOS. We can evaluate the predictions as follows:

The evaluator expects the `GT_ROOT` to be one of the following folder structures, and `GT_ROOT` and `PRED_ROOT` to have the same structure.

```bash
python vos_evaluation/sav_evaluator.py \
    --gt_root {GT_ROOT} \
    --pred_root {PRED_ROOT} \
    --scene_info /path-to-secvos/scene_info.json \ # required for SeCVOS
    --strict 
```

- Same as SA-V val and test directory structure

```
{GT_ROOT}  # gt root folder
├── {video_id}
│     ├── 000               # all masks associated with obj 000
│     │    ├── 00000.png    # mask for object 000 in frame 00000 (binary mask)
│     │    └── ...
│     ├── 001               # all masks associated with obj 001
│     ├── 002               # all masks associated with obj 002
│     └── ...
├── {video_id}
├── {video_id}
└── ...
```

- Same as [DAVIS](https://github.com/davisvideochallenge/davis2017-evaluation) directory structure

```
{GT_ROOT}  # gt root folder
├── {video_id}
│     ├── 00000.png        # annotations in frame 00000 (may contain multiple objects)
│     └── ...
├── {video_id}
├── {video_id}
└── ...
```

## Reffering VOS evaluation

For Referring VOS settings, we only calculate the J and F scores for the entire video. Unlike Semi-supervised VOS, there’s no need to skip the first and last frames. We can evaluate the predictions as follows:

```bash
python vos_evaluation/sav_evaluator.py \
    --gt_root {GT_ROOT} \
    --pred_root {PRED_ROOT} \
    --strict \
    --do_not_skip_first_and_last_frame
```

## License

The evaluation code is licensed under the [BSD 3 license](./LICENSE). 

Third-party code: the evaluation software is heavily adapted from [`VOS-Benchmark`](https://github.com/hkchengrex/vos-benchmark) and [`DAVIS`](https://github.com/davisvideochallenge/davis2017-evaluation) (with their licenses in [`LICENSE_DAVIS`](./LICENSE_DAVIS) and [`LICENSE_VOS_BENCHMARK`](./LICENSE_VOS_BENCHMARK)).