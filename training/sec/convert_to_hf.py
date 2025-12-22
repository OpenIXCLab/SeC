import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth-model', help='pth model file')
    parser.add_argument(
        '--save-path', type=str, default='./work_dirs/hf_model', help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    model._merge_lora()
    all_state_dict = model.all_state_dict()
    
    name_map = {'mllm.model.': '', 'sam2_model.': '', '.gamma': '.g_weight'}

    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        for _text in name_map.keys():
            new_key = new_key.replace(_text, name_map[_text])
        all_state_dict_new[new_key] = all_state_dict[key]

    # build the hf format model
    from inference.configuration_sec import SeCConfig
    from inference.modeling_sec import SeCModel

    internvl_config = SeCConfig.from_pretrained(cfg.model_path)
    config_dict = internvl_config.to_dict()
    config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sec.SeCConfig',
         'AutoModel': 'modeling_sec.SeCModel',
         'AutoModelForCausalLM': 'modeling_sec.SeCModel'}

    config_dict["llm_config"]["vocab_size"] = len(model.tokenizer)
    config_dict["template"] = cfg.template
    config_dict["grounding_encoder_config"] = cfg.model.grounding_encoder.cfg_path
    config_dict["grounding_maskmem_num"] = 22
    sec_hf_config = SeCConfig(
        **config_dict
    )
    # import pdb; pdb.set_trace()
    hf_sec_model = SeCModel(
        sec_hf_config, vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model,
    )
    hf_sec_model.load_state_dict(all_state_dict_new)

    hf_sec_model.save_pretrained(args.save_path)
    model.tokenizer.save_pretrained(args.save_path)
    print(f"Save the hf model into {args.save_path}")

    # copy the files

if __name__ == '__main__':
    main()