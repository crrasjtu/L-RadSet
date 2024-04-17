# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('--config', help='config file path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    args.config = 'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')


if __name__ == '__main__':
    main()
