# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools.dataset_converters import l_radset_converter as rad
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def radset_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir):
    """Prepare the info file for Custom dataset

    Related data consists of '.pkl' files recording basic infos,
    and groundtruth database.
    
    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
    """

    rad.create_radset_info_file(root_path, info_prefix)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    info_all_path = osp.join(out_dir, f'{info_prefix}_infos_all.pkl')
    update_pkl_infos('radset', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('radset', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('radset', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('radset', out_dir=out_dir, pkl_path=info_test_path)
    update_pkl_infos('radset', out_dir=out_dir, pkl_path=info_all_path)
    create_groundtruth_database(
        'RadSetDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_all.pkl',
        relative_path=False,
        with_mask=False)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', help='name of the dataset')  #metavar='kitti'
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/m2dset',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=5,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/m2dset',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='m2dset2')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    default=False,
    action='store_true',
    help='Whether to only generate ground truth database.')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()
    if args.dataset == 'radset':
        if args.only_gt_database:
            create_groundtruth_database(
                'MRDSetDataset',
                args.root_path,
                args.extra_tag,
                f'{args.extra_tag}_infos_train.pkl',
                relative_path=False)
        else:
            radset_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,
                out_dir=args.out_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
