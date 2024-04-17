# Copyright (c) OpenMMLab. All rights reserved.
from concurrent import futures as futures
from os import path as osp
import os
from pathlib import Path

import numpy as np
from skimage import io
from math import sqrt


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_radset_info_path(idx,
                        prefix,
                        info_type='images',
                        file_tail='.jpg',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path(info_type) / img_idx_str
    else:
        file_path = Path(info_type) / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='images',
                   file_tail='.jpg',
                   use_prefix_id=False):
    return get_radset_info_path(idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='labels',
                   use_prefix_id=False):
    return get_radset_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_radset_info_path(idx, prefix, 'radar', '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_radset_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_timestamp_path(idx,
                       prefix,
                       training=True,
                       relative_path=True,
                       exist_check=True,
                       use_prefix_id=False):
    return get_radset_info_path(idx, prefix, 'timestamp', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions is standard lwh format.
    annotations['location'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)
    annotations['dimensions'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    annotations = annotation_filter(annotations)
    return annotations


def annotation_filter(annotations):
    tan_data = annotations['location'][..., 1] / (annotations['location'][..., 0]+10e-5)
    valid_point_idx = (tan_data >= -1.73)
    valid_point_idx &=  (tan_data <= 1.73)
    valid_point_idx &= (annotations['location'][..., 0] >= 0)
    for key in annotations.keys():
        annotations[key] = annotations[key][valid_point_idx]
    
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_radset_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         timestamp=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    radset annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for radset]image: {
            image_0_idx: ...
            image_1_idx: ...
            ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            radareatures: 4
            velodyne_path: ...
        }
        [optional, for radset]calib: {
            R0_rect: ...
            Tr_velo_to_cam_0: ...
            Tr_velo_to_cam_1: ...
            ...
            P0: ...
            P1: ...
            ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: radset difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 8}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        if timestamp:
            timestamp_path = get_timestamp_path(
                idx, path, training, relative_path=False)
            with open(timestamp_path, 'r') as f:
                info['timestamp'] = np.int64(f.read())        
        image_info['image_path'] = []
        img_names = os.listdir(path + '/images')
        img_names.sort(key=lambda x:int(x[-1]))
        for info_type in img_names:
            info_type='images/' + info_type
            image_info['image_path'].append(get_image_path(idx, path, training, relative_path, info_type=info_type))
        if with_imageshape:
            img_path = image_info['image_path'][0]
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            P5 = np.array([float(info) for info in lines[5].split(' ')[1:13]
                           ]).reshape([3, 4])
            P6 = np.array([float(info) for info in lines[6].split(' ')[1:13]
                           ]).reshape([3, 4])
            P7 = np.array([float(info) for info in lines[7].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
                P5 = _extend_matrix(P5)
                P6 = _extend_matrix(P6)
                P7 = _extend_matrix(P7)
            R0_rect = np.array([
                float(info) for info in lines[8].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam_0 = np.array([
                float(info) for info in lines[9].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam_1 = np.array([
                float(info) for info in lines[10].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam_2 = np.array([
                float(info) for info in lines[11].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam_3 = np.array([
                float(info) for info in lines[12].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam_4 = np.array([
                float(info) for info in lines[13].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam_5 = np.array([
                float(info) for info in lines[14].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam_6 = np.array([
                float(info) for info in lines[15].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam_7 = np.array([
                float(info) for info in lines[16].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[17].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam_0 = _extend_matrix(Tr_velo_to_cam_0)
                Tr_velo_to_cam_1 = _extend_matrix(Tr_velo_to_cam_1)
                Tr_velo_to_cam_2 = _extend_matrix(Tr_velo_to_cam_2)
                Tr_velo_to_cam_3 = _extend_matrix(Tr_velo_to_cam_3)
                Tr_velo_to_cam_4 = _extend_matrix(Tr_velo_to_cam_4)
                Tr_velo_to_cam_5 = _extend_matrix(Tr_velo_to_cam_5)
                Tr_velo_to_cam_6 = _extend_matrix(Tr_velo_to_cam_6)
                Tr_velo_to_cam_7 = _extend_matrix(Tr_velo_to_cam_7)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['P5'] = P5
            calib_info['P6'] = P6
            calib_info['P7'] = P7
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam_0'] = Tr_velo_to_cam_0
            calib_info['Tr_velo_to_cam_1'] = Tr_velo_to_cam_1
            calib_info['Tr_velo_to_cam_2'] = Tr_velo_to_cam_2
            calib_info['Tr_velo_to_cam_3'] = Tr_velo_to_cam_3
            calib_info['Tr_velo_to_cam_4'] = Tr_velo_to_cam_4
            calib_info['Tr_velo_to_cam_5'] = Tr_velo_to_cam_5
            calib_info['Tr_velo_to_cam_6'] = Tr_velo_to_cam_6
            calib_info['Tr_velo_to_cam_7'] = Tr_velo_to_cam_7
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info['calib'] = calib_info

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


def add_difficulty_to_annos(info):
    max_dist = [25, 50, 100]
    min_scale = [0.44, 0.22, 0.11]  
    annos = info['annos']
    dims = annos['dimensions']  # lwh format 
    loc = annos['location']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=bool)
    moderate_mask = np.ones((len(dims), ), dtype=bool)
    hard_mask = np.ones((len(dims), ), dtype=bool)
    i = 0
    for d, l in zip(dims, loc):
        dist = int(sqrt(l[0]**2 + l[1]**2))
        # sd_factor = (dims[0] * dims[1]) / dist
        if dist > max_dist[0]:
            easy_mask[i] = False
        if dist > max_dist[1]:
            moderate_mask[i] = False
        if dist > max_dist[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff
