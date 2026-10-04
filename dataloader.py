from PIL import Image
import csv
import os
import sys
import random
import pandas as pd
import cv2
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from configuration import build_config
from torch.autograd.variable import Variable
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize

)
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} time consuming: {end_time - start_time:.2f}S")
        return result
    return wrapper

class CrossViewDataLoader(Dataset):
    def __init__(self, cfg, data_split, cross_view_type='CS', transform=None):
        self.cross_view_type = cross_view_type
        self.dataset = cfg.dataset
        self.data_split = data_split
        self.videos_folder = cfg.videos_folder
        self.transform = transform
        self.num_frames = 16
        self.height = 224
        self.width = 224

        if data_split == "train":
            self.annotations = cfg.train_annotations
        else:
            self.annotations = cfg.test_annotations

        print(f"Starting to build the {self.cross_view_type} grouping index for the {self.dataset} dataset...")
        self.build_cross_view_indices()
        print(f"Sample grouping completed, with a total of {len(self.cross_view_groups)} groups.")
        print(f"{get_cache_info()}")
    
    def __del__(self):
        clear_hdf5_cache()

    @timing_decorator
    def build_cross_view_indices(self):
        df = pd.read_csv(self.annotations)

        if self.dataset == 'numa':
            df['viewpoint'] = df['viewpoint'].astype(str).str.strip().str.replace('\n', '')
            df['viewpoint'] = pd.to_numeric(df['viewpoint'], errors='coerce')
            df['trial'] = df['video'].str.extract(r'e(\d+)').astype(int)
            self._build_numa_groups(df)

        elif self.dataset in ['ntu_rgbd_60', 'ntu_rgbd_120']:
            self._build_ntu_groups(df)
        
        elif self.dataset == 'pkummd':
            self._build_pkummd_groups(df)

    def _build_numa_groups(self, df):
        self.cross_view_groups = []

        if self.cross_view_type == 'CS':
            grouped = df.groupby(['subject', 'action', 'trial'])
            
            for (subject, action, trial), group in grouped:
                viewpoints = group['viewpoint'].dropna().unique()

                if len(viewpoints) >= 1: 
                    selected_viewpoints = sorted(viewpoints)
                    group_samples = []

                    for viewpoint in selected_viewpoints:
                        sample = group[group['viewpoint'] == viewpoint].iloc[0]
                        group_samples.append({
                            'video_id': sample['video'],
                            'subject': sample['subject'],
                            'action': sample['action'],
                            'viewpoint': sample['viewpoint'],
                            'trial': sample['trial']
                        })

                    self.cross_view_groups.append(group_samples)

        elif self.cross_view_type == 'CV':
            grouped = df.groupby(['subject', 'action', 'trial'])
            
            for (subject, action, trial), group in grouped:
                viewpoints = group['viewpoint'].unique()
                cv_viewpoints = [v for v in viewpoints if v in [1, 2]]

                if len(cv_viewpoints) >= 1: 
                    group_samples = []

                    for viewpoint in cv_viewpoints:
                        sample = group[group['viewpoint'] == viewpoint].iloc[0]
                        group_samples.append({
                            'video_id': sample['video'],
                            'subject': sample['subject'],
                            'action': sample['action'],
                            'viewpoint': sample['viewpoint'],
                            'trial': sample['trial']
                        })

                    self.cross_view_groups.append(group_samples)

        elif self.cross_view_type == 'CV_test':
            test_data = df[df['viewpoint'] == 3] 
            for idx, row in test_data.iterrows():
                group_samples = [{
                    'video_id': row['video'],
                    'subject': row['subject'],
                    'action': row['action'],
                    'viewpoint': row['viewpoint'],
                    'trial': row.get('trial', 0) 
                }]
                self.cross_view_groups.append(group_samples)
                
        elif self.cross_view_type == 'cs_single_test':
            grouped = df.groupby(['subject', 'action', 'trial'])
            
            for (subject, action, trial), group in grouped:
                viewpoints = group['viewpoint'].dropna().unique()

                for viewpoint in sorted(viewpoints):
                    sample = group[group['viewpoint'] == viewpoint].iloc[0]
                    self.cross_view_groups.append([{
                        'video_id': sample['video'],
                        'subject': sample['subject'],
                        'action': sample['action'],
                        'viewpoint': sample['viewpoint'],
                        'trial': sample['trial']
                    }])

    def _build_ntu_groups(self, df):
        
        self.cross_view_groups = []
        
        from configuration import build_config
    
        cfg = build_config(self.dataset)
        
        effective_type = self.cross_view_type  

        if effective_type == 'CS':
            grouped = df.groupby(['subject', 'action', 'repetition', 'setup'])
            
            for (subject, action, repetition, setup), group in grouped:
                cameras = group['camera'].dropna().unique()
                
                if len(cameras) >= 1:
                    group_samples = []
                    for camera in sorted(cameras):
                        sample = group[group['camera'] == camera].iloc[0]
                        group_samples.append({
                            'video_id': sample['video_id'],
                            'subject': sample['subject'],
                            'action': sample['action'],
                            'camera': sample['camera'],
                            'repetition': sample['repetition'],
                            'setup': sample['setup']
                        })
                    self.cross_view_groups.append(group_samples)

        elif effective_type == 'CV':
            grouped = df.groupby(['subject', 'action', 'repetition', 'setup'])
            
            for (subject, action, repetition, setup), group in grouped:
                cameras = group['camera'].dropna().unique()
                
                selected_cameras = [cam for cam in sorted(cameras) 
                                  if cam in cfg.cv_train_views]
                
                if len(selected_cameras) >= 2:
                    group_samples = []
                    for camera in selected_cameras:
                        sample = group[group['camera'] == camera].iloc[0]
                        group_samples.append({
                            'video_id': sample['video_id'],
                            'subject': sample['subject'],
                            'action': sample['action'],
                            'camera': sample['camera'],
                            'repetition': sample['repetition'],
                            'setup': sample['setup']
                        })
                    self.cross_view_groups.append(group_samples)
        
        elif effective_type == 'cs_single_test':
            grouped = df.groupby(['subject', 'action', 'repetition', 'setup'])
            
            for (subject, action, repetition, setup), group in grouped:
                cameras = group['camera'].dropna().unique()
                
                for camera in sorted(cameras):
                    sample = group[group['camera'] == camera].iloc[0]
                    self.cross_view_groups.append([{
                        'video_id': sample['video_id'],
                        'subject': sample['subject'],
                        'action': sample['action'],
                        'camera': sample['camera'],
                        'repetition': sample['repetition'],
                        'setup': sample['setup']
                    }])

        elif effective_type == 'CV_test':
            test_data = df[df['camera'].isin(cfg.cv_test_views)] 
            for idx, row in test_data.iterrows():
                group_samples = [{
                    'video_id': row['video_id'],
                    'subject': row['subject'],
                    'action': row['action'],
                    'camera': row['camera'],
                    'repetition': row['repetition'],
                    'setup': row['setup']
                }]
                self.cross_view_groups.append(group_samples)

    def _build_pkummd_groups(self, df):
        self.cross_view_groups = []
        
        from configuration import build_config
        
        cfg = build_config(self.dataset)
        
        if self.cross_view_type == 'CS':
            grouped = df.groupby(['subject', 'action', 'confidence'])
            
            for (subject, action, confidence), group in grouped:
                cameras = group['camera'].dropna().unique()
                
                if len(cameras) >= 1:
                    group_samples = []
                    for camera in sorted(cameras):
                        sample = group[group['camera'] == camera].iloc[0]
                        group_samples.append({
                            'video_id': sample['video_id'],
                            'subject': sample['subject'],
                            'action': sample['action'],
                            'camera': sample['camera'],
                            'start': sample['start'],
                            'end': sample['end'],
                            'confidence': sample['confidence']
                        })
                    self.cross_view_groups.append(group_samples)

        elif self.cross_view_type == 'CV':
            grouped = df.groupby(['subject', 'action', 'confidence'])
            
            for (subject, action, confidence), group in grouped:
                cameras = group['camera'].dropna().unique()
                
                selected_cameras = [cam for cam in sorted(cameras) 
                                  if cam in cfg.cv_train_views]
                
                if len(selected_cameras) >= 1: 
                    group_samples = []
                    for camera in selected_cameras:
                        sample = group[group['camera'] == camera].iloc[0]
                        group_samples.append({
                            'video_id': sample['video_id'],
                            'subject': sample['subject'],
                            'action': sample['action'],
                            'camera': sample['camera'],
                            'start': sample['start'],
                            'end': sample['end'],
                            'confidence': sample['confidence']
                        })
                    self.cross_view_groups.append(group_samples)
        
        elif self.cross_view_type == 'cs_single_test':
            for idx, row in df.iterrows():
                self.cross_view_groups.append([{
                    'video_id': row['video_id'],
                    'subject': row['subject'],
                    'action': row['action'],
                    'camera': row['camera'],
                    'start': row['start'],
                    'end': row['end'],
                    'confidence': row['confidence']
                }])

        elif self.cross_view_type == 'CV_test':
            test_data = df[df['camera'].isin(cfg.cv_test_views)]
            for idx, row in test_data.iterrows():
                group_samples = [{
                    'video_id': row['video_id'],
                    'subject': row['subject'],
                    'action': row['action'],
                    'camera': row['camera'],
                    'start': row['start'],
                    'end': row['end'],
                    'confidence': row['confidence']
                }]
                self.cross_view_groups.append(group_samples)

    def __len__(self):
        return len(self.cross_view_groups)

    def __getitem__(self, index):
        group = self.cross_view_groups[index]
        frames_list = []
        viewpoints_list = []
        actions_list = []

        for sample in group:
            if self.dataset == 'numa':
                row = [sample['subject'], sample['action'], sample['video_id'], sample['viewpoint']]
                viewpoints_list.append(int(sample['viewpoint']) - 1) 
            elif self.dataset in ['ntu_rgbd_60', 'ntu_rgbd_120']:
                row = [sample['subject'], sample['action'], sample['video_id'], sample['camera'], sample['repetition'],
                       sample['setup']]
                viewpoints_list.append(int(sample['camera']) - 1)  
            elif self.dataset == 'pkummd':
                row = [sample['subject'], sample['action'], sample['video_id'], sample['start'], sample['end']]
                viewpoints_list.append(int(sample['camera']) - 1)  

            frames = frame_creation(row, self.dataset, self.videos_folder,
                                    self.height, self.width, self.num_frames, self.transform)
            frames_list.append(frames)

            if self.dataset == 'numa':
                action_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 11: 8, 12: 9}
                original_action = int(sample['action'])
                action_label = action_mapping.get(original_action, 0)
            elif self.dataset == 'pkummd':
                action_label = int(sample['action']) - 1  
            else:
                action_label = int(sample['action']) - 1 

            actions_list.append(action_label)

        if self.dataset != 'pkummd':
            frame_shapes = [f.shape[0] for f in frames_list]
            if not all(shape == self.num_frames for shape in frame_shapes):
                return None, None, None, index
    
        if self.dataset == 'pkummd':
            min_frames = min(f.shape[0] for f in frames_list)
            frames_list = [f[:min_frames] for f in frames_list]
        
        stacked_frames = torch.stack(frames_list) 
        viewpoints = torch.tensor(viewpoints_list)
        actions = torch.tensor(actions_list)

        return stacked_frames, viewpoints, actions, index

def cross_view_collate(batch, cross_view_type='CS'):
    all_frames = []
    all_viewpoints = []
    all_actions = []
    all_indices = []

    for item in batch:
        frames, viewpoints, actions, index = item
        
        if frames is None:
            continue 
        
        all_frames.append(frames)
        all_viewpoints.append(viewpoints)
        all_actions.append(actions)
        all_indices.append(index)

    if cross_view_type == 'CS':
        target_views = 3  
    elif cross_view_type == 'CV':
        target_views = 2 
    elif cross_view_type == 'CV_test':
        target_views = 1 
    elif cross_view_type == 'cs_single_test':
        target_views = 1 
    else:
        target_views = 3 
    
    processed_frames = []
    processed_viewpoints = []
    processed_actions = []
    
    for frames, viewpoints, actions in zip(all_frames, all_viewpoints, all_actions):
        current_views = frames.shape[0]
        
        if current_views < target_views:
            if current_views == 1:
                interpolated_frame = frames[0] + torch.randn_like(frames[0]) * 0.01
                transformed_frame = torch.flip(frames[0], dims=[-1]) + torch.randn_like(frames[0]) * 0.01
                
                if target_views == 2:
                    frames = torch.cat([frames, interpolated_frame.unsqueeze(0)], dim=0)
                    viewpoints = torch.cat([viewpoints, viewpoints.repeat(1)])
                    actions = torch.cat([actions, actions.repeat(1)])
                elif target_views == 3:
                    frames = torch.cat([frames, interpolated_frame.unsqueeze(0), transformed_frame.unsqueeze(0)], dim=0)
                    viewpoints = torch.cat([viewpoints, viewpoints.repeat(2)])
                    actions = torch.cat([actions, actions.repeat(2)])
              
            elif current_views == 2 and target_views == 3:
                interpolated_frame = 0.5 * frames[0] + 0.5 * frames[1]
                frames = torch.cat([frames, interpolated_frame.unsqueeze(0)], dim=0)
                
                interpolated_viewpoint = (viewpoints[0] + viewpoints[1]) / 2
                viewpoints = torch.cat([viewpoints, interpolated_viewpoint.unsqueeze(0)])
                actions = torch.cat([actions, actions[0:1]])
        
        elif current_views > target_views:
            frames = frames[:target_views]
            viewpoints = viewpoints[:target_views]
            actions = actions[:target_views]
        
        assert frames.shape[0] == target_views, f"The number of viewpoints is incorrect after sample processing:{frames.shape[0]} != {target_views}"
        
        processed_frames.append(frames)
        processed_viewpoints.append(viewpoints)
        processed_actions.append(actions)

    batch_frames = torch.stack(processed_frames, dim=0)  # [batch_size, target_views, 16, 3, 224, 224]
    batch_viewpoints = torch.stack(processed_viewpoints, dim=0)  # [batch_size, target_views]
    batch_actions = torch.stack(processed_actions, dim=0)  # [batch_size, target_views]

    return batch_frames, batch_viewpoints, batch_actions, all_indices

_hdf5_cache = {}
_max_cache_size = 50  

def clear_hdf5_cache():
    global _hdf5_cache
    for file_path, h5_file in _hdf5_cache.items():
        h5_file.close()
    _hdf5_cache.clear()
    print("HDF5 cache has been cleared.")

def get_cache_info():
    return f"HDF5 cache: {len(_hdf5_cache)}/{_max_cache_size} files"

def frame_creation(row, dataset, videos_folder, height, width, num_frames, transform):
    if dataset == "ntu_rgbd_120" or dataset == 'ntu_rgbd_60':
        list16 = []
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], row[3], row[4]
        
        file_path = os.path.join(videos_folder, f'{video_id}.hdf5')
        if not os.path.exists(file_path) and '.avi' in video_id:
            file_path_no_avi = os.path.join(videos_folder, f'{video_id.replace(".avi", "")}.hdf5')
            if os.path.exists(file_path_no_avi):
                file_path = file_path_no_avi
        
        if file_path not in _hdf5_cache:
            if len(_hdf5_cache) >= _max_cache_size:
                oldest_key = next(iter(_hdf5_cache))
                _hdf5_cache[oldest_key].close()
                del _hdf5_cache[oldest_key]
        
            _hdf5_cache[file_path] = h5py.File(file_path, 'r')

        h5_file = _hdf5_cache[file_path]
        frames = h5_file['default'][:]
        frames = torch.from_numpy(frames).float()

        frame_indexer = np.linspace(0, int(frames.shape[0]) - 1, num_frames).astype(int)
        for i, frame in enumerate(frames):
            if i in frame_indexer:
                list16.append(frame)
        frames = torch.stack([frame for frame in list16])

        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.

        if transform:
            frames = frames.transpose(0, 1)
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        return frames


    elif dataset == "pkummd":
        skeleton = False  
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])

        if not skeleton:
            file_path = os.path.join(videos_folder,
                                    f'{video_id}_{int(subject)}_{action}_{start_frame}_{end_frame}.hdf5')
            
            if file_path not in _hdf5_cache:
                if len(_hdf5_cache) >= _max_cache_size:
                    oldest_key = next(iter(_hdf5_cache))
                    _hdf5_cache[oldest_key].close()
                    del _hdf5_cache[oldest_key]
                
                _hdf5_cache[file_path] = h5py.File(file_path, 'r')
            
            h5_file = _hdf5_cache[file_path]
            
            total_frames = h5_file['default'].shape[0]
            frame_indexer = np.linspace(0, total_frames - 1, 16).astype(int)
            
            list16 = []
            for idx in frame_indexer:
                list16.append(h5_file['default'][idx])
            
            frames = np.stack(list16)
            frames = torch.from_numpy(frames).float()

            for i, frame in enumerate(frames):
                frames[i] = frames[i] / 255.

            if transform:
                frames = frames.transpose(0, 1)
                frames = transform(frames)
                frames = frames.transpose(0, 1)

            return frames


        else:
            path = '/skeleton'
            processed_action_skeletons = []
            frame_indexer = np.linspace(start_frame, end_frame - 1, 16).astype(int)

            for i in sorted(frame_indexer):
                action_skeletons = open(os.path.join(path, f'{video_id}.txt'), 'r').readlines()[i].split(' ')
                frame_skeleton = [float(ele) for ele in action_skeletons]
                frame_skeleton = torch.tensor(frame_skeleton)

                sub1 = frame_skeleton[:75].reshape((3, 25))  # first 75 entries are 3x25 skeletons for subject 1
                sub2 = frame_skeleton[75:].reshape(
                    (3, 25))  # last 75 entries are 3x25 skeletons for subject 2 (0s if no sub 2)
                frame_skeleton = torch.stack((sub1, sub2),
                                             dim=2)  # stack them to get same 3x25x2 shape as PKUMMD website
                processed_action_skeletons.append(frame_skeleton)

            processed_skeletons = torch.stack([ele for ele in
                                               processed_action_skeletons])  # 16x3x25x2: skeletons from 16 equidistant frames in action range
            return processed_skeletons

    elif dataset == "numa":
        list16 = []
        subject, action, video_id, viewpoint = row
        
        file_path = f'{videos_folder}/{video_id[:-4]};{action}.hdf5'
        
        if file_path not in _hdf5_cache:
            if len(_hdf5_cache) >= _max_cache_size:
                oldest_key = next(iter(_hdf5_cache))
                _hdf5_cache[oldest_key].close()
                del _hdf5_cache[oldest_key]
            
            _hdf5_cache[file_path] = h5py.File(file_path, 'r')

        h5_file = _hdf5_cache[file_path]
        frames = h5_file['default'][:]
        frames = torch.from_numpy(frames)
        frames = frames.type(torch.float32)

        frame_indexer = np.linspace(0, len(frames) - 1, num_frames).astype(int)

        for i in frame_indexer:
            list16.append(frames[i])
        frames = torch.stack([frame for frame in list16])

        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.

        if transform:
            frames = frames.transpose(0, 1)
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        return frames


if __name__ == '__main__':
    shuffle = False
    cfg = build_config('pkummd')
    transform_train = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            RandomShortSideScale(
                min_size=224,
                max_size=256,
            ),
            RandomCrop(224),
            RandomHorizontalFlip(p=0.5)
        ]
    )
    transform_test = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(
                size=256
            ),
            CenterCrop(224)
        ]
    )

    for (clips, sv_clips, sa_clips, views, actions, keys) in tqdm(dataloader):
        print(clips.shape, sv_clips.shape, sa_clips.shape, views, actions)
        print(actions.shape, views.shape)
        exit()



