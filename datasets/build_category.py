import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from functools import partial
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler

from xclip.datasets.build import PIPELINES, img_norm_cfg, BaseDataset, mmcv_collate 

# ref: https://discuss.pytorch.org/t/new-subset-every-epoch/85018
class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None, output_file=""):
        self.data_source = data_source
        self._num_samples = num_samples
        self.output_file = output_file

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        sample_indices = torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist()
        if self.output_file != "":
            with open(self.output_file, 'a+') as f:
                print(sample_indices, file=f)
        return iter(sample_indices)

    def __len__(self):
        return self.num_samples

class CategoryVideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, categories_file, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file
        self.categories_file = categories_file

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    @property
    def categories(self):
        categories_all = pd.read_csv(self.categories_file)
        return categories_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(' ')
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, category, label = line_split
                    if category == '': continue # case for KDIANBXG.mp4, no given annotations
                    category = list(map(int, category.strip().split(',')))
                    label = list(map(int, label.strip().split(',')))
                else:
                    filename, category, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, category=category, label=label, tar=self.use_tar_format))
        return video_infos


def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'category', 'filename'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]
        

    train_data = CategoryVideoDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, categories_file=config.DATA.CATEGORY_LIST,
                              multi_class=True, num_classes=config.DATA.NUM_CLASSES, pipeline=train_pipeline)
    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler_train = torch.utils.data.DistributedSampler(
    #     train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    TRAIN_SAMPLE_RATIO = 0.05
    sampler = RandomSampler(train_data, num_samples=int(len(train_data) * TRAIN_SAMPLE_RATIO + 1), output_file=osp.join(config.OUTPUT, "train_indices.txt"))
    train_loader = DataLoader(
        train_data, sampler=sampler,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )
    
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'category', 'filename'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    
    val_data = CategoryVideoDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, 
                                    labels_file=config.DATA.LABEL_LIST, categories_file=config.DATA.CATEGORY_LIST,
                                    multi_class=True, num_classes=config.DATA.NUM_CLASSES, pipeline=val_pipeline)
    
    VAL_SAMPLE_RATIO = 0.05
    indices = torch.randperm(len(val_data))
    indices = indices[:int(len(indices) * VAL_SAMPLE_RATIO + 1)]
    with open(osp.join(config.OUTPUT, 'val_indices.txt'), 'a+') as f:
        print(indices.tolist(), file=f)
    sampler_val = SubsetRandomSampler(indices)
    # sampler_val = RandomSampler(val_data, num_samples=int(len(val_data) * TRAIN_SAMPLE_RATIO + 1))
    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    return train_data, val_data, train_loader, val_loader