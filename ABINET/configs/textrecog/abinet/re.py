log_config = dict(
    interval=1,                     # Print log every 1 iteration
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
    '../_base_/datasets/my_dataset.py',
    '_base_abinet.py',
]

# Load pre-trained weights
load_from = 'https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_pretrain-45deac15.pth'

# If the pre-trained weights is not working, uncomment the next line
# load_from = None

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=0.5, norm_type=2)
)

train_cfg = dict(max_epochs=40)  # Epochs increased

param_scheduler = [
    dict(type='LinearLR', end=3, start_factor=0.001, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=35, eta_min=1e-6, begin=3, end=38),
    dict(type='ConstantLR', factor=0.1, begin=38, end=40)
]

# dataset settings
train_list = [_base_.my_dataset_train]
test_list = [_base_.my_dataset_val]

# Based on v2's stable pipeline, adding safe data augmentation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    
    dict(
        type='RandomChoice',
        transforms=[
            [],
            [dict(type='Resize', scale=(130, 34), keep_ratio=False)],
        ],
        prob=[0.7, 0.3]
    ),
    
    dict(type='Resize', scale=(128, 32), keep_ratio=False),
    dict(type='Pad', size=(128, 32), pad_val=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    ),
    dict(type='PackTextRecogInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(128, 32), keep_ratio=False),
    dict(type='Pad', size=(128, 32), pad_val=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    ),
    dict(type='PackTextRecogInputs')
]

train_dataset = dict(
    type='ConcatDataset',
    datasets=train_list,
    pipeline=train_pipeline
)

test_dataset = dict(
    type='ConcatDataset',
    datasets=test_list,
    pipeline=test_pipeline
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
    prefetch_factor=2,
    pin_memory=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset,
    pin_memory=True
)

test_dataloader = val_dataloader

val_evaluator = dict(
    dataset_prefixes=['my_dataset'])
test_evaluator = val_evaluator

# Self adapative learning settings
auto_scale_lr = dict(base_batch_size=192 * 8)

work_dir = './work_dirs/re_v3_1_log'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        save_last=True,
        save_best='auto',
        max_keep_ckpts=3
    )
)

# Added visualization backend for TensorBoard
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='TextRecogLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

work_dir = './work_dirs/re_log'