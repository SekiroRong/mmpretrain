# dataset settings
dataset_type = 'CustomDataset'
data_root = r'C:\D\Data\StreetFighter'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=192, crop_ratio_range=(0.67, 1.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(type='PackInputs')
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     # dict(type='RandomResizedCrop', scale=192, crop_ratio_range=(0.67, 1.0)),
#     # dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='SimMIMMaskGenerator',
#         input_size=192,
#         mask_patch_size=32,
#         model_patch_size=4,
#         mask_ratio=0.6),
#     dict(type='PackInputs')
# ]

train_dataloader = dict(
    batch_size=64,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='',
        pipeline=train_pipeline))

# test_dataloader = dict(
#     batch_size=32,
#     num_workers=1,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     collate_fn=dict(type='default_collate'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix='',
#         pipeline=test_pipeline))

# val_dataloader = test_dataloader
# test_evaluator = dict(type='do_nothing')
# val_evaluator = test_evaluator
