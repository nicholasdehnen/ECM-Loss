_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1203,
            cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
            loss_cls=dict(
                type='EffectiveClassMarginLossNickWAuthorBgHandling',
                stats_file='lvis_v1_class_dist.txt', # needs to be adjusted based on dataset
                use_sigmoid=True,
                num_classes=1203,
                loss_weight=1.0, 
                )),
        mask_head=dict(num_classes=1203, # needs to be adjusted based on dataset
                       predictor_cfg=dict(type='NormedConv2d', tempearture=20))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2)) # detectron2 default.

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10000, 
    warmup_ratio=0.0001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# wandb log config
log_config = dict(
    interval = 50,
    hooks = [
    dict(type='MMDetWandbHook',
        init_kwargs={'project': 'ecm-loss'},
        log_checkpoint=False,
        log_checkpoint_metadata=False,
        num_eval_images=100),
    dict(type='TextLoggerHook'),
    ]

)

data = dict(
    samples_per_gpu=3,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline
            )
        )
    )
evaluation = dict(interval=12, metric=['bbox', 'segm'])


