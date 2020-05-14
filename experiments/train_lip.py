## train lip
## trainer
run_train: true
num_epochs: 30
batch_size: 64
checkpoint_dir: results/lip
save_checkpoint_freq: 1
keep_num_checkpoint: 2
resume: true
use_logger: true
log_freq: 500

## dataloader
num_workers: 4
image_size: 64
load_gt_depth: false
train_val_data_dir: /pub1/hao66/dataset/lip_dataset_cropped/train

## model
model_name: unsup3d_lip
min_depth: 0.9
max_depth: 1.1
xyz_rotation_range: 60  # (-r,r) in degrees
xy_translation_range: 0.1  # (-t,t) in 3D
z_translation_range: 0  # (-t,t) in 3D
lam_perc: 1
lam_flip: 0.5
lr: 0.0001

## renderer
rot_center_depth: 1.0
fov: 10  # in degrees
tex_cube_size: 2
