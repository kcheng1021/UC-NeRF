export DATA_DIR='/data/kcheng/codes/waymo/segment-10061305430875486848_1080_000_1100_000_with_camera_labels'

accelerate launch train.py \
    --gin_configs=configs/waymo.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.depth_dir = '/data/kcheng/codes/UC-NeRF/CER-MVS-main/results/custom/depths'" \
    --gin_bindings="Config.exp_name = './checkpoints/waymo_10063_final'" \
    --gin_bindings="Config.factor = 4" \
    --gin_bindings="Config.max_steps = 30000" \
    --gin_bindings="Config.cam_type = 6" \
    --gin_bindings="Config.brightness_correction = True" \
    --gin_bindings="Config.model_sky = True" \
    --gin_bindings="Config.virtual_poses = False" \
    --gin_bindings="Config.refine_name = '/data/kcheng/codes/waymo/waymo100613/mvs_driving_all_pose/sparse/0/pose.json'"

accelerate launch eval.py \
    --gin_configs=configs/waymo.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.depth_dir = '/data/kcheng/codes/UC-NeRF/CER-MVS-main/results/custom/depths'" \
    --gin_bindings="Config.exp_name = './checkpoints/waymo_10063_final'" \
    --gin_bindings="Config.factor = 4" \
    --gin_bindings="Config.max_steps = 30000" \
    --gin_bindings="Config.cam_type = 6" \
    --gin_bindings="Config.brightness_correction = True" \
    --gin_bindings="Config.model_sky = True" \
    --gin_bindings="Config.refine_name = '/data/kcheng/codes/waymo/waymo100613/mvs_driving_all_pose/sparse/0/pose.json'"

accelerate launch render.py \
    --gin_configs=configs/waymo.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.depth_dir = '/data/kcheng/codes/UC-NeRF/CER-MVS-main/results/custom/depths'" \
    --gin_bindings="Config.exp_name = './checkpoints/waymo_10063_final'" \
    --gin_bindings="Config.factor = 4" \
    --gin_bindings="Config.max_steps = 30000" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 120" \
    --gin_bindings="Config.render_video_fps = 5" \
    --gin_bindings="Config.cam_type = 6" \
    --gin_bindings="Config.brightness_correction = True" \
    --gin_bindings="Config.model_sky = True" \
    --gin_bindings="Config.load_sky_segments = False"  \
    --gin_bindings="Config.refine_name = '/data/kcheng/codes/waymo/waymo100613/mvs_driving_all_pose/sparse/0/pose.json'"

