COLMAP_EXE=$1
DATASET_PATH=$2
POSE_PATH=$3
IMAGE_PATH=$4
POSE_SCALE_PATH=$5
FIX_TRANS_REFINE_ROT=$6


for i in {1}
do
    ${COLMAP_EXE} point_triangulator \
        --database_path $DATASET_PATH/database.db \
        --image_path $IMAGE_PATH \
        --input_path $POSE_PATH/sparse/0 \
        --output_path $POSE_PATH/sparse/0 \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.max_extra_param 0 \
        --clear_points 0 \
        --Mapper.ba_global_max_num_iterations 30 \
        --Mapper.filter_max_reproj_error 4 \
        --Mapper.filter_min_tri_angle 0.5 \
        --Mapper.tri_min_angle 0.5 \
        --Mapper.tri_ignore_two_view_tracks 1 \
        --Mapper.tri_complete_max_reproj_error 4 \
        --Mapper.tri_continue_max_angle_error 4

    # ${COLMAP_EXE} bundle_adjuster \
    #     --input_path $DATASET_PATH/sparse/0 \
    #     --output_path $DATASET_PATH/sparse/0 \
    #     --BundleAdjustment.max_num_iterations 30

    ${COLMAP_EXE} rig_bundle_adjuster \
        --input_path $POSE_PATH/sparse/0 \
        --output_path $POSE_PATH/sparse/0 \
        --rig_config_path "$POSE_PATH/cam_rigid_config.json" \
        --estimate_rig_relative_poses 0 \
        --RigBundleAdjustment.refine_relative_poses 1 \
        --BundleAdjustment.max_num_iterations 50 \
        --BundleAdjustment.refine_focal_length 0 \
        --BundleAdjustment.refine_principal_point 0 \
        --BundleAdjustment.refine_extra_params 0
        #--RigBundleAdjustment.fix_trans_refine_rot $FIX_TRANS_REFINE_ROT \

done

python $POSE_SCALE_PATH/pose_scale_correct.py --sparse_path ${POSE_PATH}/sparse/0