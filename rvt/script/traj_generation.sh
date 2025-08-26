ROOT_DIR=/home/nil/robotics/RVT/rvt

data_path=$ROOT_DIR/data/train
meta_path=$ROOT_DIR/data/train_meta
save_dir=$ROOT_DIR/data/train_traj
traj_type=full_action
tasks=light_bulb_in

python3 $ROOT_DIR/utils/traj_utils.py \
--data_path ${data_path} \
--meta_path ${meta_path} \
--traj_type ${traj_type} \
--tasks ${tasks} \
--save_dir ${save_dir} \
