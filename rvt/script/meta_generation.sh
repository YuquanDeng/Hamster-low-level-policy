
ROOT_DIR=/home/nil/robotics/RVT/rvt

task=light_bulb_in
num_episodes=100
image_size=128
data_root_dir=$ROOT_DIR/data/train
save_dir=$ROOT_DIR/data/train_meta

python3 $ROOT_DIR/utils/meta_utils.py --tasks $task \
    --num_episodes $num_episodes \
    --store_meta_info \
    --data_root_dir  $data_root_dir \
    --save_dir $save_dir \
    --image_size $image_size
