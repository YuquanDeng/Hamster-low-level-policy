
# change the root directory here
ROOT_DIR=/home/nil/robotics/RVT/rvt 

cd $ROOT_DIR
python train.py \
--exp_cfg_path $ROOT_DIR/configs/rvt2_traj_example.yaml \
--mvt_cfg_path $ROOT_DIR/mvt/configs/rvt2_traj.yaml \
--device 0 \
--visualize

