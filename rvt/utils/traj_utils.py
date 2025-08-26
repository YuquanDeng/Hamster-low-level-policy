from rvt.utils.traj_dataset import *
import argparse
import numpy as np
import os
from PIL import Image
from termcolor import cprint
from rvt.utils.rlbench_utils import COLOSSEUM_15_TASKS

TRAJ_TYPE2TRAJ_IMG_NAME = {
    "full": "gt_traj",
    "full_action": "gt_traj_with_action",
    "grasping": "grasping_traj",
    "grasping_action": "grasping_traj_with_action"
}

def parse_traj(data_path, meta_path, traj_type, tasks, save_dir, verbose=True, debug=False, variation_dir=False):
    # data_path = '/home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128/'
    # meta_path = '/home/nil/manipulation/RVT2/rvt/test_raw_data_representation/'
    
    # key: task_name, value: [invalid_episode_1, invalid_episode_2, ...]
    if traj_type.startswith("grasping"):
        print("traj_type", traj_type)
        invalid_episodes = {}

    cprint(f"process tasks: {tasks}", "yellow")
    dataset = RLBenchTrajDataset(tasks, 
                                data_path=data_path, 
                                meta_path=meta_path, 
                                variation_mode='all', 
                                mode='localhost',
                                impainting='to_end:0', 
                                traj_image_size=128,
                                traj_type=traj_type, 
                                resample_to=32,
                                phase="test",
                                constant_speed=True,
                                image_range=100,
                                variation_dir=variation_dir)

    num_data = len(dataset)
    
    start_num = 0
    if debug:
        num_data = 100

    for i in range(start_num, num_data):
        data = dataset[i]
        task_name = data["meta"]["task"]
        episode_num = data["meta"]["episode_num"]
        
        # save invalid episode number and pass
        if not data["valid"]:
            if task_name not in invalid_episodes:
                invalid_episodes[task_name] = []
            invalid_episodes[task_name].append(episode_num)
            invalid_episodes[task_name].sort()         
            print("save the black image instead")
            # continue
        
        # get traj img
        traj_img = data["traj"].cpu().numpy() # 3, 128, 128
        traj_img = ((traj_img + 1) * 127.5).astype(np.uint8) # 0-255 scale
        traj_img = np.transpose(traj_img, (1, 2, 0)) # 128, 128, 3
        
        if not variation_dir:
            episodes_dir = "all_variations/episodes"
        else:
            episodes_dir = f"variation{data['meta']['variation_number']}/episodes"

        save_path = os.path.join(save_dir, task_name, episodes_dir)
        os.makedirs(save_path, exist_ok=True)
            
        img = Image.fromarray(traj_img)
        img.save(os.path.join(save_path, f"episode{episode_num}_{TRAJ_TYPE2TRAJ_IMG_NAME[traj_type]}.png"))
        
        if verbose:
            print(os.path.join(save_path, f"episode{episode_num}_{TRAJ_TYPE2TRAJ_IMG_NAME[traj_type]}.png"))
   
    if traj_type.startswith("grasping"):
        with open(os.path.join(save_dir, "invalid_episodes.pkl"), "wb") as file:
            output_dict = {}
            output_dict["task2invalid_episodes"] = invalid_episodes
            pickle.dump(output_dict, file)
        print(f"save {os.path.join(save_dir, 'invalid_episodes.pkl')}")
    cprint("finished processing.")

def update_invalid_episodes(invalid_grasp_traj_episodes_path, replay_dir):
    # Load the dictionary from the pickle file
    with open(invalid_grasp_traj_episodes_path, 'rb') as file:
        data = pickle.load(file)

    # Initialize the task2invalid_replay dictionary if it doesn't exist
    data["task2invalid_replay"] = {}

    # Iterate over each task_name in the task2invalid_episodes
    for task_name in data["task2invalid_episodes"].keys():
        task_dir = os.path.join(replay_dir, task_name)

        # Ensure the task directory exists
        if not os.path.isdir(task_dir):
            print(f"Task directory {task_dir} does not exist.")
            continue

        # Get all replay files and sort them numerically
        replay_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.replay')], key=lambda x: int(x.split('.')[0]))

        # Iterate over each replay file in numerical order
        for replay_file in replay_files:
            replay_path = os.path.join(task_dir, replay_file)

            # Load the replay file
            with open(replay_path, 'rb') as rf:
                replay_data = pickle.load(rf)

            episode_idx = replay_data.get("episode_idx")
            replay_idx = int(replay_file[:-7])
            
            # Check if the episode_idx is in the invalid episodes list
            if episode_idx in data["task2invalid_episodes"][task_name]:
                if task_name not in data["task2invalid_replay"]:
                    data["task2invalid_replay"][task_name] = []
                data["task2invalid_replay"][task_name].append(replay_idx)
                print(f"Task {task_name}: Invalid replay file {replay_file} (episode {episode_idx}) found and added.")

    # Save the updated dictionary back to the pickle file
    with open(invalid_grasp_traj_episodes_path, 'wb') as file:
        pickle.dump(data, file)


ALL_TASKS = [
    'close_jar', 'meat_off_grill', 'place_shape_in_shape_sorter', 'put_groceries_in_cupboard', 'reach_and_drag', 'stack_cups',
    'insert_onto_square_peg', 'open_drawer', 'place_wine_at_rack_location', 'put_item_in_drawer', 'slide_block_to_color_target',
    'sweep_to_dustpan_of_size', 'light_bulb_in', 'place_cups', 'push_buttons', 'put_money_in_safe', 'stack_blocks', 'turn_tap'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--data_path', type=str, default='/home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128/',
                        help='Path to the data directory')
    parser.add_argument('--meta_path', type=str, default='/home/nil/manipulation/RVT2/rvt/test_raw_data_representation/',
                        help='Path to the meta data directory')
    parser.add_argument('--traj_type', type=str, default='full',
                        help='Type of trajectory')
    parser.add_argument('--tasks')
    parser.add_argument('--save_dir', type=str, default="/home/nil/manipulation/RVT2/rvt/test_rlbench_traj_dataset")
    parser.add_argument('--debug', action="store_true")
    
    parser.add_argument("--task2invalid_replay", action="store_true")
    parser.add_argument("--invalid_grasp_traj_episodes_path", type=str)
    parser.add_argument("--replay_dir", type=str)
    parser.add_argument("--variation_dir", action="store_true")
    
    args = parser.parse_args()
 
    if args.task2invalid_replay:
        print("use task2invalid_replay")
        update_invalid_episodes(args.invalid_grasp_traj_episodes_path, args.replay_dir)
    else:
        if args.tasks == "all":
            tasks = ALL_TASKS
        elif args.tasks == "colosseum_15":
            tasks = COLOSSEUM_15_TASKS
        else:
            tasks = args.tasks.split(",")
        debug = args.debug
        
        parse_traj(
            data_path=args.data_path,
            meta_path=args.meta_path,
            traj_type=args.traj_type,
            tasks=tasks,
            save_dir=args.save_dir,
            verbose=False,
            debug=debug,
            variation_dir=args.variation_dir
        )




