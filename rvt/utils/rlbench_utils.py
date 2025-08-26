import os
import pickle
import glob

from PIL import Image
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor

# all 20 colosseums tasks.
COLOSSEUM_TASKS = [
    "basketball_in_hoop",
    "close_box",
    "close_laptop_lid",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "meat_on_grill",
    "move_hanger",
    "wipe_desk",
    "open_drawer",
    "slide_block_to_target",
    "reach_and_drag",
    "put_money_in_safe",
    "place_wine_at_rack_location",
    "insert_onto_square_peg",
    "stack_cups",
    "turn_oven_on",
    "straighten_rope",
    "setup_chess",
    "scoop_with_spatula"
]

# excluded 5 tasks with bad front views.
COLOSSEUM_15_TASKS = [
    'close_box', 
    'close_laptop_lid', 
    'hockey', 
    'wipe_desk',
    'open_drawer', 
    'slide_block_to_target', 
    'reach_and_drag', 
    'put_money_in_safe', 
    'place_wine_at_rack_location', 
    'insert_onto_square_peg', 
    'stack_cups', 
    'straighten_rope', 
    'setup_chess', 
    'scoop_with_spatula',
    'meat_on_grill'
]

# excluded 7 tasks repeated in peract_18.
COLOSSEUM_13_TASKS  = [
    "basketball_in_hoop",
    "close_box",
    "close_laptop_lid",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "move_hanger",
    "wipe_desk",
    "slide_block_to_target",
    "turn_oven_on",
    "straighten_rope",
    "setup_chess",
    "scoop_with_spatula"
]

# constants
EPISODE_FOLDER = 'episode%d'

CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]

IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_TYPES = [IMAGE_RGB, IMAGE_DEPTH]
IMAGE_FORMAT  = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'

DEPTH_SCALE = 2**24 - 1

# functions
def get_stored_demo_front_view(data_path, index):
  episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
  
  # low dim pickle file
  with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
    obs = pickle.load(f)

  # variation number
  with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
    obs.variation_number = pickle.load(f)
    
  num_steps = len(obs)
  for i in range(num_steps):

    # TODO: bad implementation! hot fix.
    # handle aggregated dataset mixed image format.
    rgb_filepath = os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)
    depth_filepath = os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)
    
    obs[i].front_rgb = np.array(Image.open(rgb_filepath))
    # obs[i].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
    # obs[i].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
    # obs[i].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))

    obs[i].front_depth = image_to_float_array(Image.open(depth_filepath), DEPTH_SCALE)
    near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
    far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
    obs[i].front_depth = near + obs[i].front_depth * (far - near)

    # obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    # near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
    # far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
    # obs[i].left_shoulder_depth = near + obs[i].left_shoulder_depth * (far - near)

    # obs[i].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    # near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
    # far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
    # obs[i].right_shoulder_depth = near + obs[i].right_shoulder_depth * (far - near)

    # obs[i].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    # near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
    # far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
    # obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)

    obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth, 
                                                                                    obs[i].misc['front_camera_extrinsics'],
                                                                                    obs[i].misc['front_camera_intrinsics'])

    # obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].left_shoulder_depth, 
    #                                                                                         obs[i].misc['left_shoulder_camera_extrinsics'],
    #                                                                                         obs[i].misc['left_shoulder_camera_intrinsics'])
    # obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].right_shoulder_depth, 
    #                                                                                          obs[i].misc['right_shoulder_camera_extrinsics'],
    #                                                                                          obs[i].misc['right_shoulder_camera_intrinsics'])
    # obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].wrist_depth, 
    #                                                                                        obs[i].misc['wrist_camera_extrinsics'],
    #                                                                                        obs[i].misc['wrist_camera_intrinsics'])
    
  return obs


def resample_traj(traj, gripper_open, resample_to=32, verbose=False, traj_dim=3, denormalization=False):
    """
    Resamples a given trajectory based on gripper state changes and returns the resampled trajectory,
    a binary indicator array for non-start points, and the resampled trajectory's RGB color mapping.

    Parameters:
    traj (np.ndarray): The original trajectory, a 2D array of shape (n, traj_dim), where 'n' is the number
                       of points in the trajectory and 'traj_dim' is the dimension of each point (e.g., 3 for 3D).
    gripper_open (np.ndarray): A binary array of length 'n', where each element indicates whether the gripper
                               is open (1) or closed (0) at the corresponding point in the trajectory.
    resample_to (int, optional): The number of points to resample the trajectory to. Default is 32.
    verbose (bool, optional): If True, prints the new non-start points array and the resampled trajectory.
                              Default is False.
    traj_dim (int, optional): The dimensionality of the trajectory points. Default is 3.

    Returns:
    resampled_traj (np.ndarray): The resampled trajectory, a 2D array of shape (resample_to, traj_dim).
    new_non_start_points (np.ndarray): A binary array of length 'resample_to' indicating which points
                                       in the resampled trajectory are non-start points.
    resample_traj_rgb (np.ndarray): A 2D array of shape (resample_to, 3) where each row represents the RGB
                                    color mapping of the corresponding point in the resampled trajectory.
    """
    non_start_points = np.concatenate([[-1], gripper_open[1:]-gripper_open[:-1]])+1
    
    n = traj.shape[0]
    dists = np.concatenate([[0], np.linalg.norm(np.diff(traj, axis=0), axis=-1)])
    dists = dists*non_start_points
    xp = np.cumsum(dists)
    
    total_dist = np.sum(dists)
    
    traj_start = 0
    x = np.linspace(traj_start, total_dist, resample_to)
    resampled_traj = np.stack([np.interp(x, xp, traj[:, i]) for i in range(traj_dim)], axis=1)
    new_traj_index = np.interp(x, xp, np.arange(n))
    new_non_start_points = np.ones_like(new_traj_index)
    start_point_index = np.where(non_start_points==0)[0]
    for i in start_point_index:
        # find the point in resampled_traj which is closest to traj[i]
        dist = np.linalg.norm(resampled_traj-traj[i], axis=-1)
        new_non_start_points[np.argmin(dist)] = 0

    if verbose:
        print("new_non_start_points: ", new_non_start_points)
        print("resampled_traj: ", resampled_traj)

    color_indices = np.linspace(0, 1, resampled_traj.shape[0])
    resample_traj_rgb = np.zeros_like(resampled_traj)
    for i in range(resampled_traj.shape[0]):
        resample_traj_rgb[i] = (255 * np.array(plt.cm.jet(color_indices[i]))[:3]).astype(int)
    
    resample_traj_rgb = resample_traj_rgb / 255.0 # 0-1 scale
    
    if denormalization:
        resample_traj_rgb = (resample_traj_rgb + 1) / 2 # shift between [0, 1]

    return resampled_traj, new_non_start_points, resample_traj_rgb

    

def get_variation_numbers_from_replays(task_replay_dir, raw_data_dir, save_dir):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get a sorted list of replay files from the original directory
    replay_files = sorted([f for f in os.listdir(task_replay_dir) if f.endswith('.replay')], key=lambda x: int(x.split('.')[0]))
    raw_data_episodes = sorted([f for f in os.listdir(raw_data_dir) if f.startswith('episode')], key=lambda x: int(x.split('.')[0][7:]))

    replay_idx2variation_num = {}
    for replay_file in replay_files:
        with open(os.path.join(task_replay_dir, replay_file), 'rb') as file:
            current_replay = pickle.load(file)

            replay_idx = int(replay_file.split('.')[0])
            episode_idx = current_replay["episode_idx"]
            variation_number_path = os.path.join(raw_data_dir, f'episode{episode_idx}', "variation_number.pkl")
            with open(variation_number_path, "rb") as file:
                variation_number = pickle.load(file)
            
            print("episode idx: ", episode_idx)
            print("replay idx: ", replay_idx)
            print("variation number: ", variation_number)
            print("-"*50)  
            replay_idx2variation_num[replay_idx] = variation_number
    
    assert len(replay_files) == len(replay_idx2variation_num)
    print("total number of replay: ", len(replay_idx2variation_num))
    task_name = task_replay_dir.split("/")[-1]
    save_filepath = os.path.join(save_dir, f'{task_name}_variation_number.pkl')
    with open(save_filepath, "wb") as file:
        pickle.dump(replay_idx2variation_num, file)
        print(f"save under {save_filepath}.")

def calculate_all_variations_numbers():
    variations_number_path = "/home/nil/manipulation/RVT2/rvt/debug_mask/light_bulb_in_variation_number.pkl"

    with open(variations_number_path, "rb") as file:
        variations_number = pickle.load(file)
    
    unique_variations_number = set()
    for replay_idx, variation_num in variations_number.items():
        unique_variations_number.add(variation_num)
    
    print("light bulb in all variation number: ", unique_variations_number)

def sanity_check(save_dir, task):
    save_filepath = os.path.join(save_dir, f"{task}_variation_number.pkl")
    with open(save_filepath, "rb") as file:
        replay_idx2variation_num = pickle.load(file)

    count = 0
    for idx in range(200, 250):
        print("replay idx: ", idx, "var number: ", replay_idx2variation_num[idx])



def extract_gt_mask(raw_dara_root, task, raw_data_dir, save_dir):
    episode_list = glob.glob(os.path.join(raw_data_dir, "*"))
    episode_list.sort()
    episode_path = episode_list[0] # only choose one

    foreground = tell_target(task, episode_path)
    mask_list = glob.glob(os.path.join(episode_path, "*_mask", "*.png"))
    mask_list.sort()
    mask_path = mask_list[0] # only choose one

    # extract mask
    img_array = load_mask(mask_path)
    print(np.unique(img_array))
    # best_base = find_best_base_value(img_array, task)
    mask_array = np.zeros_like(img_array)
    for mask_idx, (k, fg_list) in enumerate(foreground.items(), start=1):
        for fg in fg_list:
            mask_array[img_array==fg] = mask_idx
    mask_array = mask_array/len(foreground)
    colored_mask = cm.viridis(mask_array)
    colored_mask_image = Image.fromarray((colored_mask[:, :, :3] * 255).astype(np.uint8))
    output_path = mask_path.replace(raw_dara_root, save_dir)
    print("output: ", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # cv2.imwrite(output_path, colored_mask_image)
    colored_mask_image.save(output_path)

R_BG_VALUE = 68
G_BG_VALUE = 1
B_BG_VALUE = 84
BG_RGB_SUM_VALUE = 153 
OBJ_VALUE2RGB_VALUE = {
    316: [32, 144, 140],
    520: [253, 231, 36]
}

def draw_circle_on_mask(mask, center, value, radius=3):
    """
    Draws a filled circle on the mask at the specified center with the given radius and value.

    :param mask: 2D NumPy array representing the mask.
    :param center: Tuple of (x, y) coordinates for the center of the circle.
    :param radius: Integer representing the radius of the circle.
    :param value: Value to set within the circle.
    :return: None; the mask is modified in place.
    """
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    mask_index = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius ** 2
    mask[mask_index] = value

    return mask

def sample_from_gt_mask(mask, sample_type="pt"):
    """
    :param mask: numpy array with shape (3,128, 128)
    """


    gt_mask = np.sum(mask, axis=0)
    gt_mask[gt_mask == BG_RGB_SUM_VALUE] = 0
    object_values = np.unique(gt_mask) # (0, 316, 520)
    object_values = object_values[1:] # (0, 316, 520) -> (316, 520)


    result_mask = np.zeros((3, 128, 128), dtype=np.uint8)
    for value in object_values:
        # select index and random sample uniformly.
        indices = np.argwhere(gt_mask == value) 
        # print("values: ", value, "indices: ", indices)
        if len(indices) > 0:
            sampled_index = indices[np.random.choice(len(indices))]
            # print("sample index: ", sampled_index)

            if sample_type == "pt":
                x, y = sampled_index[0], sampled_index[1]
                result_mask[0, x, y] = OBJ_VALUE2RGB_VALUE[value][0]
                result_mask[1, x, y] = OBJ_VALUE2RGB_VALUE[value][1]
                result_mask[2, x, y] = OBJ_VALUE2RGB_VALUE[value][2]
            elif sample_type == "circle":
                r = draw_circle_on_mask(
                    mask=result_mask[0, :, :],
                    center=sampled_index,
                    value=OBJ_VALUE2RGB_VALUE[value][0]
                    )
                
                g = draw_circle_on_mask(
                    mask=result_mask[1, :, :],
                    center=sampled_index,
                    value=OBJ_VALUE2RGB_VALUE[value][1]
                    )
                
                b = draw_circle_on_mask(
                    mask=result_mask[2, :, :],
                    center=sampled_index,
                    value=OBJ_VALUE2RGB_VALUE[value][2]
                    )

    return result_mask


def main():
    task = "light_bulb_in"
    replay_root = "/home/nil/manipulation/RVT2/rvt/replay/128x128/replay_train"
    task_replay_dir = os.path.join(replay_root, task)
    save_dir = "/home/nil/manipulation/RVT2/rvt/gt_mask"
    raw_dara_root = "/home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128"
    raw_data_dir = os.path.join(raw_dara_root, task, "all_variations/episodes")

    # get_variation_numbers_from_replays(task_replay_dir=task_replay_dir, raw_data_dir=raw_data_dir, save_dir=save_dir)
    # calculate_all_variations_numbers()

    # # filter background color.
    # r = mask[0, :, :]
    # g = mask[1, :, :]
    # b = mask[2, :, :]

    # r[r == R_BG_VALUE] = 0 
    # g[g == G_BG_VALUE] = 0
    # b[b == B_BG_VALUE] = 0

    # # (128, 128) -> (1, 128, 128)
    # r = np.expand_dims(r, axis=0)
    # g = np.expand_dims(g, axis=0)
    # b = np.expand_dims(b, axis=0)

    # gt_mask = np.concatenate((r, g, b), axis=0)

    mask_filepath = "/home/nil/manipulation/RVT2/rvt/gt_mask/light_bulb_in/all_variations/episodes/episode0/front_mask/0.png"
    mask = np.array(Image.open(mask_filepath)).transpose((2, 0, 1))

    gt_point_mask = sample_from_gt_mask(mask, sample_type="pt")
    print(gt_point_mask.transpose((1, 2, 0)).shape)
    gt_point_mask = Image.fromarray(gt_point_mask.transpose((1, 2, 0)))
    gt_point_mask.save(os.path.join(save_dir, "gt_point_mask.png"))



if __name__ == "__main__":
    main()
    