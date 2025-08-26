import os
import argparse
import pickle
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras, FoVOrthographicCameras
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import glob
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
import cv2
from rvt.utils.rlbench_utils import COLOSSEUM_15_TASKS

ZFAR = {
    "front": 4.5,
    "left_shoulder": 3.2,
    "right_shoulder": 3.2,
    "wrist": 3.5,
}

FOV = {
    "front": 40,
    "left_shoulder": 40,
    "right_shoulder": 40,
    "wrist": 60,
}


def adjust_brightness(image, brightness_factor):
    """
    Adjusts the brightness of an RGB image using the HSV color space.

    Parameters:
        image (numpy.ndarray): The input RGB image with shape (3, 128, 128).
        brightness_factor (float): The factor to adjust the brightness.
                                   Should be greater than 0, where 1 means no change.

    Returns:
        numpy.ndarray: The brightness-adjusted RGB image.
    """
    # Convert the image to uint8 format if it's not
    image = image.astype(np.uint8)

    # Convert the image from RGB to HSV
    hsv_image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2HSV)

    # Adjust the brightness by modifying the V channel
    hsv_image[..., 2] = cv2.multiply(hsv_image[..., 2], brightness_factor)

    # Convert the image back to RGB
    adjusted_rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Transpose the image back to the original shape (3, 128, 128)
    adjusted_rgb_image = adjusted_rgb_image.transpose(2, 0, 1)

    return adjusted_rgb_image

def add_gt_traj(obs_history, traj_datafolder, 
                task_name, variation_number, 
                episode, mask_type, 
                step, log_dir,
                directory_type="variation_specific"):
    """
    Add gt trajectory in inference.
    """
    if directory_type == "variation_specific":
        variation_dir_name = f'variation{variation_number}'
    elif directory_type == "all_variations":
        variation_dir_name = "all_variations"

    variation_dir = os.path.join(traj_datafolder, task_name, variation_dir_name)
    if mask_type == "predicted_traj":
        # from mvt.utils import ForkedPdb
        # ForkedPdb().set_trace()
        traj_filename = f"{episode:06d}.png"
        traj_filepath = os.path.join(traj_datafolder, task_name, traj_filename)
        traj_img = np.array(Image.open(traj_filepath)).transpose(2, 0, 1) # 3, 128, 128
        # from mvt.utils import ForkedPdb
        # ForkedPdb().set_trace()
    else:
        if mask_type in ["gt_traj", "resampled_gt_traj", "gt_traj_with_binary_grasp_traj"]:
            traj_filename = f'episode{episode}_gt_traj.png'
        elif mask_type == "resampled_gt_traj_with_action":
            traj_filename = f'episode{episode}_gt_traj_with_action.png'
        elif mask_type == "noisy_gt_traj" or mask_type == "noisy_gt_traj_with_binary_grasp_traj":
            traj_filename = f'episode{episode}_noisy_gt_traj.png'
        elif mask_type == "gt_keypose_traj":
            traj_filename = f'episode{episode}_gt_keypose_traj.png'
        elif mask_type == "resampled_grasping_traj":
            traj_filename = f'episode{episode}_grasping_traj.png'
        elif mask_type == "resampled_grasping_traj_with_action":
            traj_filename = f'episode{episode}_grasping_traj_with_action.png'
        traj_filepath = os.path.join(variation_dir, 'episodes', traj_filename)
        traj_img = np.array(Image.open(traj_filepath)).transpose(2, 0, 1) # 3, 128, 128
    
    # from mvt.utils import ForkedPdb
    # ForkedPdb().set_trace()
    # print("traj_filename: ", traj_filepath)

    # NOTE: adjust brightness
    # brightness_factor = 2.0
    # traj_img = adjust_brightness(traj_img, brightness_factor)

    if mask_type == "gt_traj_with_binary_grasp_traj" or mask_type == "noisy_gt_traj_with_binary_grasp_traj":
        keypose_traj_filename = f"episode{episode}_gt_keypose_traj.png"
        keypose_traj_filepath = os.path.join(variation_dir, 'episodes', keypose_traj_filename)
        keypose_traj = np.array(Image.open(keypose_traj_filepath)).transpose((2, 0, 1)) # (H, W, 3) => (3, H, W)
        binary_keypose_traj = (np.sum(keypose_traj, axis=0) > 0).astype(int) # (3, H, W) => (H, W) 
        binary_keypose_traj = np.expand_dims(binary_keypose_traj, axis=0) # (H, W) => (1, H, W)

    # log_dir = "/srl_mvt/rvt2-vlm/rvt/tmp/eval_traj_visual"
    # save_dir = os.path.join(log_dir, "visual", task_name, variation_dir_name, 'episodes', f'episode{episode}')
    save_dir = os.path.join(log_dir, "visual", task_name,  f"episode{episode}")

    os.makedirs(save_dir, exist_ok=True)

    # print("generate visualization may affect evaluation speed")

    for key in obs_history.keys():
        if key.endswith("rgb"):
            # NOTE: assuming evaluate with front view only.
            assert key == "front_rgb"
            rgb_image = obs_history[key][0]

            if mask_type == "gt_traj_with_binary_grasp_traj" or mask_type == "noisy_gt_traj_with_binary_grasp_traj":
                obs_history[key][0] = np.concatenate((rgb_image, traj_img, binary_keypose_traj), axis=0) # 7, 128, 128
            else:
                # mask_type in ["gt_traj", "noisy_gt_traj", "predicted_traj"] etc
                obs_history[key][0] = np.concatenate((rgb_image, traj_img), axis=0) # 6, 128, 128

            # save_path = "/srl_mvt/rvt2-vlm/rvt/tmp/eval_traj_visual"
            # print("concatenate shape: ", obs_history[key][0].shape)
            # Image.fromarray(traj_img.transpose((1,2,0)).astype(np.uint8)).save(os.path.join(save_dir, f'step{step}_traj.png'))
            Image.fromarray(rgb_image.transpose((1,2,0)).astype(np.uint8)).save(os.path.join(save_dir, f'{step}.png'))

    return obs_history


def add_gt_keypose(obs_history, traj_datafolder, task_name, variation_number, episode, mask_type, step, log_dir):
    """
    Add gt keypose w/o gripper action changed in inference.
    """
    variation_dir = os.path.join(traj_datafolder, task_name, f'variation{variation_number}')
    
    # load gt_traj, gt_keypose, and gt_gripper_changed(if necessary)
    gt_traj_filename = f'episode{episode}_gt_traj.png' 
    gt_traj_filepath = os.path.join(variation_dir, 'episodes', gt_traj_filename)
    gt_traj_img = np.array(Image.open(gt_traj_filepath)).transpose(2, 0, 1) # 3, 128, 128

    gt_keypose_filename = f'episode{episode}_gt_keypose.png'
    gt_keypose_filepath = os.path.join(variation_dir, 'episodes', gt_keypose_filename)
    gt_keypose_img = np.array(Image.open(gt_keypose_filepath)).transpose(2, 0, 1) # 3, 128, 128
    gt_keypose_img = gt_keypose_img[0:1, :, :] # 1, 128, 128

    if mask_type == "gt_keypose_with_gripper_changed":
        gt_gripper_changed_filename = f'episode{episode}_gt_gripper_changed.png' 
        gt_gripper_changed_filepath = os.path.join(variation_dir, 'episodes', gt_gripper_changed_filename)
        gt_gripper_changed_img = np.array(Image.open(gt_gripper_changed_filepath)).transpose(2, 0, 1) # 3, 128, 128
        gt_gripper_changed_img = gt_gripper_changed_img[0:1, :, :] # 1, 128, 128

    # TESTING
    # save_dir = os.path.join(log_dir, task_name, f'variation{variation_number}', 'episodes', f'episode{episode}')
    # os.makedirs(save_dir, exist_ok=True)

    for key in obs_history.keys():
        if key.endswith("rgb"):
            # NOTE: assuming evaluate with front view only.
            assert key == "front_rgb"
            rgb_image = obs_history[key][0]

            if mask_type == "gt_keypose":
                obs_history[key][0] = np.concatenate((rgb_image, gt_traj_img, gt_keypose_img), axis=0) # 3+3+1, 128, 128
            elif mask_type == "gt_keypose_with_gripper_changed":
                obs_history[key][0] = np.concatenate((rgb_image, gt_traj_img, gt_keypose_img, gt_gripper_changed_img), axis=0) # 3+3+1+1, 128, 128
            
            # print("concatenate shape: ", obs_history[key][0].shape)
            # Image.fromarray(traj_img.transpose((1,2,0)).astype(np.uint8)).save(os.path.join(save_dir, f'step{step}_traj.png'))
            # Image.fromarray(rgb_image.transpose((1,2,0)).astype(np.uint8)).save(os.path.join(save_dir, f'step{step}_{key}.png')) 

    return obs_history

def get_keypoint_from_replay(replay_root_dir, start_index, num_replays, task, save_dir):
    raise NotImplementedError("Incompatible implementation with latest version.")
    for idx in range(start_index, start_index+num_replays):
        replay_filepath = os.path.join(replay_root_dir, task, f'{idx}.replay')

        with open(replay_filepath, "rb") as file:
            obs = pickle.load(file)

        dyn_cam_info = get_dyn_cam_info(obs)
        pt = torch.from_numpy(obs["gripper_pose"][0:3]).unsqueeze(0).unsqueeze(0).float()
        assert pt.shape == (1, 1, 3), f"pt.shape={pt.shape}"

        pt_img = get_pt_loc_on_img(pt=pt, dyn_cam_info=dyn_cam_info).squeeze().detach().numpy()

        rgb_img = obs["front_rgb"]
        draw_point_and_save(numpy_rgb_array=rgb_img, point=pt_img, save_dir=save_dir, filename=f'{idx}.png')
        print("2d waypoint: ", pt_img)

def get_cam_intr(cam="front"):
    fov_cam = FoVPerspectiveCameras()
    cam_intr = fov_cam.compute_projection_matrix(
        znear=0.01,
        zfar=ZFAR[cam],
        fov=FOV[cam],
        aspect_ratio=1.0,
        degrees=True,
    )
    return cam_intr

def get_dyn_cam_info(obs):
    dyn_cam_info = []
    R = []
    T = []
    scale = []
    K = []

    extr = torch.from_numpy(obs["front_camera_extrinsics"]).unsqueeze(0).unsqueeze(0)
    assert extr.shape == (1, 1, 4, 4), f"extr.shape={extr.shape}"
    extr = extr[0]
    _R = extr[0:1, 0:3, 0:3]
    _T = extr[0:1, 0:3, 3]
    _T = (-_T[0] @ _R[0]).unsqueeze(0)
    _scale = torch.ones(_T.shape)
    _K = get_cam_intr(cam="front")

    # R, T, K, scale
    R.append(_R)
    T.append(_T)
    scale.append(_scale)
    K.append(_K)
    R = torch.cat(R, 0).float()
    T = torch.cat(T, 0).float()
    scale = torch.cat(scale, 0).float()

    if K[0] is not None:
        K = torch.cat(K, 0).float()


    dyn_cam_info.append((R, T, scale, K))

    return dyn_cam_info

def _get_dyn_cam(_dyn_cam_info, pers=True, device="cpu"):
    """
    :param dyn_cam_info: tuple of (R, T, scale, K) where R is array of shape
        (num_dyn_cam, 3, 3), T (num_dyn_cam, 3), scale (num_dyn_cam) and K
        (num-dyn_cam, 4, 4)
    """
    R, T, scale, K = _dyn_cam_info


    assert len(scale.shape) == 2
    assert len(R.shape) == len(T.shape) + 1 == 3
    assert T.shape[0] == R.shape[0] == scale.shape[0]
    assert R.shape[1] == R.shape[2] == T.shape[1] == 3
    assert (K is None) or ((len(K.shape) == 3) and (K.shape == (R.shape[0], 4, 4)))
    

    if pers:
        dyn_cam = PerspectiveCameras(device=device, R=R, T=T, K=K)
    else:
        dyn_cam = FoVOrthographicCameras(
            device=device, R=R, T=T, znear=0.01, scale_xyz=scale, K=K
        )

    return dyn_cam
    
def get_pt_loc_on_img(pt, dyn_cam_info=None, image_size=(128, 128), device="cpu"):
    assert len(pt.shape) == 3
    assert pt.shape[-1] == 3
    bs, np = pt.shape[0:2]
    assert not dyn_cam_info is None
    assert (dyn_cam_info is None) or (
        isinstance(dyn_cam_info, (list, tuple))
        and isinstance(dyn_cam_info[0], tuple)
    ), dyn_cam_info

    pt_img = []

    assert pt.shape[0] == len(dyn_cam_info)
    dyn_pt_img = []
    for _pt, _dyn_cam_info in zip(pt, dyn_cam_info):
        dyn_cam = _get_dyn_cam(_dyn_cam_info, device=device)
        # (num_cam, np, 2)

        # import pdb; pdb.set_trace()
        _pt_scr = dyn_cam.transform_points_screen(
            _pt, image_size=image_size
        )[..., 0:2]
        if len(dyn_cam) == 1:
            _pt_scr = _pt_scr.unsqueeze(0)

        _pt_scr = torch.transpose(_pt_scr, 0, 1)
        # transform from camera screen to image index
        h, w = image_size
        # (np, num_img, 2)
        _dyn_pt_img = _pt_scr - torch.tensor((1 / w, 1 / h)).to(_pt_scr.device)
        dyn_pt_img.append(_dyn_pt_img.unsqueeze(0))

    # (bs, np, num_img, 2)
    dyn_pt_img = torch.cat(dyn_pt_img, 0)
    pt_img.append(dyn_pt_img)

    pt_img = torch.cat(pt_img, 2)

    return pt_img

def get_meta_info_from_demo(demo, taskname, episode_num, episode_dir, variation_number, variation_description, image_size=(128, 128)):
    """
    task: string
    variation_number: int
    lang_goal: list of instruction description(string)
    keypose_2d: (n, 2)
    keypose_3d: (n, 3)
    trajectory_2d: (N, 2)
    trajectory_3d: (N, 3)
    front_camera_extrinsics: (N, 4, 4)
    gripper_pose: (N, 7)
    gripper_open: (N,)
    keypose_indexes: list of index of keypose in the episode len=n
    episode_num: episode number
    """
    meta_info = {
        "task": taskname, 
        "variation_number": variation_number,
        "lang_goal": variation_description,
        "keypose_2d": [],
        "keypose_3d": [],
        "trajectory_2d": [],
        "trajectory_3d": [],
        "front_camera_extrinsics": [],
        "gripper_pose": [],
        "gripper_open": [],
        "keypose_indexes":[],
        "episode_num": episode_num,
        "trajectory_2d_resolution": image_size,
        "image_resolution": image_size,
        "trajectory_2d_table": [],
        "keypose_2d_table": [],
    }

    episode_keypoints = keypoint_discovery(demo)
    # print("episode length: ", len(demo))
    for idx in range(len(demo)):
        if os.path.exists(os.path.join(episode_dir, 'front_rgb', f'{idx}.png')):
            ext='png'
        elif os.path.exists(os.path.join(episode_dir, 'front_rgb', f'{idx}.jpeg')):
            ext='jpeg'
        else:
            raise KeyError
        ob = {
            "front_camera_extrinsics": demo[idx].misc["front_camera_extrinsics"], 
            "gripper_pose": demo[idx].gripper_pose,
            "front_rgb": np.array(Image.open(os.path.join(episode_dir, "front_rgb", f'{idx}.{ext}')))
            }

        dyn_cam_info = get_dyn_cam_info(ob)

        # ground truth 3d waypoints: (1, 1, 3)
        pt = torch.from_numpy(ob["gripper_pose"][0:3]).unsqueeze(0).unsqueeze(0).float() 
        assert pt.shape == (1, 1, 3), f"pt.shape={pt.shape}"

        # 2d waypoint on image frame
        pt_img = get_pt_loc_on_img(pt=pt, dyn_cam_info=dyn_cam_info, image_size=image_size).squeeze().detach().numpy()
        pt_table = pt.clone()
        pt_table[0, 0, 2] = 0.75
        pt_img_on_table = get_pt_loc_on_img(pt=pt_table, dyn_cam_info=dyn_cam_info, image_size=image_size).squeeze().detach().numpy()

        # save data in meta_info
        meta_info["trajectory_3d"].append(ob["gripper_pose"][0:3])
        meta_info["trajectory_2d"].append(pt_img)
        meta_info["trajectory_2d_table"].append(pt_img_on_table)
        meta_info["front_camera_extrinsics"].append(ob["front_camera_extrinsics"])
        meta_info["gripper_pose"].append(ob["gripper_pose"])
        meta_info["gripper_open"].append(demo[idx].gripper_open)

        if idx in episode_keypoints:
            meta_info["keypose_3d"].append(ob["gripper_pose"][0:3])
            meta_info["keypose_2d"].append(pt_img)
            meta_info["keypose_2d_table"].append(pt_img_on_table)

    meta_info["keypose_indexes"] = episode_keypoints
    meta_info["front_camera_extrinsics"] = np.stack(meta_info["front_camera_extrinsics"], axis=0)
    meta_info["keypose_2d"] = np.stack(meta_info["keypose_2d"], axis=0)
    meta_info["keypose_3d"] = np.stack(meta_info["keypose_3d"], axis=0)
    meta_info["trajectory_2d"] = np.stack(meta_info["trajectory_2d"], axis=0)
    meta_info["trajectory_3d"] = np.stack(meta_info["trajectory_3d"], axis=0)
    meta_info["gripper_pose"] = np.stack(meta_info["gripper_pose"], axis=0)
    meta_info["gripper_open"] = np.stack(meta_info["gripper_open"], axis=0)
    meta_info["trajectory_2d_table"] = np.stack(meta_info["trajectory_2d_table"], axis=0)
    meta_info["keypose_2d_table"] = np.stack(meta_info["keypose_2d_table"], axis=0)

    return meta_info

def get_meta_info_from_demo_old(demo, taskname, episode_num, episode_dir, variation_number, variation_description, image_size=(128, 128)):
    """
    task: string
    variation_number: int
    lang_goal: list of instruction description(string)
    keypose_2d: (n, 2)
    keypose_3d: (n, 3)
    trajectory_2d: (N, 2)
    trajectory_3d: (N, 3)
    front_camera_extrinsics: (N, 4, 4)
    gripper_pose: (N, 7)
    gripper_open: (N,)
    keypose_indexes: list of index of keypose in the episode len=n
    episode_num: episode number
    """
    meta_info = {
        "task": taskname, 
        "variation_number": variation_number,
        "lang_goal": variation_description,
        "keypose_2d": [],
        "keypose_3d": [],
        "trajectory_2d": [],
        "trajectory_3d": [],
        "front_camera_extrinsics": [],
        "gripper_pose": [],
        "gripper_open": [],
        "keypose_indexes":[],
        "episode_num": episode_num,
        "trajectory_2d_resolution": image_size,
        "image_resolution": image_size
    }

    episode_keypoints = keypoint_discovery(demo)
    # print("episode length: ", len(demo))
    for idx in range(len(demo)):
        if os.path.exists(os.path.join(episode_dir, 'front_rgb', f'{idx}.png')):
            ext='png'
        elif os.path.exists(os.path.join(episode_dir, 'front_rgb', f'{idx}.jpeg')):
            ext='jpeg'
        else:
            raise KeyError
        ob = {
            "front_camera_extrinsics": demo[idx].misc["front_camera_extrinsics"], 
            "gripper_pose": demo[idx].gripper_pose,
            "front_rgb": np.array(Image.open(os.path.join(episode_dir, "front_rgb", f'{idx}.{ext}')))
            }

        dyn_cam_info = get_dyn_cam_info(ob)

        # ground truth 3d waypoints: (1, 1, 3)
        pt = torch.from_numpy(ob["gripper_pose"][0:3]).unsqueeze(0).unsqueeze(0).float() 
        assert pt.shape == (1, 1, 3), f"pt.shape={pt.shape}"

        # 2d waypoint on image frame
        pt_img = get_pt_loc_on_img(pt=pt, dyn_cam_info=dyn_cam_info, image_size=image_size).squeeze().detach().numpy()

        # save data in meta_info
        meta_info["trajectory_3d"].append(ob["gripper_pose"][0:3])
        meta_info["trajectory_2d"].append(pt_img)
        meta_info["front_camera_extrinsics"].append(ob["front_camera_extrinsics"])
        meta_info["gripper_pose"].append(ob["gripper_pose"])
        meta_info["gripper_open"].append(demo[idx].gripper_open)

        if idx in episode_keypoints:
            meta_info["keypose_3d"].append(ob["gripper_pose"][0:3])
            meta_info["keypose_2d"].append(pt_img)

    meta_info["keypose_indexes"] = episode_keypoints
    meta_info["front_camera_extrinsics"] = np.stack(meta_info["front_camera_extrinsics"], axis=0)
    meta_info["keypose_2d"] = np.stack(meta_info["keypose_2d"], axis=0)
    meta_info["keypose_3d"] = np.stack(meta_info["keypose_3d"], axis=0)
    meta_info["trajectory_2d"] = np.stack(meta_info["trajectory_2d"], axis=0)
    meta_info["trajectory_3d"] = np.stack(meta_info["trajectory_3d"], axis=0)
    meta_info["gripper_pose"] = np.stack(meta_info["gripper_pose"], axis=0)
    meta_info["gripper_open"] = np.stack(meta_info["gripper_open"], axis=0)

    return meta_info


def get_representation_from_demo(demo, episode_dir, image_size=(128, 128), representation="gt_traj"):
    # extract trajectory, keypose, or keypose of gripper action changed.
    pt_imgs = []
    obs = []

    if representation == "gt_traj":
        for idx in range(len(demo)):
            rgb_filepath = os.path.join(episode_dir, "front_rgb", f'{idx}.png')
            if not os.path.exists(rgb_filepath):
                rgb_filepath = os.path.join(episode_dir, "front_rgb", f'{idx}.jpeg')
            ob = {
                "front_camera_extrinsics": demo[idx].misc["front_camera_extrinsics"], 
                "gripper_pose": demo[idx].gripper_pose,
                "front_rgb": np.array(Image.open(rgb_filepath))
                }
            obs.append(ob)
            dyn_cam_info = get_dyn_cam_info(ob)

            # ground truth 3d waypoints: (1, 1, 3)
            pt = torch.from_numpy(ob["gripper_pose"][0:3]).unsqueeze(0).unsqueeze(0).float() 
            assert pt.shape == (1, 1, 3), f"pt.shape={pt.shape}"

            # 2d waypoint on image frame
            pt_img = get_pt_loc_on_img(pt=pt, dyn_cam_info=dyn_cam_info, image_size=image_size).squeeze().detach().numpy()
            pt_imgs.append(pt_img)
    elif representation == "gt_keypose":
        episode_keypoints = keypoint_discovery(demo)
        for idx in episode_keypoints:
            ob = {
                "front_camera_extrinsics": demo[idx].misc["front_camera_extrinsics"], 
                "gripper_pose": demo[idx].gripper_pose,
                "front_rgb": np.array(Image.open(os.path.join(episode_dir, "front_rgb", f'{idx}.png')))
                }
            obs.append(ob)
            dyn_cam_info = get_dyn_cam_info(ob)

            # ground truth 3d waypoints: (1, 1, 3)
            pt = torch.from_numpy(ob["gripper_pose"][0:3]).unsqueeze(0).unsqueeze(0).float() 
            assert pt.shape == (1, 1, 3), f"pt.shape={pt.shape}"

            # 2d waypoint on image frame
            pt_img = get_pt_loc_on_img(pt=pt, dyn_cam_info=dyn_cam_info, image_size=image_size).squeeze().detach().numpy()
            pt_imgs.append(pt_img)
    elif representation == "gt_gripper_changed":
        # DEBUGGING
        # print("episode dir: ", episode_dir)

        prev_gripper_open = demo[0].gripper_open 
        for idx in range(len(demo)):
            # print("idx: ", idx, "gripper open: ", demo[idx].gripper_open)
            # only select waypoint when gripper state changed.
            curr_gripper_open = demo[idx].gripper_open
            if prev_gripper_open == curr_gripper_open:
                continue

            prev_gripper_open = curr_gripper_open
            ob = {
                "front_camera_extrinsics": demo[idx].misc["front_camera_extrinsics"], 
                "gripper_pose": demo[idx].gripper_pose,
                "front_rgb": np.array(Image.open(os.path.join(episode_dir, "front_rgb", f'{idx}.png')))
                }
            obs.append(ob)
            dyn_cam_info = get_dyn_cam_info(ob)

            # ground truth 3d waypoints: (1, 1, 3)
            pt = torch.from_numpy(ob["gripper_pose"][0:3]).unsqueeze(0).unsqueeze(0).float() 
            assert pt.shape == (1, 1, 3), f"pt.shape={pt.shape}"

            # 2d waypoint on image frame
            pt_img = get_pt_loc_on_img(pt=pt, dyn_cam_info=dyn_cam_info, image_size=image_size).squeeze().detach().numpy()
            pt_imgs.append(pt_img)
    elif representation == "gt_keypose_traj":
        prev_gripper_open = demo[0].gripper_open 
        for idx in range(len(demo)):
            # print("idx: ", idx, "gripper open: ", demo[idx].gripper_open)
            # only select waypoint when gripper state changed.
            curr_gripper_open = demo[idx].gripper_open
            gripper_touch_forces = np.abs(demo[idx].gripper_touch_forces[:3]) # forces

            # gripper touch forces must not detect force from the obj
            if curr_gripper_open and np.sum(gripper_touch_forces < 0.01) > 0:
                continue

            # if prev_gripper_open == curr_gripper_open:
            #     continue
            
            

            prev_gripper_open = curr_gripper_open
            ob = {
                "front_camera_extrinsics": demo[idx].misc["front_camera_extrinsics"], 
                "gripper_pose": demo[idx].gripper_pose,
                "front_rgb": np.array(Image.open(os.path.join(episode_dir, "front_rgb", f'{idx}.png')))
                }
            obs.append(ob)
            dyn_cam_info = get_dyn_cam_info(ob)

            # ground truth 3d waypoints: (1, 1, 3)
            pt = torch.from_numpy(ob["gripper_pose"][0:3]).unsqueeze(0).unsqueeze(0).float() 
            assert pt.shape == (1, 1, 3), f"pt.shape={pt.shape}"

            # 2d waypoint on image frame
            pt_img = get_pt_loc_on_img(pt=pt, dyn_cam_info=dyn_cam_info, image_size=image_size).squeeze().detach().numpy()
            pt_imgs.append(pt_img)
    else:
        raise NotImplementedError

    if len(obs) == 0 and (representation == "gt_gripper_changed" or representation == "gt_keypose_traj"):
        print("no gripper action changes in this episode.")
        reference_rgb_img = None
    else:
        reference_rgb_img = obs[0]["front_rgb"]
    return pt_imgs, reference_rgb_img

def save_representation(episode_num, numpy_rgb_array, points, 
                        save_dir, point_radius=1, 
                        add_noise=False, debug_visualize=False, 
                        image_size=(128, 128), representation="gt_traj"
                        ):
    # if numpy_rgb_array.shape != (128, 128, 3) or numpy_rgb_array.shape != (512, 512, 3):
    #     numpy_rgb_array = numpy_rgb_array.astype('uint8').transpose((1, 2, 0))
    # else:
    #     numpy_rgb_array = numpy_rgb_array.astype('uint8')
    if not add_noise:
        filename = f'episode{episode_num}_{representation}.png'  
    else: 
        filename = f'episode{episode_num}_noisy_{representation}.png'  

    # for edge case where not waypoint found in the episode.
    if isinstance(numpy_rgb_array, np.ndarray):
        image = Image.fromarray(numpy_rgb_array, 'RGB')
    else:
        h, w = image_size
        image = np.zeros((h, w, 3))
    mask = Image.fromarray((np.zeros_like(image) * 255).astype('uint8'), 'RGB')

    if debug_visualize:
        draw = ImageDraw.Draw(image) # draw on rgb image.
        output_img = image
    else:
        draw = ImageDraw.Draw(mask)  # draw on black background.
        output_img = mask

    color_indices = np.linspace(0, 1, len(points))
    for i, point in enumerate(points):
        # NOTE: draw trajectory with gradient color;
        #       draw keypose or gripper action changed points (gripper closed) with white color instead.
        if representation == "gt_traj" or representation == "gt_keypose_traj":
            color = tuple((255 * np.array(plt.cm.jet(color_indices[i]))[:3]).astype(int))  # Convert colormap to RGB
        elif representation == "gt_keypose" or representation == "gt_gripper_changed":
            color = (255, 255, 255) if not debug_visualize else (0, 255, 0)

        # Draw the point as a circle
        if add_noise:
            noise = np.random.normal(0, 1, 2)
            point = (point[0] + noise[0], point[1] + noise[1])

        draw.ellipse([(point[0]-point_radius, point[1]-point_radius), 
                    (point[0]+point_radius, point[1]+point_radius)], 
                    fill=color, outline=color)    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    output_img.save(save_path)
    print(f"Image saved to {save_path}")

def get_representation_from_raw_data(data_root_dir, 
                                     start_episode, 
                                     num_episodes, 
                                     tasks, 
                                     save_dir, 
                                     add_noise=False, 
                                     debug_visualize=False, 
                                     directory_type="all_variations", 
                                     image_size=(128, 128),
                                     representation="gt_traj"
                                    ):
    assert isinstance(tasks, list)
    
    if debug_visualize:
        save_dir = save_dir.replace(save_dir.split("/")[-1], save_dir.split("/")[-1]+"_debug")
        os.makedirs(save_dir, exist_ok=True)
        
    
    if directory_type == "all_variations":
        for task in tasks:
            for episode_num in range(start_episode, start_episode+num_episodes):
                variation_dir = "all_variations/episodes"
                episode_dir = os.path.join(data_root_dir, task, variation_dir, f'episode{episode_num}')

                with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), "rb") as file:
                    demo = pickle.load(file)

                pt_imgs, reference_rgb_img = get_representation_from_demo(demo=demo, episode_dir=episode_dir, image_size=image_size, representation=representation)
                save_episode_dir = os.path.join(save_dir, task, variation_dir)
                save_representation(episode_num=episode_num,
                                    numpy_rgb_array=reference_rgb_img,
                                    points=pt_imgs,
                                    point_radius=1,
                                    add_noise=add_noise,
                                    debug_visualize=debug_visualize,
                                    image_size=image_size,
                                    representation=representation,
                                    save_dir=save_episode_dir,
                                    )

    elif directory_type == "variation_specific":
        print("[WARNING] variation specific data directory doesn't support start_episode and num_episodes arguments.")
        for task in tasks:
            variation_files = glob.glob(os.path.join(data_root_dir, task, 'variation*'))
            variation_files = sorted(variation_files, key=lambda x: int(os.path.basename(x)[9:]))

            
            for variation_idx in range(len(variation_files)):
                variation_dir = variation_files[variation_idx]
                episode_files = glob.glob(os.path.join(variation_dir, "episodes", "episode*"))
                episode_files = sorted(episode_files, key=lambda x: int(os.path.basename(x)[7:]))

                for episode_idx in range(len(episode_files)):
                    episode_dir = episode_files[episode_idx]
                    with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), "rb") as file:
                        demo = pickle.load(file)

                    pt_imgs, reference_rgb_img = get_representation_from_demo(demo=demo, episode_dir=episode_dir, image_size=image_size, representation=representation)
                    save_episode_dir = os.path.join(save_dir, task, f'variation{variation_idx}', "episodes")
                    save_representation(episode_num=episode_idx,
                                        numpy_rgb_array=reference_rgb_img,
                                        points=pt_imgs,
                                        point_radius=1,
                                        add_noise=add_noise,
                                        debug_visualize=debug_visualize,
                                        image_size=image_size,
                                        representation=representation,
                                        save_dir=save_episode_dir,
                                        )

    else:
        raise NotImplementedError

def generate_episode_meta_info(data_root_dir, 
                                start_episode, 
                                num_episodes, 
                                tasks, 
                                save_dir, 
                                add_noise=False, 
                                debug_visualize=False, 
                                directory_type="all_variations", 
                                image_size=(128, 128)):
    if directory_type == "all_variations":
        for task in tasks:
            for episode_num in range(start_episode, start_episode+num_episodes):
                variation_dir = "all_variations/episodes"
                episode_dir = os.path.join(data_root_dir, task, variation_dir, f'episode{episode_num}')

                with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), "rb") as file:
                    demo = pickle.load(file)

                with open(os.path.join(episode_dir, 'variation_number.pkl'), "rb") as file:
                    variation_number = pickle.load(file)

                with open(os.path.join(episode_dir, 'variation_descriptions.pkl'), "rb") as file:
                    variation_description = pickle.load(file)

                meta_info = get_meta_info_from_demo(demo=demo,
                                                    taskname=task,
                                                    episode_num=episode_num,
                                                    variation_number=variation_number,
                                                    variation_description=variation_description,
                                                    episode_dir=episode_dir, 
                                                    image_size=image_size)
                
                # # TESTING
                for k, v in meta_info.items():
                    if isinstance(v, list):
                        print(k, v)
                    elif isinstance(v, int):
                        print(k, v)
                    elif isinstance(v, str):
                        print(k, v)
                    elif isinstance(v, np.ndarray):
                        print(k, v.shape)
                    elif isinstance(v, tuple):
                        print(k, v)
                    else:
                        raise NotImplementedError
                print("-"*50)
                # import pdb;pdb.set_trace()

                # save
                save_episode_dir = os.path.join(save_dir, task, variation_dir)
                os.makedirs(save_episode_dir, exist_ok=True)
                save_filepath = os.path.join(save_episode_dir, f'episode{episode_num}_meta_info.pkl')
                with open(save_filepath, 'wb') as file:
                    pickle.dump(meta_info, file)
                    print(f"Saved to {save_filepath}")

    elif directory_type == "variation_specific":
        for task in tasks:
            variation_dirs = sorted(glob.glob(os.path.join(data_root_dir, task, 'variation*')), key=lambda x: int(os.path.basename(x)[9:]))
            for variation_dir in variation_dirs:
                episodes_dirs = sorted(glob.glob(os.path.join(variation_dir, "episodes", "episode*")), key=lambda x: int(os.path.basename(x)[7:]))
                episodes_dirs = episodes_dirs[start_episode:start_episode+num_episodes]
                for i, episode_dir in enumerate(episodes_dirs):
                    with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), "rb") as file:
                        demo = pickle.load(file)

                    variation_number = int(os.path.basename(variation_dir)[9:])
                    episode_num = int(os.path.basename(episode_dir)[7:])

                    with open(os.path.join(variation_dir, 'variation_descriptions.pkl'), "rb") as file:
                        variation_description = pickle.load(file)

                    meta_info = get_meta_info_from_demo(demo=demo,
                                                        taskname=task,
                                                        episode_num=episode_num,
                                                        variation_number=variation_number,
                                                        variation_description=variation_description,
                                                        episode_dir=episode_dir, 
                                                        image_size=image_size)
                    
                    save_episode_dir = episode_dir.replace(data_root_dir, save_dir)
                    save_episode_dir = "/".join(save_episode_dir.split("/")[:-1])
                    os.makedirs(save_episode_dir, exist_ok=True)
                    save_filepath = os.path.join(save_episode_dir, f'episode{episode_num}_meta_info.pkl')
                    with open(save_filepath, 'wb') as file:
                        pickle.dump(meta_info, file)
                        print(f"Saved to {save_filepath}")

                
    else:
        raise NotImplementedError

SELECTED_TASKS = ['close_jar','light_bulb_in', 'stack_blocks', 'stack_cups']

ALL_TASKS = [
    'close_jar', 'meat_off_grill', 'place_shape_in_shape_sorter', 'put_groceries_in_cupboard', 'reach_and_drag', 'stack_cups',
    'insert_onto_square_peg', 'open_drawer', 'place_wine_at_rack_location', 'put_item_in_drawer', 'slide_block_to_color_target',
    'sweep_to_dustpan_of_size', 'light_bulb_in', 'place_cups', 'push_buttons', 'put_money_in_safe', 'stack_blocks', 'turn_tap'
]

def main():
    raw_data = True

    if raw_data:
        parser = argparse.ArgumentParser()

        # mandoo rvt2 raw data dir: /home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128
        # test rvt2 data dir: /home/nil/manipulation/RVT2/rvt/test_raw_data_representation
        # mandoo act3d raw data dir: /home/nil/manipulation/act3d-chained-diffuser/datasets/raw/18_peract_tasks_train
        # test act3d data dir: /home/nil/manipulation/RVT2/rvt/test_variation_specific_representation

        # ngc raw data dir:  /dataset/data/data/train
        # gt traj train dir: /srl_mvt/rvt2-vlm/rvt/data/rvt2_raw_128_gt_traj
        # ngc eval data dir: /srl_mvt/nil/act3d-chained-diffuser/datasets/raw_variations_exp
        # gt traj eval dir:  /srl_mvt/rvt2-vlm/rvt/data/act3d_raw_variations_exp_gt_traj

        parser.add_argument("--data_root_dir", type=str, default="/home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128")
        parser.add_argument("--tasks", type=str, default="all") 
        parser.add_argument("--start_episode", type=int, default=0)
        parser.add_argument("--num_episodes", type=int, default=100) 
        parser.add_argument("--save_dir", type=str, default="/home/nil/manipulation/RVT2/rvt/test_raw_data_representation")
        parser.add_argument("--add_noise", action="store_true")
        parser.add_argument("--debug_visualize", action="store_true")
        parser.add_argument("--image_size", type=int, default=128)
        # flag for process "all_variations/" or "variation{idx}/" directory 
        parser.add_argument("--directory_type", type=str, default="all_variations")  # either "all_variations" or "variation_specific"
        parser.add_argument("--representation", type=str, default="gt_traj") # {gt_traj, gt_keypose, gt_gripper_changed}
        parser.add_argument("--store_meta_info", action="store_true")
        args = parser.parse_args()

        if args.tasks == "all":
            args.tasks = ALL_TASKS
        elif args.tasks == "colosseum_15":
            args.tasks = COLOSSEUM_15_TASKS
        else:
            args.tasks = args.tasks.split(",")

        args.image_size = (args.image_size, args.image_size)

        if args.store_meta_info:
            generate_episode_meta_info(
                data_root_dir=args.data_root_dir,
                start_episode=args.start_episode,
                num_episodes=args.num_episodes,
                tasks=args.tasks,
                save_dir=args.save_dir,
                add_noise=args.add_noise,
                debug_visualize=args.debug_visualize,
                directory_type=args.directory_type,
                image_size=args.image_size                
            )
        else:
            get_representation_from_raw_data(
                data_root_dir=args.data_root_dir,
                start_episode=args.start_episode,
                num_episodes=args.num_episodes,
                tasks=args.tasks,
                save_dir=args.save_dir,
                add_noise=args.add_noise,
                debug_visualize=args.debug_visualize,
                directory_type=args.directory_type,
                image_size=args.image_size,
                representation=args.representation
            )           
    else:
        raise NotImplementedError("Incomplete Implementation.")
    
        parser = argparse.ArgumentParser()

        parser.add_argument("--replay_root_dir", type=str, default="/home/nil/manipulation/RVT2/rvt/replay/128x128/replay_train")
        parser.add_argument("--task", type=str, default="light_bulb_in")
        parser.add_argument("--start_index", type=int, default=1598)
        parser.add_argument("--num_replays", type=int, default=1) 
        parser.add_argument("--save_dir", type=str, default="/home/nil/manipulation/RVT2/rvt/test_keypoint")
        args = parser.parse_args()

        get_keypoint_from_replay(
            replay_root_dir=args.replay_root_dir,
            start_index=args.start_index,
            num_replays=int(args.num_replays),
            task=args.task,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()