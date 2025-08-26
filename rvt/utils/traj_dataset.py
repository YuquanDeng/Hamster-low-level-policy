from typing import Dict
import torch
import numpy as np
import pickle, os, glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Dict, Callable, List
import copy
import random

from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ColorJitter

variation_dict = {
    'close_jar': {'all': [-1],
                  'test_25': [0, 10, 14, 16, 18],
                  'train_75': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19]},
    'light_bulb_in': {'all': [-1],
                      'test_25': [2, 3, 11, 12, 19],
                      'train_75': [0, 1, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18]},
    'stack_cups': {'all': [-1],
                   'test_25': [2, 3, 9, 16, 17],
                   'train_75': [0, 1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19]},
}
 

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, np.ndarray):
            result[key] = func(value)
        else:
            result[key] = value
    return result

class RLBenchTrajDataset():
    def __init__(self,
            tasks,
            # variation_mode='all',  # 'all', '75_train', '25_test', '50_train', '50_test''
            mode=None,
            data_path=None,
            local_data_path=None,
            server_data_path=None,
            meta_path=None,
            local_meta_path=None,
            server_meta_path=None,
            load_meta_into_memory=False,
            preprocess=None,
            resample_to=-1,
            constant_speed=False,
            variation_mode='train_75',
            image_range=10,
            impainting='to_end:0',
            traj_type='full',
            traj_image_size=128,
            replicate=1,
            phase='train',
            random_flip=False,
            random_text=False,
            random_crop = 0.,
            random_color_jitter=0.,
            variation_dir=False,
            **kwargs
            ):
        """    
        episode{i}_meta_info.pkl:
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
        self.input_params = {'tasks': tasks, 'load_meta_into_memory': load_meta_into_memory, 
                             'preprocess': preprocess, 'resample_to': resample_to, 
                             'constant_speed': constant_speed, 'variation_mode': variation_mode}
        self.input_params.update(kwargs)
        # super().__init__()
        
        self.image_range = image_range if phase=='train' else min(image_range, 10)
        self.traj_add_noise = False
        self.traj_img_size = traj_image_size
        self.preprocess=preprocess
        self.resample_to = resample_to
        self.constant_speed=constant_speed
        
        self.tasks = tasks
        self.phase = phase
        if self.tasks[0] == 'all':
            """
            close_jar               place_shape_in_shape_sorter  reach_and_drag
            insert_onto_square_peg  place_wine_at_rack_location  slide_block_to_color_target
            light_bulb_in           push_buttons                 stack_blocks
            meat_off_grill          put_groceries_in_cupboard    stack_cups
            open_drawer             put_item_in_drawer           sweep_to_dustpan_of_size
            place_cups              put_money_in_safe            turn_tap
            """
            self.tasks = ['close_jar', 'place_shape_in_shape_sorter', 'reach_and_drag',
                     'insert_onto_square_peg', 'place_wine_at_rack_location', 'slide_block_to_color_target',
                     'light_bulb_in', 'push_buttons', 'stack_blocks',
                     'meat_off_grill', 'put_groceries_in_cupboard', 'stack_cups',
                     'open_drawer', 'put_item_in_drawer', 'sweep_to_dustpan_of_size',
                     'place_cups', 'put_money_in_safe', 'turn_tap']
        elif self.tasks[0] == 'colosseum_15':
            self.tasks = ['close_box', 'close_laptop_lid', 'hockey', 
                        'wipe_desk', 'open_drawer', 'slide_block_to_target', 
                        'reach_and_drag', 'put_money_in_safe', 'place_wine_at_rack_location', 
                        'insert_onto_square_peg', 'stack_cups', 'straighten_rope', 
                        'setup_chess', 'scoop_with_spatula','meat_on_grill'
                        ]

        if data_path is None:
            assert mode is not None
            self.data_path = local_data_path if mode == 'localhost' else server_data_path
        else:
            self.data_path = data_path
        if meta_path is None:
            assert mode is not None
            self.meta_path = local_meta_path if mode == 'localhost' else server_meta_path
        else:
            self.meta_path = meta_path
        self.input_params['data_path'] = self.data_path
        self.input_params['meta_path'] = self.meta_path
        
        self.variation_dir = variation_dir
        self.traj_type = traj_type
        self.impainting = impainting
        self.load_meta_into_memory = load_meta_into_memory
        self.variation_mode = variation_mode
        self.all_meta_file = []
        self.cache = []
        self.browse_all_meta()
        self.replicate = replicate if phase=='train' else 1 # use to replicate the dataset to accerleate training
        self.random_flip = random_flip
        self.random_text = random_text
        self.random_crop = random_crop
        self.random_color_jitter = random_color_jitter
        
        
        self.transform = Compose([
            # Resize((128, 128)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def browse_all_meta(self):
        self.all_meta_file = []
        self.cache = []
        # path example: /srl_mvt/rvt2-vlm/rvt/data/rlbench_gt_traj_128/close_jar/all_variations/episodes/episode0_meta_info.pkl
        
        if not self.variation_dir:
            sub_var_dir = 'all_variations'
        else:
            sub_var_dir = "variation*"

        for task in self.tasks:
            files = [x for x in glob.glob(os.path.join(self.meta_path, f"{task}/{sub_var_dir}/episodes/episode*_meta_info.pkl"))]
            for file in files:
                if (self.variation_mode == 'all' or self.variation_mode == 'debug'):
                    if self.phase == 'train':
                        meta = self.load_single_meta(file)
                        # if meta['gripper_open'] do not contain 0, then it is invalid and skip
                        if 0 in meta['gripper_open']:
                            self.all_meta_file.append(file)
                            if self.load_meta_into_memory:
                                self.cache.append(meta)
                    else:
                        self.all_meta_file.append(file)
                        if self.load_meta_into_memory:
                            meta = self.load_single_meta(file)
                            self.cache.append(meta)
                else:
                    meta = self.load_single_meta(file)
                    if meta['variation_number'] in variation_dict[task][self.variation_mode]:
                        self.all_meta_file.append(file)
                        if self.load_meta_into_memory:
                            self.cache.append(meta)
        if self.variation_mode == 'debug':
            self.all_meta_file = self.all_meta_file[::10]
            self.cache = self.cache[::10]
                                
    def load_single_meta(self, meta_file):
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        return meta 
            
    def get_validation_dataset(self):
        raise DeprecationWarning
        val_set = copy.copy(self)
        if self.variation_mode == 'train_75':
            val_set.variation_mode = 'test_25'
        else:
            raise NotImplementedError
        val_set.image_range = 0
        val_set.browse_all_meta()
        return val_set
        
    def reshuffle(self):
        self.shuffled_index = np.random.shuffle(np.arange(len(self.all_meta_file)))
    
    def get_meta_from_index(self, idx):
        # if idx % self.__len__() == 0:
        #     self.reshuffle()
        if self.load_meta_into_memory:
            return self.cache[idx%len(self.cache)]
        else:
            return self.load_single_meta(self.all_meta_file[idx%len(self.all_meta_file)])
        
    def load_image_from_meta(self, meta):
        # print(meta)
        # if self.image_range < 0:
        #     image_range = max(len(meta['trajectory_2d'])+self.image_range, 10)
        # else:
        #     image_range = min(self.image_range, len(meta['trajectory_2d']))
        image_range = max(int(len(meta['trajectory_2d'])*self.image_range/100), 1)
        image_index = np.random.randint(image_range)
        
        # ext = 'jpeg' if '1000_eps' in self.data_path else 'png'
        #image_name = f"{meta['task']}/all_variations/episodes/episode{meta['episode_num']}/front_rgb/{image_index}.{ext}"
        
        if not self.variation_dir:
            image_name = f"{meta['task']}/all_variations/episodes/episode{meta['episode_num']}/front_rgb/{image_index}"
        else:
            image_name = f"{meta['task']}/variation{meta['variation_number']}/episodes/episode{meta['episode_num']}/front_rgb/{image_index}"
        
        image_path = os.path.join(self.data_path, image_name + '.png')
        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_path, image_name + '.jpeg')
        
        img = Image.open(image_path)
        return img, image_index

    def pad_image(self, image, traj_2dpoints):
        assert image.size==(640, 480)
        if self.padding_method == 'zero':
            new_image = Image.new("RGB", [640,640], (0, 0, 0))
            new_image.paste(image, (0, 80))
            traj_2dpoints = traj_2dpoints+np.array([[0, 80]])
        elif self.padding_method == 'crop':
            new_image = image.crop((80, 0, 560, 480))
            traj_2dpoints = traj_2dpoints-np.array([[80, 0]])
        else:
            raise NotImplementedError
        return new_image, traj_2dpoints
    
    def __len__(self) -> int:
        return len(self.all_meta_file)*self.replicate

    
    def unnormalize_traj(self, traj):
        return (traj*0.5+0.5)*self.traj_img_size
    
    def get_impainting_mask(self, image_index_new, traj_img_idx):
        traj_dim = 2
        impainting_mode, impainting_param = self.impainting.split(':')
        impainting_param = int(impainting_param)
        if impainting_mode == 'to_end':
            min_index = max(image_index_new, 1)
            max_index = max(int(len(traj_img_idx)*impainting_param/100), min_index+1)
            start_index = max(np.random.randint(min_index, max_index), 1)
            mask = np.zeros([len(traj_img_idx), traj_dim])
            mask[start_index:] = 1
        elif impainting_mode == 'clip':
            clip_length = int(len(traj_img_idx)*impainting_param/100)
            max_index = len(traj_img_idx)
            min_index = min(image_index_new, max_index-clip_length-1)
            start_index = np.random.randint(min_index, max_index-clip_length)
            mask = np.zeros([len(traj_img_idx), traj_dim])
            mask[start_index:start_index+clip_length] = 1
        elif impainting_mode == 'none':
            mask = np.ones([len(traj_img_idx), traj_dim])
            start_index = 0
        else:
            raise NotImplementedError
        return mask, start_index
    
    def _meta_to_data(self, meta):
        image, image_index = self.load_image_from_meta(meta)
        
        traj_img = np.zeros([self.traj_img_size, self.traj_img_size, 3])
        valid = True
        if 'trajectory_2d' in meta:
            if self.traj_type.startswith('grasping') and ('gripper_open' not in meta or 0 not in meta['gripper_open']):
                # grasping traj not valid in this example
                assert self.phase != 'train'
                print(f'no valid gripper_open at {meta["task"]}{meta["episode_num"]}')
                valid = False
                pass
            else:
                gripper_open = meta['gripper_open'] if 'gripper_open' in meta else None
                traj_2dpoints, _, non_start_points = self.resample_traj(meta['trajectory_2d'], image_index, gripper_open=gripper_open)
                if image.height != image.width:
                    image, traj_2dpoints = self.pad_image(image, traj_2dpoints)
                traj_2dpoints[:, 0] = traj_2dpoints[:, 0]/image.size[0]*self.traj_img_size
                traj_2dpoints[:, 1] = traj_2dpoints[:, 1]/image.size[1]*self.traj_img_size
                traj_img = self.draw_trajectory(traj_2dpoints, normalized=False, non_start_points=non_start_points)
                if self.traj_type.endswith('action') and 'gripper_open' in meta:
                    traj_img = self.draw_action_points(traj_img, meta['trajectory_2d'], gripper_open)
        else:
            assert self.phase != 'train'
            # print("no gt trajectory_2d in meta")
            pass
        
        image = np.array(image)
        traj_img = np.array(traj_img)
        if self.phase == 'train':
            temp_disable_flip = meta['task'] in ['turn_tap']
            image, traj_img = self.data_augmentation(image, traj_img, temp_disable_flip)
            
        traj_img = self.transform(traj_img)
        if type(meta['lang_goal']) == list:
            if self.random_text:
                text = np.random.choice(meta['lang_goal'])
            else:
                text = meta['lang_goal'][0]
        else:
            text = meta['lang_goal']
        data = {
            'obs': {
                'image': np.array(image), # 128, 128, 3
                'text': text, # string
            },
            'traj': traj_img,
            'meta': meta,
            'valid': valid,
        }
        return data
    # def _meta_to_data(self, meta):
    #     image, image_index = self.load_image_from_meta(meta)
    #     if self.resample_to > 0:
    #         traj_2dpoints, traj_img_idx = self.resample_traj(meta['trajectory_2d'], image_index, gripper_open=meta['gripper_open'])
    #     else:
    #         traj_2dpoints, traj_img_idx = meta['trajectory_2d']
    #     if image.height != image.width:
    #         # if image is not square, pad it to square
    #         image, traj_2dpoints = self.pad_image(image, traj_2dpoints)
    #     # print(traj_2dpoints, traj_2dpoints.shape)
    #     # traj_img = self.draw_trajctory(traj_2dpoints)
    #     traj_2dpoints = self.normalize_traj(traj_2dpoints)
    #     if type(meta['lang_goal']) == list:
    #         text = meta['lang_goal'][0]
    #     else:
    #         text = meta['lang_goal']
    #     # text = np.random.choice(meta['lang_goal'])
    #     # find image_index in traj_img_idx
    #     for image_index_at_trajectory in range(len(traj_img_idx)-1):
    #         if traj_img_idx[image_index_at_trajectory] <= image_index:
    #             break
        
    #     impainting_mask, impainting_start_idx = self.get_impainting_mask(image_index_at_trajectory, traj_img_idx)            
        
    #     # one_hot_variation = np.zeros(20)
    #     # one_hot_variation[meta['variation_number']] = 1
    #     history_traj = traj_2dpoints.copy()
    #     history_traj[impainting_mask==0, ] = 0
    #     traj_start_index = impainting_start_idx
    #     data = {
    #         'obs': {
    #             'image': np.array(image), # 128, 128, 3
    #             'text': text, # string
    #             'image_index': image_index_at_trajectory,
    #             'traj_start_index': traj_start_index, # int
    #             'history_traj': history_traj, # N, 2
    #             'impainting_mask': impainting_mask, # N
    #         },
    #         # 'traj': traj_img, # T, 3, 128, 128
    #         'traj': traj_2dpoints, # N, 2
    #         # 'cond_mask': ~impainting_mask # N
    #     }
    #     if self.preprocess is not None:
    #         image_token = self.preprocess['image'](image=image, return_tensors="pt")
    #         text_token = self.preprocess['text'](text, padding="max_length", return_tensors="pt")
    #         data['obs']['image_token'] = image_token
    #         data['obs']['text_token'] = text_token
    #     return data
            
    
    def resample_traj(self, traj, obs_image_index, gripper_open=None):
        # downsample and interperploate to 32 points
        # traj: N, 2
        if self.traj_type.startswith('full'):
            traj = traj 
            non_start_points = np.concatenate([[-1], gripper_open[1:]-gripper_open[:-1]])+1
        elif self.traj_type.startswith('grasping'):
            traj = traj[gripper_open==0]
            non_start_points = np.concatenate([[-1], gripper_open[1:]-gripper_open[:-1]])[gripper_open==0]+1
        else:
            raise NotImplementedError
        
        n = traj.shape[0]
        # constant_speed is useless
        
        dists = np.concatenate([[0], np.linalg.norm(np.diff(traj, axis=0), axis=-1)])
        dists = dists*non_start_points
        xp = np.cumsum(dists)
        # add 0 to the beginning
        # xp = np.concatenate([[0], xp])
        total_dist = np.sum(dists)
        if self.resample_to > 0:
            resample_to = self.resample_to
        else:
            resample_to = int(total_dist/self.step_size+0.5)
        if self.impainting.startswith("none"):
            traj_start = xp[obs_image_index+1]
        else:
            traj_start = 0
        x = np.linspace(traj_start, total_dist, resample_to)
        
        # print(xp, len(xp), traj.shape[0])
        resampled_traj = np.stack([np.interp(x, xp, traj[:, i]) for i in range(2)], axis=1)
        new_traj_index = np.interp(x, xp, np.arange(n))
        new_non_start_points = np.ones_like(new_traj_index)
        start_point_index = np.where(non_start_points==0)[0]
        for i in start_point_index:
            # find the point in resampled_traj which is closest to traj[i]
            dist = np.linalg.norm(resampled_traj-traj[i], axis=-1)
            new_non_start_points[np.argmin(dist)] = 0
        # print(new_non_start_points)
        return resampled_traj, new_traj_index, new_non_start_points
        
    def draw_trajectory(self, points, representation='gt_traj', normalized=False, non_start_points=None):
        # Check if points need to be normalized
        if normalized:
            points = (points + 1) / 2 * self.traj_img_size

        # Generate a linear color gradient for the trajectory
        color_indices = np.linspace(0, 1, len(points))
        image = np.zeros((self.traj_img_size, self.traj_img_size, 3), dtype=np.uint8)

        # Loop through the points to draw lines between consecutive points
        for i in range(1, len(points)):
            if non_start_points is not None:
                if non_start_points[i] < 1:
                    # print('skip start point', i)
                    continue
            # Choose the appropriate color
            if representation == "gt_traj":
                color = (255 * np.array(plt.cm.jet(color_indices[i]))[:3]).astype(int).tolist()  # Convert colormap to RGB
            else:
                raise NotImplementedError
            
            # Extract the current and previous point coordinates
            start_point = (int(points[i - 1][0]), int(points[i - 1][1]))
            end_point = (int(points[i][0]), int(points[i][1]))

            # Draw a line between the previous and current point
            cv2.line(image, start_point, end_point, color, thickness=self.traj_img_size//128)
            # plt.imshow(image)
            # plt.axis('off')
            # plt.show()

        return image
    
    def draw_action_points(self, image, traj, gripper_open):
        action_points = np.concatenate([[0], gripper_open[1:] - gripper_open[:-1]]) #-1 for close, 1 for open
        open_points = np.where(action_points==1)[0]
        close_points = np.where(action_points==-1)[0]
        for i in close_points:
            # draw a circle with radius 5 and no fill, thickness 1, red color
            cv2.circle(image, (int(traj[i][0]), int(traj[i][1])), 3*self.traj_img_size//128, (0, 0, 255), 2*self.traj_img_size//128)
        for i in open_points:
            # draw a circle with radius 5 and no fill, thickness 1, blue color
            cv2.circle(image, (int(traj[i][0]), int(traj[i][1])), 3*self.traj_img_size//128, (255, 0, 0), 2*self.traj_img_size//128)
        return image
        
    def debug_visualize(self, idx):
        meta = self.get_meta_from_index(idx)
        data = self._meta_to_data(meta)
        img = data['obs']['image']
        traj_2dpoints = data['traj']
        traj_img = self.draw_trajctory(traj_2dpoints, normalized=True)
        plt.subplot(1,3,1)
        plt.imshow(np.array(img))
        plt.title(meta['trajectory_dir'])
        plt.subplot(1,3,2)
        plt.imshow(traj_img)
        plt.subplot(1,3,3)
        plt.imshow((np.array(img)*0.5+traj_img*0.5).astype('uint8'))
        plt.title(data['obs']['text'])
        plt.show()
    
    def data_augmentation(self, image, traj_img, temp_disable_flip=False):
        if self.random_flip and not temp_disable_flip:
            if np.random.rand() > 0.5:
                # print("flip!")
                image = np.flip(image, axis=1).copy()
                traj_img = np.flip(traj_img, axis=1).copy()
            else:
                # print("not flip!")
                pass
            
        if self.random_crop>0:
            ih, iw, _ = image.shape
            th, tw, _ = traj_img.shape
            ratio_h, ratio_w = np.random.uniform(0, self.random_crop, 2)
            ih_max, iw_max = int(self.random_crop*ih), int(self.random_crop*iw)
            th_max, tw_max = int(self.random_crop*th), int(self.random_crop*tw)
            ih_start, iw_start = int(ratio_h*ih), int(ratio_w*iw)
            th_start, tw_start = int(ratio_h*th), int(ratio_w*tw)
            image = image[ih_start:ih-ih_max+ih_start, iw_start:iw-iw_max+iw_start]
            traj_img = traj_img[th_start:th-th_max+th_start, tw_start:tw-tw_max+tw_start]
            image = cv2.resize(image, (ih, iw))
            traj_img = cv2.resize(traj_img, (th, tw))
            
        if self.random_color_jitter>0:
            image = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=self.random_color_jitter)(Image.fromarray(image))
        return image, traj_img
            
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:        
        meta = self.get_meta_from_index(idx//self.replicate)
        data = self._meta_to_data(meta)
        # torch_data = dict_apply(data, torch.from_numpy)
        return data

if __name__ == "__main__":
    data_path = '/Manipulation/rvt2-vlm/rvt/data/1600860/train/'
    meta_path = '/Manipulation/rvt2-vlm/rvt/data/rlbench_gt_traj_128/'
    tasks = ['light_bulb_in']
    dataset = RLBenchTrajDataset(tasks, data_path=data_path, meta_path=meta_path, 
                                 variation_mode='train_75', mode='localhost',
                                 impainting='none:0', traj_image_size=128,
                                 traj_type='grasping', 
                                 resample_to=32, constant_speed=True, image_range=100)
    for i in range(10):
        dataset.debug_visualize(i)
    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

