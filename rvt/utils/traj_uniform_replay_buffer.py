from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
import os
import pickle
from os.path import join
import numpy as np
from PIL import Image


class TrajUniformReplayBuffer(UniformReplayBuffer):
    def __init__(self, 
                 trajectory_root_dir: str, 
                 lang_instruction_path:str, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_root_dir = trajectory_root_dir
        self.lang_instruction_path = lang_instruction_path
        
        if not os.path.exists(self.lang_instruction_path):
            raise FileNotFoundError(f"Language instruction file not found: {self.lang_instruction_path}")
        
        with open(self.lang_instruction_path, "rb") as file:
            self.taskname2lang_instruction = {}
            taskname2lang_instruction = pickle.load(file)
            self.taskname2lang_instruction = taskname2lang_instruction # None or Dict[taskname] = clip lang embedding.
        
        # TODO: use logger
        

    def _get_from_disk(self, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Here we fake a mini store (buffer)
        store = {store_element.name: {}
                 for store_element in self._storage_signature}
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            for i in range(start_index, end_index):
                task_replay_storage_folder = self._task_replay_storage_folders[self._index_mapping[i, 0]]
                task_index = self._index_mapping[i, 1]

                task_name = task_replay_storage_folder.split("/")[-1]
                with open(join(task_replay_storage_folder, '%d.replay' % task_index), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        # for case of training with only front view, we skip the non-front view image in the replay.
                        if k not in store.keys():
                            continue
                    
                        if k[-3:] == "rgb":
                            store[k][i] = self._add_traj(rgb_img=v, episode_idx=d['episode_idx'], task_name=task_name)
                        elif k == "lang_goal_embs": # (77, 512)
                            # use the same language for the same task
                            store[k][i] = self.taskname2lang_instruction[task_name] 
                        else:
                            store[k][i] = v # NOTE: potential bug here, should % self._replay_capacity                  
        else:
            for i in range(end_index - start_index):
                idx = (start_index + i) % self._replay_capacity
                task_replay_storage_folder = self._task_replay_storage_folders[self._index_mapping[idx, 0]]
                task_index = self._index_mapping[idx, 1]
                with open(join(task_replay_storage_folder, '%d.replay' % task_index), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        store[k][idx] = v
        return store
    
    def _add_traj(self, rgb_img, episode_idx, task_name):
        traj_filename = "episode%s_gt_traj_with_action.png" % episode_idx
        traj_filepath = join(self.trajectory_root_dir, task_name, "all_variations/episodes", traj_filename)

        traj_img = np.array(Image.open(traj_filepath)) # (H, W, 3)
        traj_img = traj_img.transpose((2, 0, 1)) # (3, H, W)
        
        rgb_traj_concat_img = np.concatenate((rgb_img, traj_img), axis=0) # (6, H, W)
        return rgb_traj_concat_img