import os
import torch


class CheckpointManager:
    def __init__(self, directory, file_name, max_to_keep=3, file_ext='pt'):
        '''
        Constructs the CheckpointManager
        Adapted from https://github.com/Ovikx/PyTorch-Checkpoint-Manager/blob/master/src/ckpt_manager/ckpt_manager.py
        Args:
            directory : String, the path the states will be saved to or loaded from
            file_name : String, the name of the saved file
            maximum : Integer, the maximum number of saves to keep; if None, all saves are kept.
            file_ext : String, the file extension of the saved file
        '''
        self.directory = directory
        if self.directory[-1] != '/':
            self.directory += '/'
        self.file_name = file_name
        self.max_to_keep = max_to_keep
        self.file_ext = file_ext

    def index_from_file(self, file_name):
        return int(file_name[len(self.file_name) + 1:-(len(self.file_ext) + 1)])

    def save(self, state, step=None):
        '''
        Saves the state dictionary

        Args:
            step : Integer, a custom index to concatenate to the end of the file name (intended for step numbers)
        '''
        dir_contents = os.listdir(self.directory)

        if len(dir_contents) > 0 and step == None:
            index = max([self.index_from_file(file_name) for file_name in dir_contents]) + 1
        else:
            index = step if step != None else 1

        save_dir = f'{self.directory}{self.file_name}_{index}.{self.file_ext}'
        torch.save(state, save_dir)

        self.purge()
        # print(f'Saved state to {save_dir}')

    def purge(self):
        if self.max_to_keep is None:
            return

        dir_contents = os.listdir(self.directory)

        if len(dir_contents) > self.max_to_keep:
            indices = sorted([self.index_from_file(v) for v in dir_contents])
            for index in indices[:len(indices) - self.max_to_keep]:
                for directory in dir_contents:
                    if f'{index}.{self.file_ext}' in directory:
                        os.remove(f'{self.directory}{directory}')

    def get_latest_checkpoint_to_restore_from(self):
        dir_contents = os.listdir(self.directory)
        for directory in dir_contents:
            if self.file_name in directory:
                max_index = max([self.index_from_file(v) for v in dir_contents])
                ckpt = f'{self.directory}{self.file_name}_{max_index}.{self.file_ext}'
                return ckpt

        return None

    # def load(self):
    #     '''
    #     Returns the state dictionary of the highest indexed file in the save directory. Returns constructor asset input if no such file exists.
    #     '''
    #     dir_contents = os.listdir(self.directory)

    #     for directory in dir_contents:
    #         if self.file_name in directory:
    #             max_index = max([self.index_from_file(v) for v in dir_contents])
    #             load_dir = f'{self.directory}{self.file_name}_{max_index}.{self.file_ext}'
    #             print(f'Loading states from {load_dir}')
    #             return torch.load(load_dir)

    #     print('Initializing fresh states')
    #     return self.assets
