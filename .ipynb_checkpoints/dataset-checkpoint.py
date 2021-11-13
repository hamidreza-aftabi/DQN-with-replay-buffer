import torch
import pickle
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        state_tensor = torch.tensor(item[0]).float()
        
        if item[1] == 0:
            action_tensor= torch.tensor([1,0]).float()
        
        else:
            action_tensor= torch.tensor([0,1]).float()
        
        state_action_dic = {'state': state_tensor, 'action': action_tensor}
        return state_action_dic
    
        raise NotImplementedError()
