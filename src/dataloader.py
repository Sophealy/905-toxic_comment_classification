import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class ToxicDataset(Dataset):

    def __init__(self,csv_path):
        self.data = pd.read_csv(csv_path)

    def __getitem__(self,index):
        comment_id  = self.data['id'].iloc[index]
        targets = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
        y = torch.from_numpy( self.data[targets].iloc[index].values)
        y =y.type(torch.FloatTensor)

        return comment_id,y

    def __len__(self):
        return len(self.data.index)
