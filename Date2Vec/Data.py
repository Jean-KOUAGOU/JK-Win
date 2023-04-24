from torch.utils.data import Dataset
import torch
import numpy as np

class NextDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]

        #print(dates)

    def __len__(self):
        return len(self.dates)-1
    
    def __getitem__(self, idx):
        return np.array(self.dates[idx]).astype(np.float32), np.array(self.dates[idx+1]).astype(np.float32)

class TimeDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]

        #print(dates)

    def __len__(self):
        return len(self.dates)-1
    
    def __getitem__(self, idx):
        x = np.array(self.dates[idx]).astype(np.float32)
        return x, x
    
class TrainDataLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        dates = [date.split(",") for date in data['datetime']]
        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datetime = self.dates[idx]
        target = self.data.iloc[idx].values[1:].astype(float)
        features = torch.FloatTensor(datetime)
        target = torch.FloatTensor(target)
        return features, target
    
class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        dates = [date.split(",") for date in data['datetime']]
        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datetime = self.dates[idx]
        features = torch.FloatTensor(datetime)
        return features


if __name__ == "__main__":
    dt = open("dates.txt", 'r').readlines()
    dataset = NextDateDataset(dt)
    print(dataset[0])