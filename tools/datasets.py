from torch.utils.data import Dataset


class GetDataset(Dataset):
    def __init__(self, X, Log_Label):
        self.data = X
        self.Log_Label = Log_Label

    def __getitem__(self, index):
        data = self.data[index]
        log_l = self.Log_Label[index]
        return data, log_l, index

    def __len__(self):
        return len(self.Log_Label.T[0])
