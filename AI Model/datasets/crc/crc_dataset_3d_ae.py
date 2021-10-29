"""
Created by Wang Han on 2019/6/13 12:41.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
from os.path import join

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CRCDataset3DAE(Dataset):
    def __init__(self, data_root, sample_csv, transforms=None):
        self.data_root = data_root
        self.records = pd.read_csv(sample_csv, dtype='str')
        self.transforms = transforms
        print(">>> The number of records is {}".format(len(self.records)))

    def _resolve_record(self, record):
        img_path = join(self.data_root, '{}_crc_image.npy'.format(record.case_id))
        img = np.load(img_path)
        return img

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        img = self._resolve_record(record)
        if self.transforms is not None:
            img = self.transforms(img)
        sample = {
            'image': img,
            'sample_name': record.case_id
        }
        return sample
