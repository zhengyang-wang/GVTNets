import os
import pandas as pd
import numpy as np

from czireader import CziReader

class CziDataset(object):
    """Dataset for CZI files."""

    def __init__(self, dataframe: pd.DataFrame = None,
                    raw_dataset_dir: str = None,
                    path_csv: str = None, 
                    transform_source = None,
                    transform_target = None):
        
        self.raw_dataset_dir = raw_dataset_dir

        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
            
        self.transform_source = transform_source
        self.transform_target = transform_target
        
        assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        has_target = not np.isnan(element['channel_target'])
        # print(self.raw_dataset_dir)
        # print(os.path.join(self.raw_dataset_dir, element['path_czi']))
        czi = CziReader(os.path.join(self.raw_dataset_dir, element['path_czi']))
        
        im_out = list()
        im_out.append(czi.get_volume(element['channel_signal']))
        if has_target:
            im_out.append(czi.get_volume(element['channel_target']))
        
        if self.transform_source is not None:
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])

        if has_target and self.transform_target is not None:
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])
                
        im_out = [im.astype(np.float32) for im in im_out]
        
        return im_out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()