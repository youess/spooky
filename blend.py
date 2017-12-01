# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


files = [
    "bagNN/sub/Markov_char_given_4charHistory.csv",
    "bagNN/sub/best_rcnn_res.csv",
    "bagNN/sub/20171117-17-43-keras_kernel_v2.csv",
    "bagNN/sub/2017-12-01_17_TextInception.csv"
]

data = [pd.read_csv(f) for f in files]

tmp = data[0]

for a in tmp.columns.tolist()[1:]:

    tmp.loc[:, a] = np.round(np.mean([d[a].values for d in data], axis=0), 4)

tmp.to_csv('bagNN/sub/blend_four_files_mean.csv', index=False)
