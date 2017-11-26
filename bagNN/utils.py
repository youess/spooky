# -*- coding: utf-8 -*-

import os
import pandas as pd
from datetime import datetime


def add_timestamp(text):
    tsp = datetime.timestamp()
    return tsp.strftime('%Y-%m-%d_%H') + "_" + text


def read_csv(filename, loc='./data'):
    filename = os.path.join(loc, filename)
    return pd.read_csv(filename)


def read_data(loc='./data'):
    train = read_csv('train.csv', loc)
    test = read_csv('test.csv', loc)
    return train['text'], train['author'], test['text']


def write_sub(y_predict, filename, loc='./data/sub'):
    filename = os.path.join(loc, filename)
    sub = read_csv('sample_submission.csv')
    sub.iloc[:, 1:] = y_predict
    sub.to_csv(filename, index=False, float_format='%.4f')
