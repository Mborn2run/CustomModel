import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
import os
from utils import time_features
import warnings
warnings.filterwarnings('ignore')


class SeriesDataset(Dataset):
    def __init__(self, input_path, target_path, flag='train', size = None, scale = True, timeenc = 1, freq = 'h',
                 input_index = ['timestamp', 'airTemperature', 'windSpeed'], 
                 target_index = ['Fox_education_Andre']):
        super().__init__()
        # size [seq_len, label_len, pred_len],
        # seq_len: 历史数据的时间步数; label_len: 模型的训练中的预测多少个时间步数的未来数据, pred_len: 模型将生成多长的未来预测,不一定包含在训练数据中
        '''
        假设你有一组每小时的气温数据, 你想要建立一个模型来预测未来一天(24小时)的气温。在这种情况下:

        label_len 是 24, 因为你要从历史数据的末端开始,预测未来 24 小时的气温。
        pred_len 也是 24, 因为你希望模型生成未来 24 小时的气温预测。

        这里 label_len 和 pred_len 的值是相同的，因为你的预测目标是连续的 24 小时。在这种情况下，它们几乎是等效的，因为你想要模型学习如何预测未来 24 个时间步，并生成相同长度的预测。

        然而，如果你的任务是更复杂的，比如从过去 24 小时的气温数据预测未来 7 天的天气情况，那么情况就不同了：

        label_len 仍然是 24, 因为你要从历史数据的末端开始, 预测未来 24 小时的气温。
        pred_len 是 7 * 24, 因为你希望模型生成未来 7 天每小时的天气情况预测。

        在这种情况下, label_len 和 pred_len 的值不同，因为你的预测目标是连续的 7 天，每天有 24 小时，总共 7 * 24 小时。
        这两个参数的区别在于, label_len 控制你要从历史数据中截取多少时间步用于监督学习，而 pred_len 控制你要模型生成多长时间步的未来预测。
        '''
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale # 是否进行标准化
        self.timeenc = timeenc # 是否进行时间编码
        self.freq = freq # 如果要自定义时间编码，请指定时间编码的频率

        self.__read_data__(input_path, target_path, input_index, target_index)


    def __read_data__(self, input_path, target_path, input_index, target_index):
        self.scaler = StandardScaler()
        # 12 * 30 * 24 表示一年中的小时数，即 12 个月，每个月 30 天，每天 24 小时。如果self.set_type为0, 表示使用训练数据
        # 长度为一整年; 验证集是从训练数据后的第一个数据到四个月长度; 测试集是从验证集后的数据
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type] # 索引起始位置
        border2 = border2s[self.set_type] # 索引结束位置

        input = pd.read_csv(input_path).loc[:,input_index]
        target = pd.read_csv(target_path).loc[:,target_index]

        df_data = pd.concat((input, target), axis=1).iloc[:, 1:]
        time_dim = input[['timestamp']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = time_dim[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['timestamp'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestamp','date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class SeriesData_Pred(Dataset):
    def __init__(self, input_path, target_path,
                 input_index = ['timestamp', 'airTemperature', 'windSpeed'], target_index = ['Fox_education_Andre'], 
                 flag='pred', size=None, scale=True, inverse=False, timeenc=1, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.__read_data__(input_path, target_path, input_index, target_index)

    def __read_data__(self, input_path, target_path, input_index, target_index):
        self.scaler = StandardScaler()
        input = pd.read_csv(input_path).loc[12 * 30 * 24 + 8 * 30 * 24:,input_index]
        target = pd.read_csv(target_path).loc[12 * 30 * 24 + 8 * 30 * 24:,target_index]
        target = target.iloc[:-2]  # 因为数据中target多两个

        df_raw = pd.concat((input, target), axis=1)
        '''
        df_raw.columns: ['timestamp', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(target_index[0])
        cols.remove('timestamp')

        # border1 = len(df_raw) - self.seq_len
        # border2 = len(df_raw)
        border1 = 0
        border2 = self.seq_len

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
            mean = self.scaler.mean_
            std = self.scaler.scale_
        else:
            mean = 0
            std = 1
            data = df_data.values

        tmp_stamp = df_raw[['timestamp']][border1:border2]
        tmp_stamp['timestamp'] = pd.to_datetime(tmp_stamp.timestamp)
        pred_dates = pd.date_range(tmp_stamp.timestamp.values[-1], periods=self.pred_len + 1, freq=self.freq)

        data_to_save = {
            'mean':  mean[-1],
            'std': std[-1],
            'pred_dates': pred_dates[1:].values
        }
        np.save('median.npy', data_to_save)
        

        df_stamp = pd.DataFrame(columns=['timestamp'])
        df_stamp.timestamp = list(tmp_stamp.timestamp.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)