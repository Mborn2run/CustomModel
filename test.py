from data_processing import SeriesDataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import LSTMModel

factory_kwargs = {'device': 1, 'dtype': 2}
print({**factory_kwargs})

a= [1,2,3]
del a[0]
print(len(a))

input_path = "data/building-data-genome-project-2/data/processing/weather/Hog.csv"
target_path = "data/building-data-genome-project-2/data/processing/electricity/Hog.csv"
target_index = ['Hog_education_Jordan']
input_index = ['timestamp', 'airTemperature', 'dewTemperature', 'windSpeed']
time_set = SeriesDataset(input_path, target_path, timeenc=0, input_index=input_index, target_index=target_index)
time_loader = DataLoader(time_set, batch_size=1, shuffle=False)
model = LSTMModel(input_size=3, hidden_size=64, num_layers=2, output_size=1)