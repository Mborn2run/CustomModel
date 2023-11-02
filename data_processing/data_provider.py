from data_processing.data_loader import SeriesDataset, SeriesData_Pred
from torch.utils.data import DataLoader

data_dict = {
    'train': SeriesDataset,
    # 'pred' : SeriesData_Pred,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = SeriesData_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        input_path=args.input_path,
        target_path=args.target_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        # features=args.features,
        input_index=args.input_index,
        target_index=args.target_index,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
