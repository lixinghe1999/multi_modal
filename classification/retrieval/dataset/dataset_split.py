import pandas as pd 
import argparse
import h5py
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-m', '--mode', type=str, default='label')
    args = parser.parse_args()
    participants = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']
    if args.mode == 'label':
        train_list = pd.read_pickle('dataset/epic_kitchen_100/EPIC_train.pkl')
        test_list = pd.read_pickle('dataset/epic_kitchen_100/EPIC_val.pkl')
        # just split the original train and test list
        new_train_list = train_list[train_list['participant_id'].isin(participants)]
        new_test_list = test_list[test_list['participant_id'].isin(participants)]
        print(len(new_train_list), len(new_test_list))

        # split the sub-dataset independently
        new_data_list = pd.concat([new_train_list, new_test_list])
        new_train_list = new_data_list.sample(frac=0.8, random_state=42)
        new_test_list = new_data_list.drop(new_train_list.index)
        print(len(new_train_list), len(new_test_list))

        new_train_list.to_pickle('EPIC_train.pkl')
        new_test_list.to_pickle('EPIC_val.pkl')
    elif args.mode == 'imu':
        def epic_100(x):
            return len(x.split('_')[1])
        train_list = pd.read_pickle('dataset/epic_kitchen_100/EPIC_train.pkl')
        test_list = pd.read_pickle('dataset/epic_kitchen_100/EPIC_val.pkl')
        # just split the original train and test list
        new_train_list = train_list[train_list['participant_id'].isin(participants)]
        new_test_list = test_list[test_list['participant_id'].isin(participants)]

        new_train_list = new_train_list.loc[new_train_list['video_id'].apply(epic_100) >= 3]
        new_test_list = new_test_list.loc[new_test_list['video_id'].apply(epic_100) >= 3]
        print(len(new_train_list), len(new_test_list))

        # split the sub-dataset independently
        new_data_list = pd.concat([new_train_list, new_test_list])
        new_train_list = new_data_list.sample(frac=0.8, random_state=42)
        new_test_list = new_data_list.drop(new_train_list.index)
        print(len(new_train_list), len(new_test_list))

        new_train_list.to_pickle('EPIC_train_100.pkl')
        new_test_list.to_pickle('EPIC_val_100.pkl')
    else: # for audio split
        audio = h5py.File('../EPIC_audio.hdf5', 'r')
        keys_list = list(audio.keys())
        filtered_list = [keys_list[i] for i in range(len(keys_list)) if keys_list[i].split('_')[0] in participants]
        print(len(filtered_list), filtered_list)

        new_audio = h5py.File('../split_EPIC_audio.hdf5', 'w')
        for key in filtered_list:
            new_audio.create_dataset(key, data=audio[key])