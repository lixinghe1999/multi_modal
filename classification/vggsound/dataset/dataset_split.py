import pandas as pd 
train_list = pd.read_pickle('dataset/epic_kitchen_label/EPIC_train_action_labels.pkl')
test_list = pd.read_pickle('dataset/epic_kitchen_label/EPIC_val_action_labels.pkl')
participants = ['P01', 'P02', 'P03', 'P04', 'P06', 'P07']
new_train_list = train_list[train_list['participant_id'].isin(participants)]
new_test_list = test_list[test_list['participant_id'].isin(participants)]
print(len(new_train_list), len(new_test_list))
new_train_list.to_pickle('EPIC_train_action_labels.pkl')
new_test_list.to_pickle('EPIC_val_action_labels.pkl')