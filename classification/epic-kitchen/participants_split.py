import pandas as pd 
train_list = pd.read_pickle('train_val/EPIC_train_action_labels.pkl')
test_list = pd.read_pickle('train_val/EPIC_val_action_labels.pkl')
participants = ['P02']
new_train_list = train_list[train_list['participant_id'].isin(participants)]
new_test_list = test_list[test_list['participant_id'].isin(participants)]
print(len(new_train_list), len(new_test_list))
new_train_list.to_pickle('EPIC_train_action_labels.pkl')
new_test_list.to_pickle('EPIC_val_action_labels.pkl')