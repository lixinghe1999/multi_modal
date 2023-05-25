import os
dataset_path = '../EPIC-KITCHENS'
participants = ['P04', 'P05', 'P06']
modalities = ['flow_frames']
for participant in participants:
    for modal in modalities:
        path = os.path.join(dataset_path, participant, modal)
        tars = os.listdir(path)
        tars = [tar for tar in tars if tar.endswith('.tar')]
        for tar in tars:
            print(participant, modal, tar)
            if os.path.exists(path + '/' + tar[:-4]):
                print('already exists', tar)
                continue
            else:
                os.mkdir(path + '/' + tar[:-4])
            os.system('tar -xf ' + path + '/' + tar + ' -C ' + path + '/' + tar[:-4])