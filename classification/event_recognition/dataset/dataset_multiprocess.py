import multiprocessing
import os
def worker(participant):
    path = os.path.join(dataset_path, participant)
    dest = '../epic-kitchen/' + participant
    os.system('mv ' + path + ' ' + dest)

dataset_path = '/hdd0/EPIC-KITCHENS'
participants = ['P01', 'P02', 'P03', 'P04', 'P06', 'P07']
pool = multiprocessing.Pool(6)
pool.map(worker, participants)

           