import os, pdb
import pickle
from collections import defaultdict


path_dir = './temp'
file_list = os.listdir(path_dir)
belief_state = defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema

for pickle_path in file_list:
    with open(f'./temp/{pickle_path}', 'rb') as f:
        item = pickle.load(f)
    pdb.set_trace()