import json
import numpy as np
import os
# Turns all the json files in a folder into numpy arrays
def json_to_nparray():
    for filename in os.listdir('Dataset_tokenized_BPE'):
        if filename.endswith('.json'):
            with open('Dataset_tokenized_BPE/' + filename) as f:
                data = json.load(f)
                np.save('Dataset_tokenized_BPE/' + filename[:-5], np.array(data))
            continue
        else:
            continue
        
# json_to_nparray()

# open .npy files from a folder and concatenate them into one numpy array
def npy_to_nparray():
    data = []
    for filename in os.listdir('Dataset_tokenized_BPE'):
        if filename.endswith('.npy'):
            data.append(np.load('Dataset_tokenized_BPE/' + filename, allow_pickle=True))
            continue
        else:
            continue
    return data

data = npy_to_nparray()
print(len(data))

ldata = np.concatenate(data, axis=0)

print(type(ldata))
