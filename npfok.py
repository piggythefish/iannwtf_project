import numpy as np 
import os

npfok = np.array(())
for filename in os.listdir('Dataset_tokenized_BPE'):
    if filename.endswith('.npy'):
        for i in range(len(np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens'))):
            for j in range(len(np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens')[i])):
                npfok = np.append(npfok, np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens')[i][j])
                print(j, i, filename, len(np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens')), len(np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens')[i]), np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens')[i][j])
        continue
    else:
        continue