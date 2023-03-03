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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(50, 512)
print (pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
