import tensorflow as tf
import numpy as np
import os

def get_tokens(): # function to get all tokens from the dataset and put them in a numpy array
    data = np.array(()) # data aggregator for one piece
    piece_counter = 0 # counter for the number of pieces
    for filename in os.listdir('Dataset_tokenized_BPE'): # iterate through all files in the folder
        if filename.endswith('.npy'): # if the file is a .npy file
            data = np.append(data, 89) # append the start token
            for i in range(len(np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens'))): # iterate through all the tokens in the file
                data = np.append(data, np.load(('Dataset_tokenized_BPE/' + filename), allow_pickle=True)[()].get('tokens')[i]) # append the tokens of one piece to the numpy array
            data = np.append(data, 176) # append the end token
            piece_counter += 1 # increase the piece counter
            continue 
        else: 
            continue 
    return data, piece_counter

def split_input_target(chunk): # split the input and the target
    input_text = chunk[:-1] # input is the sequence without the last token
    target_text = chunk[1:] # target is the sequence without the first token
    return input_text, target_text


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

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)