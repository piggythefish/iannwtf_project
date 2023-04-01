import tensorflow as tf
import numpy as np
import json
import os

# Turns all the json files from tokenizaion in a folder into numpy arrays
def json_to_nparray(path):
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            with open(path + filename) as f:
                data = json.load(f)
                np.save(path + filename[:-5], np.array(data))
            continue
        else:
            continue

def get_tokens(path): # function to get all tokens from the dataset and put them in a numpy array
    to_shuffle = np.array(()) # to_shuffle aggregator for one piece
    data = np.array(()) # data aggregator for one piece
    piece_counter = 0
    
    for filename in os.listdir(path):
        if filename.endswith('.npy'):
            to_shuffle = np.append(to_shuffle, np.load(path + filename, allow_pickle=True)[()])
    np.random.shuffle(to_shuffle)
    
    for i in range(len(to_shuffle)):
        data = np.append(data, 1)
        for j in range(len(to_shuffle[i].get('tokens'))):
            data = np.append(data, to_shuffle[i].get('tokens')[j])
        data = np.append(data, 2)
        piece_counter += 1
        
    return data, piece_counter

# create input and target sequences
def split_input_target(chunk): # split the input and the target
    input_text = chunk[:-1] # input is the sequence without the last token
    target_text = chunk[1:] # target is the sequence without the first token
    return input_text, target_text

# usual shuffling, batching and prefetching
def make_batches(ds, batch_size):
  return ds.shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# generate the position encoding
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

# our masked loss function
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

# our masked accuracy function
def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

# function to create the dataset
def get_ds(npy_path, seq_length, batch_size):
    tokens = np.load(npy_path) # load the numpy array with all tokens
    data = tf.cast(tokens, tf.int32) # cast the tokens to int32
    n = int(0.9*len(data)) # split the data into train and validation 
    train_data = data[:n] # 90% train, 10% validation
    val_data = data[n:]  
    train_ds, val_ds = tf.data.Dataset.from_tensor_slices(train_data), tf.data.Dataset.from_tensor_slices(val_data) # create the dataset objects
    # create the input and target sequences
    train_ds, val_ds = train_ds.batch(seq_length+1, drop_remainder=True).map(split_input_target, tf.data.AUTOTUNE), val_ds.batch(seq_length+1, drop_remainder=True).map(split_input_target, tf.data.AUTOTUNE)
    # shuffle, batch and prefetch
    train_ds, val_ds = make_batches(train_ds, batch_size), make_batches(val_ds, batch_size)
    return train_ds, val_ds

# A validation method to discard MIDIs we do not want
def midi_valid(midi) -> bool:
    if any(ts.numerator != 4 for ts in midi.time_signature_changes):
        return False  # time signature different from 4/*, 4 beats per bar
    if midi.max_tick < 10 * midi.ticks_per_beat:
        return False  # this MIDI is too short
    return True

# function to generate the tokens always taking the token with the highest probability
def generate_tokens_greedy(model, start_token, max_len):
    for i in range(max_len):
        logits = model(start_token) # generate logits for the next token
        probs = tf.nn.softmax(logits, axis = -1) # get the probabilities 
        argmax = tf.argmax(probs, axis = -1) # get the token with the highest probability
        start_token_numpy = start_token.numpy() # convert the tensor to numpy array
        start_token_numpy[:, i+1] = argmax[:,i].numpy() # add the predicted token to the sequence
        start_token = tf.constant(start_token_numpy, dtype=tf.int64) # convert the numpy array to tensor
        if argmax[:,i].numpy() == 176: # stop when the end token is predicted
            break
    return start_token

# function to generate the tokens sampling from the logits
def generate_tokens_sampling(model, start_token, max_len):
    for i in range(max_len):
        logits = model(start_token) # generate logits for the next token
        start_token_numpy = start_token.numpy() # convert the tensor to numpy array
        start_token_numpy[:, i+1] = tf.random.categorical(logits[:,i], num_samples=1).numpy() # sample a token from the logits and add it to the sequence
        start_token = tf.constant(start_token_numpy, dtype=tf.int64) # convert the numpy array to tensor
        if start_token_numpy[:, i+1] == 176: # stop when the end token is predicted
            break
    return start_token