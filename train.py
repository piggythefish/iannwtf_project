import tensorflow as tf
import datetime
from functions import *
from model import *

# hyperparameters
d_model = 96
dff = 384
n_heads = 8
n_layers = 6
dropout_rate = 0.1
seq_length = 2048
vocab_size = 512
batch_size = 16
epochs = 200

# filepath to data
small_data = 'data/SMALL_2M_BPE.npy'

# create dataset objects from data
train_ds, val_ds = get_ds(small_data, seq_length, batch_size)

# learning rate schedule
learning_rate = CustomSchedule(d_model)

# optimizer as defined in the paper attention is all you need
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=2e-9)

# create the model
model = Transformer(vocab_size, d_model, n_heads, dff, dropout_rate, n_layers, seq_length)

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = './checkpoints/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=100)

# create a log directory for tensorboard
logdir = f"/logs/fit/"
log_dir = logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# compile the model
model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])

# train the model
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, cp_callback])

model.save_weights(f'/models/dmodel:{d_model} dff:{dff} nheads:{n_heads} nlayers:{n_layers} dropout:{dropout_rate} epochs:{epochs} seqlen:{seq_length} batch:{batch_size} date:{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.h5')
