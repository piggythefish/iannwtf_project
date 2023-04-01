import tensorflow as tf
from miditok import REMI
import datetime
from functions import *
from model import Transformer

d_model = 96 # also embedding size
dff = 384 # inner feedforward layer dim
n_heads = 8 # number of heads in the multihead attention layer
n_layers = 6 # number of layers
dropout_rate = 0.1 # dropout rate
epochs = 200
seq_length = 2048 # length of the sequence
batch_size = 16 # batch size
vocab_size = 512
model = Transformer(vocab_size, d_model, n_heads, dff, dropout_rate, n_layers, seq_length)
model.build(input_shape=(None, seq_length))

model_1 = "models/params0.7Mdata2Mseqlen4096.h5" # small model on small dataset
model_2 = "models/params0.7Mdata2Mseqlen2048(best).h5" # small model on small dataset (bigger learning rate, best model)
model_3 = "models/params1.3Mdata2Mseqlen4096.h5" # "big" model on small dataset (heavily overfitting, worst model)

max_len = seq_length - 1
start_token = tf.constant([[1] + [0] * (max_len)], dtype=tf.int64)
model.load_weights(model_2)
tokenizer = REMI()
tokenizer.load_params('tokenizer_configs/SMALL_2M.txt')
tokenizer(generate_tokens_sampling(model, start_token, max_len), [(0, False)]).dump(f'sample_song {datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.mid')
