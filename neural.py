import sys

import TypedFlow.typedflow_rts as tyf
import tensorflow as tf
import numpy as np
import os
import itertools
import random

from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

from tqdm import tqdm
import pickle

def words_to_ints(ws, vocab):
    maxvalue = max(vocab.values())
    for w in ws:
        if w not in vocab:
            maxvalue += 1
            vocab[w] = maxvalue
    xs = [vocab[x] for x in ws]
    return xs, vocab

data = {'train':{},'val':{},'test':{}}
data['train']['pos'] = open('data/i_train.txt').readlines()
data['train']['neg'] = open('data/ni_train.txt').readlines()
data['val']['pos'] = open('data/i_val.txt').readlines()
data['val']['neg'] = open('data/ni_val.txt').readlines()
data['test']['pos'] = open('data/i_test.txt').readlines()
data['test']['neg'] = open('data/ni_test.txt').readlines()

vocab = {"@@@@@":0}
def data_to_array(dt,vocab):
    ys = []
    xs = []
    for l in data[dt]['pos']:
        words = tknzr.tokenize(l.strip())
        ws,vocab = words_to_ints(words, vocab)
        ys.append(1)
        xs.append(words)
    for l in data[dt]['neg']:
        words = tknzr.tokenize(l.strip())
        ws,vocab = words_to_ints(words, vocab)
        ys.append(0)
        xs.append(words)
    return xs,ys,vocab

train_x, train_y,vocab = data_to_array('train',vocab)
val_x, val_y, vocab = data_to_array('val',vocab)
test_x, test_y, _ = data_to_array('test',vocab)

maxlen = max([len(max(train_x, key=len)),len(max(val_x, key=len)),len(max(test_x, key=len))])
print("Max. length:", maxlen)
train_x = pad_sequences(train_x, value=0, maxlen=maxlen)
val_x = pad_sequences(val_x, value=0, maxlen=maxlen)
test_x = pad_sequences(test_x, value=0, maxlen=maxlen)
train_y = np.array(train_y, dtype=np.int32)
val_y = np.array(val_y, dtype=np.int32)
test_y = np.array(test_y, dtype=np.int32)


def train_generator(batch_size):
    for batch_index in range(int(len(train_x) / batch_size)):
        x = []
        y = []
        x += train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
        y += train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
        yield {"x":np.array(x),"y":np.array(y)}     

def val_generator(batch_size):
    for batch_index in range(int(len(val_x)/batch_size)):
        x = []
        y = []
        x += val_x[batch_index * batch_size:(batch_index + 1) * batch_size]
        y += val_y[batch_index * batch_size:(batch_index + 1) * batch_size]
        yield {"x":np.array(x),"y":np.array(y)}

# comment out if you don't have CUDA
tyf.cuda_use_one_free_device()

model = mkModel(tf.train.AdamOptimizer(1e-4))
sess = tf.Session()
saver = tf.train.Saver()

tyf.initialize_params(sess,model)
tyf.train(sess,model,train_generator,val_generator, epochs=30, 
          callbacks=[tyf.Save(sess,saver,"/scratch/checkpoints_vlad/model_CNN.ckpt"), 
                     tyf.StopWhenValidationGetsWorse(3)])





## Local Variables:
## python-shell-interpreter: "nix-shell"
## python-shell-interpreter-args: "--run python"
## End:
    
    

