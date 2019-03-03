import sys

import TypedFlow.typedflow_rts as tyf
import tensorflow as tf
import numpy as np
from model import mkModel
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
data['train']['pos'] = open('data/m_train.txt').readlines()  + open('data/p_train.txt').readlines()   + open('data/i_train.txt').readlines() 
data['train']['neg'] = open('data/nm_train.txt').readlines() + open('data/np_train.txt').readlines()  + open('data/ni_train.txt').readlines()
data['val']['pos'] =   open('data/m_val.txt').readlines()    + open('data/p_val.txt').readlines()     + open('data/i_val.txt').readlines()   
data['val']['neg'] =   open('data/nm_val.txt').readlines()   + open('data/np_val.txt').readlines()    + open('data/ni_val.txt').readlines()  
data['test']['pos'] =  open('data/m_test.txt').readlines()
data['test']['neg'] =  open('data/nm_test.txt').readlines()

vocab = {"@@@@@":0}
def data_to_array(dt,vocab,char=False):
    ys = []
    xs = []
    for l in data[dt]['pos']:
        words = tknzr.tokenize(l.strip())
        if char:
            words = list(" ".join(words).lower())
        else:
            words = " ".join(words).lower().split(" ")
        ws,vocab = words_to_ints(words, vocab)
        ys.append(1)
        xs.append(ws)
    for l in data[dt]['neg']:
        words = tknzr.tokenize(l.strip())
        if char:
            words = list(" ".join(words).lower())
        else:
            words = " ".join(words).lower().split(" ")
        ws,vocab = words_to_ints(words, vocab)
        ys.append(0)
        xs.append(ws)
    return xs,ys,vocab

train_x, train_y,vocab = data_to_array('train',vocab)
val_x, val_y, vocab = data_to_array('val',vocab)
test_x, test_y, vocab = data_to_array('test',vocab)

maxlen = max([len(max(train_x, key=len)),len(max(val_x, key=len)),len(max(test_x, key=len))])
print("Max. length:", maxlen)
print("Vocab. size", len(vocab))
maxlen=164 #60
train_x = pad_sequences(train_x, value=0, maxlen=maxlen)
val_x = pad_sequences(val_x, value=0, maxlen=maxlen)
test_x = pad_sequences(test_x, value=0, maxlen=maxlen)
train_y = np.array(train_y, dtype=np.int32)
val_y = np.array(val_y, dtype=np.int32)
test_y = np.array(test_y, dtype=np.int32)

# print(val_x.shape)
# print(val_y.shape)
# print(len(vocab.items()))
# print(train_x)
# print(train_y)


def train_generator(batch_size):
    for batch_index in range(int(len(train_x) / batch_size)):
        # x = []
        # y = []
        x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
        y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
        yield {"x":np.array(x),"y":np.array(y)}     

def val_generator(batch_size):
    for batch_index in range(int(len(val_x)/batch_size)):
        # x = []
        # y = []
        x = val_x[batch_index * batch_size:(batch_index + 1) * batch_size]
        y = val_y[batch_index * batch_size:(batch_index + 1) * batch_size]
        yield {"x":np.array(x),"y":np.array(y)}

def metrics(cm):
    """computes acc, prec, recall, f1 for laughter class
    """
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[0][0]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return {"accuracy" : (tp+tn)/(tp+tn+fp+fn),
            "precision": precision,
            "recall"   : recall,
            "f_1"      : 2*precision*recall/(precision+recall)}

# comment out if you don't have CUDA
tyf.cuda_use_one_free_device()

def main(mode,name):
    if mode == 'train':
        model = mkModel(tf.train.AdamOptimizer(1e-4))
        sess = tf.Session()
        saver = tf.train.Saver()

        tyf.initialize_params(sess,model)
        tyf.train(sess,model,train_generator,val_generator, epochs=500, 
                  callbacks=[tyf.Save(sess,saver,"/scratch/checkpoints_vlad/incongr/model_{}.ckpt".format(name)), 
                             tyf.StopWhenValidationGetsWorse()])
    else:
        tf.reset_default_graph()
        sess = tf.Session()
        model = mkModel(tf.train.AdamOptimizer(1e-4))
        saver = tf.train.Saver()
        saver.restore(sess, "/scratch/checkpoints_vlad/incongr/model_{}.ckpt".format(name))
        predictions = tyf.predict(sess,model, {"x":np.array(test_x)})
        gold_y = test_y
        pred_y = [np.argmax(p) for p in predictions]
        tfcm = tf.confusion_matrix(gold_y,pred_y)
        with tf.Session():
            cm = tf.Tensor.eval(tfcm,feed_dict=None, session=None)
            print('Confusion Matrix: \n', tf.Tensor.eval(tfcm,feed_dict=None, session=None))
            print(metrics(cm))
            print("/".join(["{:.3f}".format(v) for v in metrics(cm).values()]))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])





## Local Variables:
## python-shell-interpreter: "nix-shell"
## python-shell-interpreter-args: "--run python"
## End:
    
    

