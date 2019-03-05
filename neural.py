import sys

import TypedFlow.typedflow_rts as tyf
import tensorflow as tf
import numpy as np
from model import mkModel
import os
import itertools

import pickle

# print(val_x.shape)
# print(val_y.shape)
# print(len(vocab.items()))
# print(train_x)
# print(train_y)

data = pickle.load(open('data/dump/all.pickle', 'rb'))
vocab = pickle.load(open('data/dump/vocab.pickle','rb'))
v = {}
for x,y in vocab.items():
    v[y] = x

train_x = np.concatenate((data['p']['train_x'],data['i']['train_x'] ))
val_x =   np.concatenate((data['p']['val_x']  ,data['i']['val_x']  ))
train_y = np.concatenate((data['p']['train_y'],data['i']['train_y']))
val_y =   np.concatenate((data['p']['val_y']  ,data['i']['val_y']  ))

test_x =  data['p']['test_x']
test_y =  data['p']['test_y']


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

def outcome(y, y_):
    if y_:
        if y_ == y: 
            return 'tp'
        else:
            return 'fp'
    else:
        if y_ == y:
            return 'tn'
        else:
            return 'fn'


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
        texts = []
        for xi,yi,yi_ in zip(test_x,gold_y,pred_y):
            start = np.nonzero(xi)[0][0]
            o = outcome(yi, yi_)
            s = " ".join([v[x] for x in xi[start:]])
            texts.append((o,s))
        tp = [d[1] for d in texts if d[0] == 'tp']
        tn = [d[1] for d in texts if d[0] == 'tn']
        fp = [d[1] for d in texts if d[0] == 'fp']
        fn = [d[1] for d in texts if d[0] == 'fn']
        open("error-analysis/{}/tp.txt".format(name), "w").writelines("%s\n" % l for l in tp)
        open("error-analysis/{}/tn.txt".format(name), "w").writelines("%s\n" % l for l in tn)
        open("error-analysis/{}/fp.txt".format(name), "w").writelines("%s\n" % l for l in fp)
        open("error-analysis/{}/fn.txt".format(name), "w").writelines("%s\n" % l for l in fn)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])





## Local Variables:
## python-shell-interpreter: "nix-shell"
## python-shell-interpreter-args: "--run python"
## End:
    
    

