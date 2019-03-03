
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

import pickle

def words_to_ints(ws, vocab):
    maxvalue = max(vocab.values())
    for w in ws:
        if w not in vocab:
            maxvalue += 1
            vocab[w] = maxvalue
    xs = [vocab[x] for x in ws]
    return xs, vocab

data = {'i':{'train':{},'val':{},'test':{}},
        'p':{'train':{},'val':{},'test':{}},
        'm':{'train':{},'val':{},'test':{}}}
data['i']['train']['pos'] = open('data/i_train.txt').readlines()  
data['i']['train']['neg'] = open('data/ni_train.txt').readlines() 
data['i']['val']['pos'] =   open('data/i_val.txt').readlines()    
data['i']['val']['neg'] =   open('data/ni_val.txt').readlines()   
data['i']['test']['pos'] =  open('data/i_test.txt').readlines()
data['i']['test']['neg'] =  open('data/ni_test.txt').readlines()
data['p']['train']['pos'] = open('data/p_train.txt').readlines()  
data['p']['train']['neg'] = open('data/np_train.txt').readlines() 
data['p']['val']['pos'] =   open('data/p_val.txt').readlines()    
data['p']['val']['neg'] =   open('data/np_val.txt').readlines()   
data['p']['test']['pos'] =  open('data/p_test.txt').readlines()
data['p']['test']['neg'] =  open('data/np_test.txt').readlines()
data['m']['train']['pos'] = open('data/m_train.txt').readlines()  
data['m']['train']['neg'] = open('data/nm_train.txt').readlines() 
data['m']['val']['pos'] =   open('data/m_val.txt').readlines()    
data['m']['val']['neg'] =   open('data/nm_val.txt').readlines()   
data['m']['test']['pos'] =  open('data/m_test.txt').readlines()
data['m']['test']['neg'] =  open('data/nm_test.txt').readlines()


vocab = {"@@@@@":0}
intdata = {'i':{},
           'p':{},
           'm':{}}
def data_to_array(s,dt,vocab,char=False):
    ys = []
    xs = []
    for l in data[s][dt]['pos']:
        words = tknzr.tokenize(l.strip())
        if char:
            words = list(" ".join(words).lower())
        else:
            words = " ".join(words).lower().split(" ")
        ws,vocab = words_to_ints(words, vocab)
        ys.append(1)
        xs.append(ws)
    for l in data[s][dt]['neg']:
        words = tknzr.tokenize(l.strip())
        if char:
            words = list(" ".join(words).lower())
        else:
            words = " ".join(words).lower().split(" ")
        ws,vocab = words_to_ints(words, vocab)
        ys.append(0)
        xs.append(ws)
    return xs,ys,vocab

for s in ['i','p','m']:
    intdata[s]['train_x'], intdata[s]['train_y'],vocab = data_to_array(s,'train',vocab)
    intdata[s]['val_x'], intdata[s]['val_y'],vocab = data_to_array(s,'val',vocab)
    intdata[s]['test_x'], intdata[s]['test_y'],vocab = data_to_array(s,'test',vocab)

findata = {'i':{},
           'p':{},
           'm':{}}
    
# maxlen = max([len(max(train_x, key=len)),len(max(val_x, key=len)),len(max(test_x, key=len))])
# print("Max. length:", maxlen)
print("Vocab. size", len(vocab))
maxlen=164 
for s in ['i','p','m']:
    findata[s]['train_x'] = pad_sequences(intdata[s]['train_x'], value=0, maxlen=maxlen)
    findata[s]['val_x'] = pad_sequences(intdata[s]['val_x'], value=0, maxlen=maxlen)
    findata[s]['test_x'] = pad_sequences(intdata[s]['test_x'], value=0, maxlen=maxlen)
    findata[s]['train_y'] = np.array(intdata[s]['train_y'], dtype=np.int32)
    findata[s]['val_y'] = np.array(intdata[s]['val_y'], dtype=np.int32)
    findata[s]['test_y'] = np.array(intdata[s]['test_y'], dtype=np.int32)

pickle.dump(findata,open('data/dump/all.pickle', 'wb'))
pickle.dump(vocab,open('data/dump/vocab.pickle', 'wb'))

