
# coding: utf-8

# In[1]:

from gensim import utils
import gensim.models.doc2vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import gensim
import sys
import numpy as np
from gensim import corpora, models
import csv
import _pickle as cPickle
from sklearn.externals import joblib
import bz2
from random import shuffle
import ast
from sklearn.linear_model import LogisticRegression


# In[ ]:

with open("tokens_after_lemmas_and_rm_stopwords_1000.txt") as f:
    data = f.readlines()
    
doc2vec_data = []
for line in data:
    line = ast.literal_eval(line)
    temp = ' '.join(str(token) for token in line).replace("t shirt","t-shirt")
    doc2vec_data.append(temp)
    
File = open('doc2vec_data.txt', 'w') 
for item in doc2vec_data:
    File.write("%s\n" % item)
    
sentences=gensim.models.doc2vec.TaggedLineDocument("doc2vec_data.txt")
model = gensim.models.doc2vec.Doc2Vec(sentences,size = 200, window = 10, min_count = 5, iter = 20, workers=32)

model.save('doc2vec_model.d2v')


# In[2]:

model = Doc2Vec.load('doc2vec_model.d2v')


# In[3]:

#The full model was run on a server. This is a sample model run on 1000 lines of text. 
#See doc2vec_output.jpeg to see output of 30Million line model.

sims = model.docvecs.most_similar(99)

print(sims)


# In[ ]:

#The commands below were run on a server. See doc2vec_output.jpeg to see full outputs.

print(model.doesnt_match("halloween costume devil party  scarf".split()))
print(model.doesnt_match("black blue yellow shirt navy black green orange".split()))
print(model.doesnt_match("summer winter fall t-shirt spring hot cold".split()))
print(model.doesnt_match("straight slim fit custom regular winter".split()))


print(model.most_similar(positive=['boy', 'king'], negative=['girl']))
print(model.most_similar(positive=['blue', 'shirt'], negative=['blue']))
print(model.most_similar(positive=['calvin', 'klein'], negative=['tommy']))
print(model.most_similar(positive=['cotton', 'material'], negative=['polyester']))
print(model.most_similar(positive=['nike', 'run'], negative=['express']))



print(model.most_similar_cosmul(positive=['calvin', 'klein'], negative=['tommy']) )
print(model.most_similar_cosmul(positive=['skinny', 'jean'], negative=['large']) )
print(model.most_similar_cosmul(positive=['black', 'dress'], negative=['navy']) )
print(model.most_similar_cosmul(positive=['blue', 'coat'], negative=['yellow']) )


