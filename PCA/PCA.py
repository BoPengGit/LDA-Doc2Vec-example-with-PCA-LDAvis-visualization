
# coding: utf-8

# In[1]:

import numpy as np
from gensim import corpora, models
import gensim
import csv
import _pickle as cPickle
from sklearn.externals import joblib
import bz2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
get_ipython().magic('pylab inline')


# In[2]:

lda = joblib.load('ldamodel_20_topics_1M.pkl')
dictionary = gensim.corpora.Dictionary.load_from_text('lemmas_nostopwords_with_otherdatacleaning_dictionary_1000000.txt')
corpus = gensim.corpora.MmCorpus('lemmas_nostopwords_corpus_1000000.mm')


# In[9]:

color = []
for corpus_line in corpus[:1000000]:
    sorted_topic_line = list(sorted(lda[corpus_line], key=lambda x: x[1], reverse=True))
    color.append(sorted_topic_line[0][0])
    
    
lda_output = []
for line in corpus[:1000000]:
    lda_output.append(lda[line])
    
topics_data = np.zeros(shape=(1000000,20))

for i, line in enumerate(lda_output):
    for topic_line in line:
            topics_data[i][topic_line[0]] = topic_line[1]


# In[16]:

print(topics_data[1])


# In[10]:

X_pca = PCA().fit_transform(topics_data )
figure(num=None, figsize=(18, 11), dpi=80, facecolor='w', edgecolor='k')
scatter(X_pca[:, 0], X_pca[:, 1], c=color)


# In[11]:

topics_data_norm  = stats.zscore(topics_data, axis=0)

X_pca = PCA().fit_transform(topics_data_norm )
figure(num=None, figsize=(18, 11), dpi=80, facecolor='w', edgecolor='k')
scatter(X_pca[:, 0], X_pca[:, 1], c=color)


# In[13]:

topics_data = np.zeros(shape=(10000000,20))

for i, line in enumerate(lda_output):
    for topic_line in line:
            topics_data[i][topic_line[0]] = topic_line[1]
            
topics_data_norm  = stats.zscore(topics_data, axis=0)

color = []
for line in topics_data_norm:
    color_number =[i for i, j in enumerate(line) if j == max(line)]
    color.append(int(color_number[0]))

X_pca = PCA().fit_transform(topics_data_norm )
figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
scatter(X_pca[:, 0], X_pca[:, 1], c=color)


# In[20]:

shape(topics_data[1])


# In[ ]:



