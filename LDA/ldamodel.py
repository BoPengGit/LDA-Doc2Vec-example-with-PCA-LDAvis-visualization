
# coding: utf-8

# In[3]:

import sys
import numpy as np
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim import corpora, models
import gensim
import csv
import _pickle as cPickle
from sklearn.externals import joblib
from string import digits
import bz2
import pyLDAvis
import pyLDAvis.gensim
'''
Thank you to Jordan Barber's Latent Dirchlet Allocation (LDA) with Python blog. Some of his code from his tutorial
is used here.  https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

Also thanks much to Ben Mabey for creating the pyLDAvis package in python that is used here as well. 
''' 


# In[ ]:

with open(sys.argv[1]) as f:
    data = f.readlines()

lines = []
for num in data:
    if type(num) != float:
        lines.append(num.replace("_"," "))

newlines = []
for line in lines:
    newlines.append(line.translate({ord(k): None for k in digits}))

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')


texts = []
for i in newlines:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    
    # add tokens to list
    texts.append(stopped_tokens)
    
#Clean data. Some tokenized words end up as just "s" as the word. 
#For example baby's could be split into 'baby', and 's'. 
i = 0
while i < 5:
    i +=1
    for sentence in texts:
        for word in sentence:
            if word  == 's':f
                sentence.remove(word)


# In[ ]:

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = gensim.models.ldamulticore.LdaMulticore(corpus, id2word=dictionary, 
                                              num_topics=200, chunksize=1000, passes=20, workers=30)


tokens_after_lemmas_and_rm_stopwords = open('tokens_after_lemmas_and_rm_stopwords.txt', 'w')
for item in texts:
    tokens_after_lemmas_and_rm_stopwords.write("%s\n" % item)
    
dictionary.save_as_text('lemmas_nostopwords_with_otherdatacleaning_dictionary_' + sys.argv[2] + '.txt')

corpora.MmCorpus.serialize('lemmas_nostopwords_corpus_'+ sys.argv[2] +'.mm', corpus)
    
joblib.dump(lda, 'ldamodel_'+ sys.argv[2]+ '.pkl')


# In[6]:

print(corpus[56])


# In[4]:

dictionary = gensim.corpora.Dictionary.load_from_text('lemmas_nostopwords_with_otherdatacleaning_dictionary_1000000.txt')
corpus = gensim.corpora.MmCorpus('lemmas_nostopwords_corpus_1000000.mm')
lda = joblib.load('ldamodel_1000000.pkl')

(lda.print_topics(num_topics=20, num_words=8))


# In[4]:

lda_vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(lda_vis)

