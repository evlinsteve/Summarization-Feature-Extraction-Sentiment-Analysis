# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:49:12 2018

@author: DELL
"""

import re,nltk
import numpy as np
import gensim as gn
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models
import os.path
from nltk.util import skipgrams
import pickle

ps = PorterStemmer()
stemmer=PorterStemmer()
stopWords = set(stopwords.words('english'))

NegationCues=["dont","never","not","nothing","nowhere","noone","none","not","havent","hasnt","hadnt","cant","couldnt","	shouldnt","wont","wouldnt","dont","doesnt","isnt","arent","aint"]


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def getSkipgram(sentence,n,k):
    grams=[]
    for i in range(k+1):
     words = [w for w in skipgrams(sentence, n, i)]
     grams = grams+words
    return grams  

def getbigrams(words):
    words= getSkipgram(words,2,1)
    pkl_filename='source directory'
    with open(pkl_filename, 'rb') as file:  
      skipgramslist = pickle.load(file)
    result=[]
    for w in words:
        if w in skipgramslist:
            result.append(w)
    return result

def tokenizeReviews(s):
    
    text = re.sub("[^a-zA-Z]", " ", s)
    tokens = nltk.tokenize.TextTilingTokenizer()
    tokens = nltk.word_tokenize(text)
    grams=[]
    result=[]
    lista=[]
    listb=[]
    tag=nltk.pos_tag(tokens)
    tokens= [a for (a, b) in tag if (b == 'NN') or (b == 'NNP') or (b == 'JJ') or (b == 'JJR') or (b== 'VB') or (b == 'VBD')  or (b== 'RB') ]
    text=[t for t in tokens if not t in string.punctuation and len(t) > 2]
    tokenswithoutStopwords = [w for w in text if not w in stopWords  or w in NegationCues]  
    stems = stem_tokens( tokenswithoutStopwords, stemmer)    
    grams=getbigrams(stems)
    for a,b in grams:
      lista.append(a)
      listb.append(b)
    for a,b in grams:
        temp=a+"-"+b
        result.append(temp)     
    for s in stems:
        if s not in lista or s not in listb:
          result.append(s)      
    return result



def main():  
  termMatrix=[]  
  ##getting review datas
  source = 'source directory'
  for root, dirs, filenames in os.walk(source):
    for fn in filenames:
        try:
          fullpath = os.path.join(source, fn)
          f = open(fullpath) 
          l = [l for l in f.readlines() if l.splitlines()]
          for review in l:
            termMatrix.append(tokenizeReviews(review))
          f.close()
         
        except:
             continue
  
  arr = np.array(termMatrix)
  
  ## Corpus Dictionary
  dictionary = corpora.Dictionary(arr)
  dictionary.filter_extremes(no_below=27)
  dictionary.save('source directory')
  
  ## Term map to access words in dictionery with wordIDS
  termsMap=dictionary.token2id;
  dictionaryMap= dict((v,k) for k,v in termsMap.items())
  list_pickle_path = 'source directory'
  list_pickle = open(list_pickle_path, 'wb',)
  pickle.dump(dictionaryMap, list_pickle)
  list_pickle.close()
  
  corpus = [dictionary.doc2bow(text) for text in arr]
  try: 
    np.save('source directory', arr)
  except:
   print("couldnt save")
  ldamodel = gn.models.ldamodel.LdaModel(corpus, num_topics=20, id2word =dictionary, passes=20)
  print(ldamodel.show_topics(num_topics=20,num_words=25))
  ldamodel.save('source directory')

  

 
if __name__ == "__main__": main()