# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:34:59 2018

@author: DELL
"""

import pandas as pd
import re,nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import os.path
from nltk.util import skipgrams


stemmer = PorterStemmer()
stopWords = set(stopwords.words('english'))
stop=['tea','late','always','affinia','anything','someone','square', 'wine','glass','half','seen','store','anywhere','kitchen','final','world','issue','share','notice','argonaut','central','left','downtown','second','grand-central','first','night','review','montreal','strip','took','look','got','open','return','said','told','bellagio','quot','hour','belleclair','algonquin','lake','laptop','marriot','michigan','god','soho','florenc','albert','baldwin','bear','drake','hotel','delhi','chicago','sheraton','then','jan','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','york','dubai']
NegationCues=["dont","never","not","nothing","nowhere","noone","none","not","havent","hasnt","hadnt","cant","couldnt","	shouldnt","wont","wouldnt","dont","doesnt","isnt","arent","aint"]

vocabulary={}
skipgramvocab={}
skipgramslist=[]

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
    result=[]
    for w in words:
        if w in skipgramslist:
            result.append(w)
    return result        
        
def tokenize(text):
    text=text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    tag=nltk.pos_tag(tokens)
    tokens= [a for (a, b) in tag if (b == 'NN') or (b == 'NNP') or (b == 'JJ') or (b == 'JJR') or (b== 'VB') or (b == 'VBD')  or (b== 'RB') ]
    text=[t for t in tokens if not t in string.punctuation and len(t) > 2]
    tokenswithoutStopwords = [w for w in text if not w in stopWords and not w in stop or w in NegationCues]  
    value=0
    stems = stem_tokens( tokenswithoutStopwords, stemmer)
    stems= getSkipgram(stems,2,1)
    for v in stems:
        if v not in skipgramvocab:
          skipgramvocab[v]=0
        else:
          value= skipgramvocab[v]  
          skipgramvocab[v]=value+1
    return stems 

def main(): 
     reviews=[]
     
     ##Extracting each reviews
     
     source = 'D:/RIT/Capstone/Capstone Data/Positive'
     for root, dirs, filenames in os.walk(source):
        for fn in filenames:
           try:
             fullpath = os.path.join(source, fn)
             f = open(fullpath) 
             l = [l for l in f.readlines() if l.splitlines()]
             for review in l:
                reviews.append(review)
             f.close()
         
           except:
              continue
          
     source = 'D:/RIT/Capstone/Capstone Data/Negative'
     for root, dirs, filenames in os.walk(source):
        for fn in filenames:
           try:
             fullpath = os.path.join(source, fn)
             f = open(fullpath) 
             l = [l for l in f.readlines() if l.splitlines()]
             for review in l:
                reviews.append(review)
             f.close()
           except:
             continue  
         
     source = 'D:/RIT/Capstone/Capstone Data/hotels/ALL'
     for root, dirs, filenames in os.walk(source):
         for fn in filenames:
          try:
           fullpath = os.path.join(source, fn)
           f = open(fullpath) 
           l = [l for l in f.readlines() if l.splitlines()]
           for review in l:
             reviews.append(review)
           f.close()
          except:
              continue 
          
     finaldata = pd.DataFrame({
     'text':reviews,
     })
     i=0
     for line in finaldata.text.tolist():
       tokenize(line)
     for key, value in skipgramvocab.items():
       value=value/len(finaldata)
       skipgramvocab[key]=value
  
     for key, value in skipgramvocab.items():
       if(value>0.05):
          skipgramslist.append(key)   
          i=i+1
     print("length",len(skipgramslist))     
     ## Save skipgramlist
     list_pickle_path = 'D:/RIT/Capstone/Capstone Data/FinalPrograms/SkipGramVocab.pkl'
     list_pickle = open(list_pickle_path, 'wb',)
     pickle.dump(skipgramslist, list_pickle)
     list_pickle.close()   

if __name__ == "__main__": main()