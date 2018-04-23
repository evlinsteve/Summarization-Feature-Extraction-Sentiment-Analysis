# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:24:32 2018

@author: DELL
"""

import pickle
import os.path
import numpy as np
import pandas as pd
import re,nltk
import string
import gensim as gn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models
from numpy import array
from gensim import corpora, models
from nltk.util import skipgrams

def find(source,choice):
  
   ## Get text for topic identification, feature Extraction and sentiment analysis
   ## For all reviews in hotel
   #source='D:/RIT/Capstone/Capstone Data/Positive'
   if choice==0:
      result=findForHotel(source)  
   else :
      result=findForReview(source)  
   return result

def findForReview(source):
    lda = models.ldamodel.LdaModel.load('D:/RIT/Capstone/Capstone Data/ModelNew/'+'lda20new.saved')
    pkl_filename = "D:/RIT/Capstone/Capstone Data/FinalPrograms/models/TopicModelPluslogisticRegression20.pkl"  
    try:
      with open(pkl_filename, 'rb') as file:  
       log_model = pickle.load(file)   
    except:
       print("Cant model regression model")

##Importing dictionery for Vocabulary
       
    dictionary = corpora.dictionary.Dictionary.load('D:/RIT/Capstone/Capstone Data/FinalPrograms/models/'+'dictionary20.saved') 
    topicMapping= {0: "HoteL stay Experience", 1:"Internet/ Wifi",2:"Room,Smell,Smoke", 3: "Room Cleanliness", 4:"Water,Shower",5:"Room Characterstics",6:"Breakfast,Food",7:"Cleanliness",
    8:"Bed Comfort",9:"Service, Airport",10:"Staff Friendliness",11:"Service",12:"Location",13:"Place, Stay",14:"Booking/Management",15:"Hotel Services",16:"Value/ Worth",
    17:"Room, floor",18:"Front-Desk",19:"Will you come next time?", 20:"Location, Commute",21:"Helpful",22:"Hotel Service",23:"Pay",24:"Casino,Resort"}    
    topicID=[]
    features=[]
    featuresAll=[]          
 ## Extracting topic vector,List of top words for predicted topic and topic prediction for review
    topicVector,featureList,topicPrediction=findTopic(source,lda)
    topicID.append(topicPrediction)
    topicArray=np.reshape(array(topicVector), (1, -1))
    for wordtuple in featureList:
        l=wordtuple[1]  
        if topicPrediction == l[0]:
             features.append(dictionary.get(wordtuple[0]))
        predict=log_model.predict(topicArray)
        topicname=topicMapping[topicPrediction]          
    ## Data frame for each reviews with features and sentiments. 
    featuresAll.append(features)     
    dfreviews = pd.DataFrame({     
     'topicID': topicID ,
      'topicName':topicname,
      'features':featuresAll,
     'sentiment':predict
    })
    
    return  dfreviews
    
def findForHotel(source):

    lda = models.ldamodel.LdaModel.load('D:/RIT/Capstone/Capstone Data/ModelNew/'+'lda20new.saved')
    pkl_filename = "D:/RIT/Capstone/Capstone Data/FinalPrograms/models/TopicModelPluslogisticRegression20.pkl"  
    try:
      with open(pkl_filename, 'rb') as file:  
       log_model = pickle.load(file)   
    except:
       print("Cant model regression model")
##Importing dictionery for Vocabulary
    dictionary = corpora.dictionary.Dictionary.load('D:/RIT/Capstone/Capstone Data/FinalPrograms/models/'+'dictionary20.saved') 
    topicMapping= {0: "HoteL stay Experience", 1:"Internet/ Wifi",2:"Room,Smell,Smoke", 3: "Room Cleanliness", 4:"Water,Shower",5:"Room Characterstics",6:"Breakfast,Food",7:"Cleanliness",
    8:"Bed Comfort",9:"Service, Airport",10:"Staff Friendliness",11:"Service",12:"Location",13:"Place, Stay",14:"Booking/Management",15:"Hotel Services",16:"Value/ Worth",
    17:"Room, floor",18:"Front-Desk",19:"Will you come next time?", 20:"Location, Commute",21:"Helpful",22:"Hotel Service",23:"Pay",24:"Casino,Resort"}    
    topicID=[]
    featuresList=[]
    senti=[]
    reviewMap={}
    sentiMap={}
    topicname=[]       
    with open(source, "r") as ins:
      lines =ins.readlines()
      for line in lines:
          features=[]          
          ## Extracting topic vector,List of top words for predicted topic and topic prediction for review
          topicVector,featureList,topicPrediction=findTopic(line,lda)
          topicID.append(topicPrediction)
          topicArray=np.reshape(array(topicVector), (1, -1))
          for wordtuple in featureList:
           l=wordtuple[1]  
           if topicPrediction == l[0]:
             features.append(dictionary.get(wordtuple[0]))
          predict=log_model.predict(topicArray)
          featuresList.append(features)
          if predict==0:
              senti.append("Negative")
          else:
              senti.append("Positive")
          topicname.append(topicMapping[topicPrediction])          
          if topicPrediction not in reviewMap:
               reviewMap[topicPrediction] =features 
          else:
              reviewMap[topicPrediction].append(features)
          if topicPrediction not in sentiMap:
               sentiMap[topicPrediction]=predict 
          else:
              if(predict==0):
                 sentiMap[topicPrediction]=sentiMap[topicPrediction]-1 
              else :
                 if(predict==1):   
                   sentiMap[topicPrediction]=sentiMap[topicPrediction]+1 
    
    ## Data frame for combining all reviews of a hotel
    dfAllreviews = pd.DataFrame(columns=['topicID','topic name','features', 'sentiment']) 
    i=0
    for key, value in reviewMap.items():
        opinion=""
        newlist=[]
        valueMap={}
        print("new review")
        if isinstance(value, list):
            newlist=value
        else:    
         for l in value:
            for word in l:
                if len(word)>2:
                 if word not in valueMap:
                    valueMap[word]=0
                 else:
                    v=valueMap[word]
                    valueMap[word]=v+1            

         for w in sorted(valueMap, key=valueMap.get, reverse=True):  
           newlist.append(w)
        print(newlist)
        if sentiMap[key]>0:
            opinion="Positive"
        else:
            opinion="Negative"
        dfAllreviews.loc[i]=[key,topicMapping[key],newlist,opinion]
        i=i+1
    return dfAllreviews;
    
    
def findTopic(reviews,lda):
    terms=tokenizeReviews(reviews)
    bow = lda.id2word.doc2bow(terms) 
    topicVector, word_topics, phi_values = lda.get_document_topics(bow, per_word_topics=True)
    topicID=[x[0] for x in topicVector]
    Vector=topicVector
    topics = sorted(Vector,key=lambda x:x[1],reverse=True)
    topic=topics[0][0]
    for x in range(0,25):
        if x not in topicID:
            topicVector.append((x,0.0))
    topicVector = sorted(topicVector,key=lambda x:x[0])
    topics=[x[1] for x in topicVector]
    return topics,word_topics,topic
      
      
def tokenizeReviews(s):
    stemmer = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    stop=['tea','late','always','affinia','anything','someone','square', 'wine','glass','half','seen','store','anywhere','kitchen','final','world','issue','share','notice','argonaut','central','left','downtown','second','grand-central','first','night','review','montreal','strip','took','look','got','open','return','said','told','bellagio','quot','hour','belleclair','algonquin','lake','laptop','marriot','michigan','god','soho','florenc','albert','baldwin','bear','drake','hotel','delhi','chicago','sheraton','then','jan','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','york','dubai']
    NegationCues=["dont","never","not","nothing","nowhere","noone","none","not","havent","hasnt","hadnt","cant","couldnt","	shouldnt","wont","wouldnt","dont","doesnt","isnt","arent","aint"]
    text = re.sub("[^a-zA-Z]", " ", s)
    tokens = nltk.word_tokenize(text)
    grams=[]
    result=[]
    lista=[]
    listb=[]
    tag=nltk.pos_tag(tokens)
    tokens= [a for (a, b) in tag if (b == 'NN') or (b == 'NNP') or (b == 'JJ') or (b == 'JJR') or (b== 'VB') or (b == 'VBD')  or (b== 'RB') ]
    text=[t for t in tokens if not t in string.punctuation and len(t) > 2]
    tokenswithoutStopwords = [w for w in text if not w in stopWords and not w in stop or w in NegationCues]  
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
      

def getbigrams(words):
    words= getSkipgram(words,2,1)
    pkl_filename='D:/RIT/Capstone/Capstone Data/FinalPrograms/SkipGramVocab.pkl'
    with open(pkl_filename, 'rb') as file:  
      skipgramslist = pickle.load(file)
    result=[]
    for w in words:
        if w in skipgramslist:
            result.append(w)
    return result

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
