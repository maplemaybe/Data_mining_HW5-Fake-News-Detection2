# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:00:13 2020

@author: user
"""

import numpy as np
import pandas as pd

#將資料讀取出來
df_sample_submission = pd.read_csv('sample_submission (1).csv')
df_train = pd.read_csv('train.csv',delimiter='\t')
df_test = pd.read_csv('test.csv',delimiter='\t')

df_test = pd.concat([df_test,df_sample_submission['label']], axis=1)
df_test = df_test.drop(['id'], axis=1)
#df_sample_submission.head()

df_train.drop(df_train.loc[df_train['label']=='label'].index, inplace=True)
df_train = df_train.reset_index(drop=True)
df_train
df_test

#b.去除停頓詞

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
stop_words=['to', 'the','my','and','it','you','is','for','in','of','on','that','this','me','so','have','from',]
cv = CountVectorizer(stop_words=(list(stopwords)+stop_words))

list_train = [row[0] for row in df_train.itertuples(index=False, name=None)]
x_train = cv.fit_transform(list_train)
x_train.toarray()
#cv.vocabulary_

#c. 文字轉向量（Tfidf 、Ｗord2vec …等 ）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
def get_tfidf_transformer(input_content):
    # Compute the IDF values
    # input with all training data    
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True).fit(cv.fit_transform(input_content).toarray())
    return tfidf_transformer

def tfidf(input_content, tfidf_transformer):
    # Initialize CountVectorizer
    word_count_vector = cv.fit_transform(input_content).toarray()
    # tf-idf scores
    tf_idf_vector=tfidf_transformer.transform(cv.transform(input_content))
    df_tf_idf_vector=pd.DataFrame(tf_idf_vector.toarray())
    return (df_tf_idf_vector)

content_ifidf_transformer=get_tfidf_transformer(df_train['text'])
df_train = pd.concat([df_train,tfidf(df_train['text'],content_ifidf_transformer)], axis=1)

x = tfidf(df_train['text'],content_ifidf_transformer).values.tolist()
y = df_train['label'].values.tolist()

print(np.shape(x))
print(np.shape(y))

content_ifidf_transformer_test=get_tfidf_transformer(df_test['text'])
df_test = pd.concat([df_test,tfidf(df_test['text'],content_ifidf_transformer_test)], axis=1)

x_test = tfidf(df_test['text'],content_ifidf_transformer_test).values.tolist()
y_test = df_test['label'].values.tolist()

#----------建模
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

#from tensorflow.python.keras import layers
#from tensorflow.python.keras.layers import core
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense,Dropout,Activation,Flatten
#from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import SimpleRNN

modelRNN = Sequential()  #建立模型
#Embedding層將「數字list」轉換成「向量list」
modelRNN.add(Embedding(output_dim=32,   #輸出的維度是32，希望將數字list轉換為32維度的向量
     input_dim=3800,  #輸入的維度是3800，也就是我們之前建立的字典是3800字
     input_length=380)) #數字list截長補短後都是380個數字

#加入Dropout，避免overfitting
#modelRNN.add(Dropout(0.7)) 	#隨機在神經網路中放棄20%的神經元，避免overfitting

#------------建立RNN層------------
modelRNN.add(SimpleRNN(units=16))
#建立16個神經元的RNN層

#------------建立隱藏層------------
modelRNN.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
#ReLU激活函數
#modelRNN.add(Dropout(0.35))

#------------建立輸出層------------
modelRNN.add(Dense(units=1,activation='sigmoid'))
#建立一個神經元的輸出層
#Sigmoid激活函數

#------------查看模型摘要------------
modelRNN.summary()

#------------定義訓練模型------------
modelRNN.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

x_train = sequence.pad_sequences(x, maxlen=380)
y_train = np.array(y)
x_test_ = sequence.pad_sequences(x_test, maxlen=380)
y_test_ = np.array(y_test)
#x_train = x
#y_train = y
train_history = modelRNN.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)

#validation_split =0.2 設定80%訓練資料、20%驗證資料
#執行10次訓練週期
#每一批次訓練100筆資料
#verbose 顯示訓練過程


scores = modelRNN.evaluate(x_test_, y_test_,verbose=1)
scores[1]


modelRNN_2 = Sequential()  #建立模型
#Embedding層將「數字list」轉換成「向量list」
modelRNN_2.add(Embedding(output_dim=32,   #輸出的維度是32，希望將數字list轉換為32維度的向量
     input_dim=3800,  #輸入的維度是3800，也就是我們之前建立的字典是3800字
     input_length=380)) #數字list截長補短後都是380個數字

#加入Dropout，避免overfitting
modelRNN_2.add(Dropout(0.35)) 	#隨機在神經網路中放棄35%的神經元，避免overfitting

#------------建立RNN層------------
modelRNN_2.add(SimpleRNN(units=16))
#建立16個神經元的RNN層

#------------建立隱藏層------------
modelRNN_2.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
#ReLU激活函數
modelRNN_2.add(Dropout(0.35))

#------------建立輸出層------------
modelRNN_2.add(Dense(units=1,activation='sigmoid'))
#建立一個神經元的輸出層
#Sigmoid激活函數
#------------查看模型摘要------------
modelRNN_2.summary()

#------------定義訓練模型------------
modelRNN_2.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

x_train = sequence.pad_sequences(x, maxlen=380)
y_train = np.array(y)
x_test_ = sequence.pad_sequences(x_test, maxlen=380)
y_test_ = np.array(y_test)
#x_train = x
#y_train = y
train_history_2 = modelRNN_2.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)

#validation_split =0.2 設定80%訓練資料、20%驗證資料
#執行10次訓練週期
#每一批次訓練100筆資料
#verbose 顯示訓練過程


scores_2 = modelRNN_2.evaluate(x_test_, y_test_,verbose=1)
scores_2[1]

#LSTM

#前處理皆與RNN同，這裡跳過
#匯入模組
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense,Dropout,Activation,Flatten
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM

modelLSTM = Sequential() #建立模型
modelLSTM.add(Embedding(output_dim=32,input_dim=3800,input_length=380)) 

#輸出的維度是32，希望將數字list轉換為32維度的向量
#輸入的維度是3800，也就是我們之前建立的字典是3800字
#數字list截長補短後都是380個數字

modelLSTM.add(Dropout(0.2)) #隨機在神經網路中放棄20%的神經元，避免overfitting

#---------建立LSTM層---------
modelLSTM.add(LSTM(32)) 
#建立32個神經元的LSTM層

#---------建立隱藏層---------
modelLSTM.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
modelLSTM.add(Dropout(0.2))

#---------建立輸出層---------
modelLSTM.add(Dense(units=1,activation='sigmoid'))
 #建立一個神經元的輸出層

#------------查看模型摘要------------
modelLSTM.summary()

#------------定義訓練模型------------
modelLSTM.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

x_train = sequence.pad_sequences(x, maxlen=380)
y_train = np.array(y)
x_test_ = sequence.pad_sequences(x_test, maxlen=380)
y_test_ = np.array(y_test)
#x_train = x
#y_train = y
train_history_LSTM = modelLSTM.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)

#validation_split =0.2 設定80%訓練資料、20%驗證資料
#執行10次訓練週期
#每一批次訓練100筆資料
#verbose 顯示訓練過程


scores_LSTM= modelLSTM.evaluate(x_test_, y_test_,verbose=1)
scores_LSTM[1]

modelLSTM_2 = Sequential() #建立模型
modelLSTM_2.add(Embedding(output_dim=32,input_dim=3800,input_length=380)) 

#輸出的維度是32，希望將數字list轉換為32維度的向量
#輸入的維度是3800，也就是我們之前建立的字典是3800字
#數字list截長補短後都是380個數字

modelLSTM_2.add(Dropout(0.2)) #隨機在神經網路中放棄20%的神經元，避免overfitting

#---------建立LSTM層---------
modelLSTM_2.add(LSTM(32)) 
#建立32個神經元的LSTM層

#---------建立隱藏層---------
modelLSTM_2.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
modelLSTM_2.add(Dropout(0.2))

#---------建立輸出層---------
modelLSTM_2.add(Dense(units=1,activation='sigmoid'))
 #建立一個神經元的輸出層

#------------查看模型摘要------------
modelLSTM_2.summary()

#------------定義訓練模型------------
modelLSTM_2.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 


x_train = sequence.pad_sequences(x, maxlen=380)
y_train = np.array(y)
x_test_ = sequence.pad_sequences(x_test, maxlen=380)
y_test_ = np.array(y_test)
#x_train = x
#y_train = y
train_history_LSTM_2 = modelLSTM_2.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)

#validation_split =0.2 設定80%訓練資料、20%驗證資料
#執行10次訓練週期
#每一批次訓練100筆資料
#verbose 顯示訓練過程

scores_LSTM_2= modelLSTM_2.evaluate(x_test_, y_test_,verbose=1)
scores_LSTM_2[1]




