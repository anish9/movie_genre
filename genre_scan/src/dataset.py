import numpy as np
import pandas as pd
from .model_ops import *
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
import os

dataset = pd.read_csv(os.getcwd()+"/genre_scan/src/data/movie_dataset.csv")
dataset.dropna(axis=0,inplace=True)
dataset['genre'] = dataset['genre'].apply(lambda x: preprocess_text(x))
dataset['name'] = dataset['name'].apply(lambda x: preprocess_text(x))
dataset['plot'] = dataset['plot'].apply(lambda x: preprocess_text(x))


condition = []
for (x,y) in dataset["genre"].value_counts().items():
    if (y>=400 and y<=8000) and x!="":
        condition.append(x)
      
dataset = dataset[dataset.genre.isin(condition)]

dataset.reset_index(drop=True,inplace=True)

genre_Encoder = LabelEncoder()
dataset["genre"] = genre_Encoder.fit_transform(dataset["genre"])
dataset = dataset[["genre","plot"]]
sentence_lengths =[len(list(set(dataset["plot"][i].split()))) for i in range(dataset.shape[0])]
maxlen= int(np.mean(sentence_lengths))
vectorizer = TextVectorization(max_tokens=sum(sentence_lengths),output_mode="int",output_sequence_length=maxlen)
vectorizer.adapt(dataset["plot"])

index = int(dataset.shape[0]*0.20)

train_set =dataset.iloc[index:,:]
test_set = dataset.iloc[-index:,:]

X_train_set = vectorizer(train_set["plot"])
X_test_set = vectorizer(test_set["plot"])
Y_train_set = train_set["genre"]
Y_test_set = test_set["genre"]

classes=np.unique(Y_train_set).shape[0]