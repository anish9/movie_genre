import numpy as np
import pandas as pd
from .model_ops import *
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv("data/drugsComTest_raw.csv")
dataset.dropna(axis=0,inplace=True)
dataset['review'] = dataset['review'].apply(lambda x: preprocess_text(x))
dataset['drugName'] = dataset['drugName'].apply(lambda x: preprocess_text(x))
dataset['condition'] = dataset['condition'].apply(lambda x: preprocess_text(x))

condition = []
for (x,y),(w,z) in zip(dataset["drugName"].value_counts().items(), dataset['condition'].value_counts().items()):
    if (y>=90 and y<=900) and (z>=200 and z<=2000):
        condition.append(w)
        
dataset = dataset[dataset.condition.isin(condition)]
dataset = dataset[dataset["rating"]>=6]
dataset.reset_index(drop=True,inplace=True)

drugName_Encoder = LabelEncoder()
condition_Encoder = LabelEncoder()
dataset["drugName"] = drugName_Encoder.fit_transform(dataset["drugName"])
dataset["condition"] = condition_Encoder.fit_transform(dataset["condition"])
dataset = dataset[["drugName","condition","review"]]
sentence_lengths =[len(list(set(dataset["review"][i].split()))) for i in range(dataset.shape[0])]

vectorizer = TextVectorization(max_tokens=sum(sentence_lengths),output_mode="int",output_sequence_length=int(np.mean(sentence_lengths)))
vectorizer.adapt(dataset["review"])


index = int(dataset.shape[0]*0.15)

train_set =dataset.iloc[index:,:]
test_set = dataset.iloc[-index:,:]

X_train_set = vectorizer(train_set["review"])
X_test_set = vectorizer(test_set["review"])
Y_train_set = train_set["condition"]
Y_test_set = test_set["condition"]

classes  =np.unique(Y_train_set).shape[0]