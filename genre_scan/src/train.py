import tensorflow as tf
import numpy as np
from .dataset import *
from tensorflow.keras.callbacks import ModelCheckpoint


train_config = dict()

batch_size=8

auto = tf.data.AUTOTUNE
train_set = tf.data.Dataset.from_tensor_slices((X_train_set,Y_train_set))
train_set = train_set.map(process_data,num_parallel_calls=auto)
train_set = train_set.batch(batch_size).shuffle(2000).prefetch(auto).cache().repeat()

test_set = tf.data.Dataset.from_tensor_slices((X_test_set,Y_test_set))
test_set = test_set.map(process_data,num_parallel_calls=auto)
test_set = test_set.batch(batch_size).shuffle(2000)

train_steps = X_train_set.shape[0]//batch_size
val_steps = X_test_set.shape[0]//batch_size


vocab = vectorizer.vocabulary_size()
seq_length = X_train_set.shape[1]
embedding_size = 256
dense_dim=512
transformer_heads = 4

# train_config["embedding_size"] = embedding_size
# train_config["vocab"]=vocab
# train_config["dense_dim"] = dense_dim
# train_config["seq_length"] = seq_length
# train_config["transformer_heads"]=transformer_heads

model_ = Classifier_Model(embedding_size,dense_dim,seq_length,vocab,heads=transformer_heads,labels=classes)

if  __name__ == "__main__":  
    ckpt = ModelCheckpoint("genre_Weights/classifier",save_best_only=True,save_weights_only=True,monitor='val_acc',mode='max') 
    model_.compile(optimizer="Nadam",loss="sparse_categorical_crossentropy",metrics=["acc"])
    model_.fit(train_set,batch_size=batch_size,validation_data=test_set,
               steps_per_epoch=train_steps,validation_steps=val_steps,epochs=25,callbacks=[ckpt])