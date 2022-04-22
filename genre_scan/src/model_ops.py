import string
import re
import numpy as np
from .dataset import *
import tensorflow as tf


def preprocess_text(text):
    text = str(text)
    characters = string.punctuation
    text = text.lower()
    text = text.strip(characters)
    list_text = text.split()
    try:
        ignore = re.findall("[a-z]+["+characters+"]+[0-9|a-z|()*+,-./:;<=>?@[\\]^_`]+",text)[0]
        text =[i for i in list_text if i != ignore]
        text= " ".join(text)
        text = re.sub('[%s]' %re.escape(characters),"",text)
    except:
        text = " ".join(list_text)
        text = re.sub('[%s]' %re.escape(characters),"",text)
    return text


def get_mapping(vectorizer):
    mapping = {}
    for i,x in enumerate(vectorizer.get_vocabulary()):
        mapping[i]=x
    return mapping

def process_data(inp,out):
    return tf.cast(inp,tf.int32),tf.cast(out,tf.int32)


class Postional_Encoding(tf.keras.layers.Layer):
    def __init__(self,embedding_depth,sequence_length):
        super(Postional_Encoding,self).__init__()
        self.embedding_depth = embedding_depth
        self.sequence_length = sequence_length
        self.supports_masking = True
        
    def call(self,data):
        batch_dim = tf.shape(data)[0]
        embeds = np.arange(self.embedding_depth)[np.newaxis,:]
        embeds = 1 / np.power(10000, (2 * (embeds//2)) / np.float32(self.embedding_depth))
        location_id = np.arange(self.sequence_length)[:,np.newaxis]
        pos = embeds*location_id
        pos[:,::2] = np.sin(pos[:,::2])
        pos[:,1::2] = np.cos(pos[:,1::2])
        pos = tf.tile(pos[tf.newaxis,:,:],(batch_dim,1,1))
        return tf.cast(pos,tf.float32)

    def compute_mask(self,data,mask=None):
        return tf.not_equal(0,data)
    
class Transformer_Encoder(tf.keras.layers.Layer):
    def __init__(self,embedding_depth,dense_dim,seq_length,vocab,heads=2,**kwargs):
        super(Transformer_Encoder,self).__init__(**kwargs)
        self.embedding_depth = embedding_depth
        self.seq_length = seq_length
        self.vocab = vocab
        self.heads = heads
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab,self.embedding_depth)
        self.pos_encod_layer = Postional_Encoding(self.embedding_depth,self.seq_length)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=self.heads,key_dim=self.embedding_depth)
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(dense_dim, activation="relu"), tf.keras.layers.Dense(self.embedding_depth),])
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True
        
        
    def call(self,x):
        embed = self.embedding_layer(x)
        pos = self.pos_encod_layer(x)
        padding_mask = tf.cast(self.pos_encod_layer.compute_mask(x),tf.int32)[:,:,tf.newaxis]
        attention_input = embed+pos
        attention_out = self.attention(attention_input,attention_input,attention_input,attention_mask=padding_mask)
        layer_norm1 = self.layernorm_1(attention_input+attention_out)
        dense_proj = self.dense_proj(layer_norm1)
        layer_norm2 = self.layernorm_2(dense_proj+layer_norm1)
        return layer_norm2
    
class Classifier_Model(tf.keras.models.Model):
    def __init__(self,embedding_size,dense_dim,seq_length,vocab,heads=2,labels=None):
        super(Classifier_Model,self).__init__()
        self.attention = Transformer_Encoder(embedding_size,dense_dim,seq_length,vocab,heads=heads)
        self.classifier_head = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling1D(),
                                           tf.keras.layers.Dense(labels,activation="softmax")])
        
        
    def call(self,x):
        x = self.attention(x)
        x = self.classifier_head(x)
        return x