from .src.train import model_
from .src.dataset import *
from .src.model_ops import *
import json
from collections import Counter
from collections import defaultdict
from scipy.special import softmax
import os

model_.load_weights(os.getcwd()+"/genre_scan/src/genre_Weights/classifier")

# possible_labels  =[action,adult,adventure,animation,comedy,crime,family,
#            horror,music,romance,short,sport,thriller,western]


def predict(text:str):
    text = preprocess_text(text)
    text = vectorizer(text)
    text = tf.cast(tf.expand_dims(text,axis=0),tf.int32)
    prob = model_.predict(text)
    output_dict = {}
    index = tf.argsort(prob,axis=-1)[0][-3:].numpy().tolist()
    scores = prob[0][index]
    labels = genre_Encoder.inverse_transform(index)
    for x,y in zip(labels,scores):
        output_dict[y] = x
    output_dict = {v: round(k*100,3) for k, v in sorted(output_dict.items(), key=lambda x: x[0],reverse=True)}
    return output_dict

def compose_matrix(plot):
    final_compose = {}
    
    max_length = 65
    score_thresh= 40.0
    list_sent = plot.split()
    interval = len(list_sent)//max_length
    plot_scorer = defaultdict(list)
    if len(list_sent)>80:
        if interval !=0:
            seq=[]
            for i in range(interval):
                start,end = max_length*(i),max_length*(i+1)
                seq.append(" ".join(list_sent[start:end]))
            for i in seq:
                out = predict(i)
                for x,y in out.items():
                    if float(y) >= score_thresh:
                        plot_scorer[x].append(y)

        for key,scores in plot_scorer.items():
            if len(scores)>=1:
                score = np.mean(scores)
                final_compose[key] = score
        return final_compose
    else:
        final_compose = predict(plot)
        return final_compose

def apply_prob(raw_scores):
    dicts = {}
    keys = list(raw_scores.keys())
    raw_scores = list(raw_scores.values())
    for i,x in enumerate(softmax(raw_scores)):
        score = round(x*100,2)
        if score > 0.0:
            dicts[keys[i]] =str(score)+" %"
    return dicts

def predict_main(plot):
    raw_compose = compose_matrix(plot)
    filter_out = apply_prob(raw_compose)
    return filter_out




