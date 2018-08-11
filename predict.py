import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.externals import joblib

def get_ready():
    tfidf_vect = joblib.load('./assets/ti_vec.pkl')
    selector = joblib.load('./assets/selector.pkl')
    model = load_model('./assets/IMDB_mlp_model_0.90.h5')
    return tfidf_vect, selector, model

def predict(ttX, tfidf_vect, selector, model):
    x_val = tfidf_vect.transform([ttX])
    x_val = selector.transform(x_val).astype('float32')
    # Predict model and return prediction
    return model.predict(x_val)
