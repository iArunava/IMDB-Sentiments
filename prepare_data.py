from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#Vectorization Parameters
NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOC_FREQ = 2

def ngram_vectorize(train_texts, train_labels, val_texts):
    kwargs = {
        'ngram_range' : NGRAM_RANGE,
        'dtype' : 'int32',
        'strip_accents' : 'unicode',
        'decode_error' : 'replace',
        'analyzer' : TOKEN_MODE,
        'min_df' : MIN_DOC_FREQ
    }

    tfidf_vect = TfidfVectorizer(**kwargs)

    # Fit Vocab from train texts and trainsform train and val texts
    x_train = tfidf_vect.fit_transform(train_texts)
    x_val = tfidf_vect.transform(val_texts)

    # Select top 'k' of the vectorized features
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(train_texts, train_labels)
    x_train = selector.transform(train_texts).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val
