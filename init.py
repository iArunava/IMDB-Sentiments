from load_dataset import *
from prepare_data import *

path = './'

trX, trY, ttX, ttY = load_imdb_dataset(path)
x_train, x_val = ngram_vectorize(trX, trY, ttX)
