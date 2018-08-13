from load_dataset import *
from prepare_data import *
from build_model import *

path = './'

trX, trY, ttX, ttY = load_imdb_dataset(path)
results = train_ngram_model((trX, trY, ttX, ttY))
print ('With lr=1e-3 | val_acc={results[0]} | val_loss={results[1]}'.format(results=results))
print ('===========================================================================================')
print (results)
