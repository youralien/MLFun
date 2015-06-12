from imdb import imdb
from foxhound import iterators
from foxhound.preprocessing import Tokenizer
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded, LenClip

from foxyfuel import FoxyDataStream

# # # # # # # # # # # 
# Modeling Training #
# # # # # # # # # # #

# Data Loading
trX, teX, trY, teY = imdb()

# Word Vectors
vect = Tokenizer(min_df=10, max_features=100000)
vect.fit(trX)

# Transforms
trXt=lambda x: intX(SeqPadded(vect.transform(LenClip(x, 1000))))
teXt=lambda x: intX(SeqPadded(vect.transform(x)))
Yt=lambda y: floatX(y).reshape(-1, 1)

# Foxhound Iterators
train_iterator = iterators.Linear(trXt=trXt, trYt=Yt)
test_iterator = iterators.Linear(trXt=teXt, trYt=Yt)

# DataStreams
train_stream = FoxyDataStream(trX, trY, train_iterator)
test_stream = FoxyDataStream(teX, teY, test_iterator)
