from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Mapping

lotr = '/data1/user_data/lotr-nets/dataset/lotr.hdf5'


def transpose_stream(data):
    """ taken from johnarevalo/blocks-char-rnn """
    return (data[0].T, data[1].T)

def get_stream(hdf5_file, which_set, batch_size=None):
    """ taken from johnarevalo/blocks-char-rnn """
    dataset = H5PYDataset(
        hdf5_file, which_sets=(which_set,), load_in_memory=True)
    if batch_size == None:
        batch_size = dataset.num_examples
    stream = DataStream(dataset=dataset, iteration_scheme=ShuffledScheme(
        examples=dataset.num_examples, batch_size=batch_size))
    # Required because Recurrent bricks receive as input [sequence, batch,
    # features]
    return Mapping(stream, transpose_stream)

train = get_stream(lotr, 'train', batch_size=128)
dev = get_stream(lotr, 'dev', batch_size=128)


