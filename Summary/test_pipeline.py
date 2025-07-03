import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from data_loader import DataLoader
from summ_pre_processor import SummaryPreProcessor
from pipeline import Pipeline

sos = '[SOS]'
eos = '[EOS]'
data_loader = DataLoader()
pre_processor = SummaryPreProcessor(sos, eos)

# do not filter for [ and ], because they ard part of the sos and eos.
filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'
oov_token = '[UNK]'
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token, lower=False)
pipeline = Pipeline(data_loader, pre_processor, tokenizer, encoder_maxlen = 150, decoder_maxlen = 50, sos = sos, eos = eos)
dataset, document, summary, document_test, summary_test = pipeline.load("data/corpus", buffer_size = 10000, batch_size = 64)

pipeline.build_model(num_layers = 2, embedding_dim = 128, fully_connected_dim = 128, num_heads = 2, positional_encoding_length = 256)


# UNCOMMENT THIS TO TRAIN AND SAVE THE WEIGHTS.
# Be aware, this will take a long time on a windows machine because tensorflow will not use the GPU on a windows machine without WSL
#pipeline.train(100)
#pipeline.save_weights_to_dir("weights")


# the model is build, but not yet connected all the layers. That will happed during a training or summarization task.
# so this is a simple summarization task to just connect the layers so we can load the weights.
txt = "[SOS] this is just to set the weights.  [EOS]"
sum = pipeline.summarize(txt)
pipeline.load_weights_from_dir("weights")

print(f"Show training results")
for i in range(5):
    # Check a summary of a document from the training set
    print(f'-- example {i}:')
    print(document[i])
    print('\nHuman written summary:')
    print(summary[i])
    print('\nModel written summary:')
    print(pipeline.summarize(document[i]))
    print("------------------------")

print("===========================================")
print(f"Show test results")
for i in range(5):
    # Check a summary of a document from the training set
    print(f'-- example: {i}')
    print(document_test[i])
    print('\nHuman written summary:')
    print(summary_test[i])
    print('\nModel written summary:')
    print(pipeline.summarize(document_test[i]))
    print("------------------------")

