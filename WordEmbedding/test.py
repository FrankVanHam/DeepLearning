#!/usr/bin/env python
# coding: utf-8

from data_loader import DataLoader
from processor import Processor

# create loaded to load data
loader = DataLoader()

# load GloVe: Global Vectors for Word Representation
word_to_index, index_to_word, word_to_vec_map = loader.load_glove("data", "glove.6B.50d.txt")

processor = Processor(word_to_vec_map)

triads_to_try = [('italy', 'italian',    'spain'), 
                 ('india', 'delhi',      'japan'), 
                 ('man',   'woman',      'boy'),
                 ('small', 'smaller',    'large'),
                 ('small', 'smaller',    'big'),
                 ('woman', 'prostitute', 'man')]

print("printing some similarities. Will take a minute...")
for triad in triads_to_try:
    analog = processor.complete_analogy(*triad)
    print(f"'{triad[0]}' relates to '{triad[1]}' as '{triad[2]}' relates to '{analog}'")
print(f"Note that not all analogies went well, but still amazing that this information is stored in embeddings")
