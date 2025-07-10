#!/usr/bin/env python
# coding: utf-8

from data_loader import DataLoader
from pipeline import Pipeline

# create loaded to load data
loader = DataLoader()

# load training data
X_train, Y_train = loader.load_data("data", "train_emoji.csv")
#_X_test, _Y_test = loader.load_data("data", "test_emoji.csv")
# load GloVe: Global Vectors for Word Representation
word_to_index, index_to_word, word_to_vec_map = loader.load_glove("data", "glove.6B.50d.txt")

# determine the max length of the data
max_len = len(max(X_train, key=lambda x: len(x.split())).split())

# the emojis we are using
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

# create a pipeline to build a model, train it and use it.
pipeline = Pipeline(emoji_dictionary, word_to_vec_map, word_to_index, max_len)

pipeline.build_model((max_len,))
pipeline.train(X_train, Y_train)

#acc = pipeline.evaluate(_X_test, _Y_test)
#print()
#print("Test accuracy = ", acc)

# test a string"
test_str = "you are failing this exercise"
icon = pipeline.predict(test_str)
print(f"{test_str} results in emoji =--> {icon}")