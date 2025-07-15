import csv, os
import numpy as np

class DataLoader:
    def read_glove_vecs(self, glove_file):
        with open(glove_file, 'r', encoding="utf-8") as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
            i = 1
            words_to_index = {}
            index_to_words = {}
            for w in sorted(words):
                words_to_index[w] = i
                index_to_words[i] = w
                i = i + 1
        return words_to_index, index_to_words, word_to_vec_map

    def read_csv(self, filename):
        phrase = []
        emoji = []

        with open (filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)

            for row in csvReader:
                phrase.append(row[0])
                emoji.append(row[1])

        X = np.asarray(phrase)
        Y = np.asarray(emoji, dtype=int)

        return X, Y

    def load_data(self, data_dir, file):
        return self.read_csv(os.path.join(data_dir, file))

    def load_glove(self, data_dir, file):
        return self.read_glove_vecs(os.path.join(data_dir, file))
