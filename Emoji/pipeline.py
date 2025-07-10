import emoji
import numpy as np
import tensorflow as tf

""" pipeline to build a model that predicts the emoji to match a string.

"""
class Pipeline:
    def __init__(self, emoji_dictionary, word_to_vec_map, word_to_index, max_len):
        self.word_to_vec_map = word_to_vec_map
        self.word_to_index = word_to_index
        self.max_len = max_len
        self.emoji_dictionary = emoji_dictionary
        self.C = len(self.emoji_dictionary.keys())
        
        any_word = next(iter(self.word_to_vec_map.keys()))
        self.emb_dim = self.word_to_vec_map[any_word].shape[0]    # define dimensionality of the GloVe word vectors


    def convert_to_one_hot(self, Y):
        """ convert to one-hot encoding """
        Y = np.eye(self.C)[Y.reshape(-1)]
        return Y

    def label_to_emoji(self, label):
        """
        Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
        """
        return emoji.emojize(self.emoji_dictionary[str(label)], language='alias')

    def sentences_to_indices(self, X):
        """
        Converts an array of sentences (strings) into an array of indices corresponding to the word embeddings
        
        Arguments:
        X -- array of sentences (strings), of shape (m,)
        
        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        """
        m = X.shape[0]                                   # number of training examples
        X_indices = np.zeros((m, self.max_len))
        for i in range(m):
            # Convert the ith training sentence to lower case and split it into words. You should get a list of words.
            sentence = X[i]
            sentence_words = sentence.lower().split()
            j = 0
            for w in sentence_words:
                # if w exists in the word_to_index dictionary
                if w in self.word_to_index:
                    X_indices[i, j] = self.word_to_index[w]
                    j =  j + 1        
        return X_indices

    def pretrained_embedding_layer(self):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe dimensional vectors.
        Returns:
        embedding_layer -- pretrained layer Keras instance
        """
        
        vocab_size = len(self.word_to_index) + 1              # adding 1 to fit Keras embedding (requirement for Unknown word)
        emb_matrix = np.zeros((vocab_size, self.emb_dim))
        # the word vector representation of the idx'th word of the vocabulary
        for word, idx in self.word_to_index.items():
            emb_matrix[idx, :] = self.word_to_vec_map[word]
        # set non trainable to not overwrite the embeddings
        embedding_layer = tf.keras.layers.Embedding(vocab_size, self.emb_dim, trainable=False)
        # Build the embedding layer, it is required before setting the weights of the embedding layer. 
        embedding_layer.build((None,))        
        # Set the weights of the embedding layer to the embedding matrix. The layer is now pretrained.
        embedding_layer.set_weights([emb_matrix])
        return embedding_layer
    
    def predict(self, str):
        """ Predict the emoji for the provided string """
        x_test = np.array([str])
        X_test_indices = self.sentences_to_indices(x_test)
        predictions = self.model.predict(X_test_indices)
        predict = np.argmax(predictions)
        return self.label_to_emoji(predict)
    
    def evaluate(self, X_test, Y_test):
        """ evaluate a test set and return the accuracy """
        X_test_indices = self.sentences_to_indices(X_test)
        Y_test_oh = self.convert_to_one_hot(Y_test)
        loss, acc = self.model.evaluate(X_test_indices, Y_test_oh)
        return acc

    def train(self, X_train, Y_train, epochs = 50, batch_size = 32):
        """ Train the model. """
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        X_train_indices = self.sentences_to_indices(X_train)
        Y_train_oh = self.convert_to_one_hot(Y_train)
        self.model.fit(X_train_indices, Y_train_oh, epochs = epochs, batch_size = batch_size, shuffle=True)

    def build_model(self, input_shape):
        """
        Build the internal model
        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        Returns:
        model -- a model instance in Keras
        """
        sentence_indices = tf.keras.layers.Input(shape=input_shape, dtype='int32')
        # Create the embedding layer pretrained with GloVe Vectors
        embedding_layer = self.pretrained_embedding_layer()
        embeddings = embedding_layer(sentence_indices)   
        # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
        X = tf.keras.layers.LSTM(units = 128, return_sequences= True)(embeddings)
        # Add dropout with a probability of 0.5
        X = tf.keras.layers.Dropout(rate = 0.5, )(X) 
        # Propagate X trough another LSTM layer with 128-dimensional hidden state
        X = tf.keras.layers.LSTM(units = 128, return_sequences= False)(X)
        # Add dropout with a probability of 0.5
        X = tf.keras.layers.Dropout(rate = 0.5, )(X)  
        # Propagate X through a Dense layer with C units
        X = tf.keras.layers.Dense(units = self.C)(X)
        # Add a softmax activation
        Y = tf.keras.layers.Activation("softmax")(X)
        # Create Model instance which converts sentence_indices into X.
        self.model = tf.keras.models.Model(inputs=[sentence_indices], outputs=Y)