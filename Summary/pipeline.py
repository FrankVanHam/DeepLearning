''' concept of a 'pipeline' borrow concept from huggingface '''

import tensorflow as tf
import pandas as pd
import time

from transformer import Transformer

class Pipeline:
    def __init__(self, data_loader, pre_processor, tokenizer, encoder_maxlen, decoder_maxlen, sos, eos):
        self.encoder_maxlen = encoder_maxlen
        self.decoder_maxlen = decoder_maxlen
        self.sos = sos
        self.eos = eos
        self.data_loader = data_loader
        self.pre_processor = pre_processor
        self.tokenizer = tokenizer

    def save_weights_to_dir(self, dir):
        self.transformer.save_weights(dir + "/transformer.weights.h5")

    def load_weights_from_dir(self, dir):
        self.transformer.load_weights(dir + "/transformer.weights.h5")

    def load(self, data_dir, buffer_size, batch_size):
        train_data, test_data = self.data_loader.load(data_dir)

        document, summary = self.pre_processor.process(train_data)
        document_test, summary_test = self.pre_processor.process(test_data)
        
        inputs, targets = self._tokenize(document, summary)
        self.dataset = self._create_dataset(inputs, targets, buffer_size, batch_size)
        return self.dataset, document, summary, document_test, summary_test
    
    def _tokenize(self, document, summary):
        documents_and_summary = pd.concat([document, summary], ignore_index=True)
        self.tokenizer.fit_on_texts(documents_and_summary)
        inputs = self.tokenizer.texts_to_sequences(document)
        targets = self.tokenizer.texts_to_sequences(summary)
        return inputs, targets
    
    def _vocab_size(self):
        return len(self.tokenizer.word_index) + 1
    
    def _create_dataset(self, inputs, targets, buffer_size, batch_size):
        # Pad the sequences.
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=self.encoder_maxlen, padding='post', truncating='post')
        targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=self.decoder_maxlen, padding='post', truncating='post')

        inputs = tf.cast(inputs, dtype=tf.int32)
        targets = tf.cast(targets, dtype=tf.int32)

        # Create the final training dataset.
        return tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(buffer_size).batch(batch_size)
    
    def build_model(self, num_layers, embedding_dim, fully_connected_dim, num_heads, positional_encoding_length):

        # Initialize the model
        self.transformer = Transformer(
            num_layers, 
            embedding_dim, 
            num_heads, 
            fully_connected_dim,
            self._vocab_size(),
            self._vocab_size(),
            positional_encoding_length, 
            positional_encoding_length,
        )
    
    @tf.function
    def _train_step(self, inp, tar):
        """
        One training step for the transformer
        Arguments:
            inp (tf.Tensor): Input data to summarize
            tar (tf.Tensor): Target (summary)
        Returns:
            None
        """
        # The target is divided into tar_inp and tar_real. 
        # tar_inp is passed as an input to the decoder. 
        # tar_real is that same input shifted by 1: At each location in tar_input, tar_real contains the next token that should be predicted.
        # The transformer is an auto-regressive model: it makes predictions one part at a time, and uses its output so far to decide what to do next.
        # During training this example uses teacher-forcing. Teacher forcing is passing the true output to the next 
        # time step regardless of what the model predicts at the current time step.
        tar_inp  = tar[:, :-1]
        tar_real = tar[:, 1:]

        # Create masks
        enc_padding_mask = self._create_padding_mask(inp)
        look_ahead_mask = self._create_look_ahead_mask(tf.shape(tar_inp)[1])
        dec_padding_mask = self._create_padding_mask(inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(
                inp,
                tar_inp, 
                training=True, 
                enc_padding_mask=enc_padding_mask, 
                look_ahead_mask=look_ahead_mask, 
                dec_padding_mask=dec_padding_mask
            )
            loss = self._masked_loss(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        self.train_loss(loss)

    def _masked_loss(self, real, pred):
        # create a mask to filter out the '0's that were used for padding.
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        # invoke the 'normal' loss function.
        loss_ = self.loss_object(real, pred)
        # Mask off the losses on padding.
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def train(self, epochs):
        # Training loop
        self.optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # note form_logits is False because the transformer is using a softmax layer at the end.
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        losses = []
        for epoch in range(epochs):    
            start = time.time()
            self.train_loss.reset_state()
            number_of_batches=len(list(enumerate(self.dataset)))

            for (batch, (inp, tar)) in enumerate(self.dataset):
                print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
                self._train_step(inp, tar)
            
            print (f'Epoch {epoch+1}, Loss {self.train_loss.result():.4f}')
            losses.append(self.train_loss.result())
        return losses

    def next_word(self, inp, tar):
        """
        Helper function for summarization that uses the model to predict just the next word.
        Arguments:
            encoder_input (tf.Tensor): Input data to summarize
            output (tf.Tensor): (incomplete) target (summary)
        Returns:
            predicted_id (tf.Tensor): The id of the predicted word
        """
        enc_padding_mask = self._create_padding_mask(inp)
        # Create a look-ahead mask for the output
        look_ahead_mask = self._create_look_ahead_mask(tf.shape(tar)[1])
        # Create a padding mask for the input (decoder)
        dec_padding_mask = self._create_padding_mask(inp)

        # Run the prediction of the next word with the transformer model
        predictions, attention_weights = self.transformer(
                inp,
                tar, 
                training=False, 
                enc_padding_mask=enc_padding_mask, 
                look_ahead_mask=look_ahead_mask, 
                dec_padding_mask=dec_padding_mask
            )
        # filter the predictions and take the word with the highest probability (argmax). Note that this is the 'greedy' implementation
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        return predicted_id

    def summarize(self, input_document):
        """
        A function for summarization using the transformer model
        Arguments:
            input_document (tf.Tensor): Input data to summarize
        Returns:
            _ (str): The summary of the input_document
        """    
        input_document = self.tokenizer.texts_to_sequences([input_document])
        input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=self.encoder_maxlen, padding='post', truncating='post')
        encoder_input = tf.expand_dims(input_document[0], 0)
        
        output = tf.expand_dims([self.tokenizer.word_index[self.sos]], 0)
        
        for i in range(self.decoder_maxlen):
            predicted_id = self.next_word(encoder_input, output)
            output = tf.concat([output, predicted_id], axis=-1)
            
            if predicted_id == self.tokenizer.word_index[self.eos]:
                break

        return self.tokenizer.sequences_to_texts(output.numpy())[0]  # since there is just one translated document
    
    def _create_padding_mask(self, decoder_token_ids):
        """
        Creates a matrix mask for the padding cells
        
        Arguments:
            decoder_token_ids (matrix like): matrix of size (n, m)
        
        Returns:
            mask (tf.Tensor): binary tensor of size (n, 1, m)
        """    
        seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    
        # add extra dimensions to add the padding to the attention logits. 
        # this will allow for broadcasting later when comparing sequences
        return seq[:, tf.newaxis, :] 
    
    def _create_look_ahead_mask(self, sequence_length):
        """
        Returns a lower triangular matrix filled with ones
        
        Arguments:
            sequence_length (int): matrix size
        
        Returns:
            mask (tf.Tensor): binary tensor of size (sequence_length, sequence_length)
        """
        mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
        return mask 