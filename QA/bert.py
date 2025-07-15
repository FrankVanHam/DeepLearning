from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf

class Bert:
    """ DistilBert, specially for Question and Answering """
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

    def answer_me(self, context, question):
        #Tokenize context and question in a way that is expected from the model
        inputs = self.tokenizer(question, context, return_tensors="tf")
        outputs = self.model(**inputs)

        # Get the start and end logits from the model's outputs
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the predicted answer span indices with the highest probabilities
        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

        # Get the predicted answer by slicing the input with the answer indexes.
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        predicted_answer = self.tokenizer.decode(predict_answer_tokens)

        return predicted_answer