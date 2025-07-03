import re

''' Preprocess the training and test data before tokenizing '''
class SummaryPreProcessor:
    def __init__(self, sos, eos):
        self.sos = sos
        self.eos = eos
        
    def process(self, input_data):
        summary  = input_data.apply(lambda row : self.preprocess_util(row['summary']),  axis = 1)
        document = input_data.apply(lambda row : self.preprocess_util(row['dialogue']), axis = 1)
        return document, summary
    
    def preprocess_util(self, input_data):
        # Convert all text to lowercase
        lowercase = input_data.lower()
        # Remove newlines, tabs
        removed_newlines = re.sub("\n|\r|\t", " ",  lowercase)
        # remove double spaces
        removed_double_spaces = ' '.join(removed_newlines.split(' '))
        # Add start of sentence and end of sentence tokens
        s = self.sos + ' ' + removed_double_spaces + ' '+ self.eos
        return s