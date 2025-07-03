import pandas as pd

class DataLoader:
    def load(self, data_dir):
        ''' Get the training data and test data from two json files as two pandas'''
        train_data = pd.read_json(f"{data_dir}/train.json")
        # dont need ID column
        train_data.drop(['id'], axis=1, inplace=True)

        test_data = pd.read_json(f"{data_dir}/test.json")
        test_data.drop(['id'], axis=1, inplace=True)
        
        return train_data, test_data