import numpy as np
from data_loader import DataLoader
from nn import NeuralNet

# Loade pictures from cats and non-cats
train_x, train_set_y_orig, test_x, test_set_y_orig, classes = DataLoader().load_data("data")

# The input is a (64,64,3) image which is flattened to a vector of size (12288,1)
# model is 
layer_dims = [12288, 20, 7, 5, 1] #  4-layer model

# train the network on the training set.
nn = NeuralNet(layer_dims)
nn.train(train_x, train_set_y_orig, num_iterations = 2500, print_cost = True)

# test the network on the test set. 
m = test_x.shape[1]
pred = nn.predict(test_x)
print("Accuracy: "  + str(np.sum((pred == test_set_y_orig)/m)))

# will be around 80%, not too bad.
