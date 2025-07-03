from tkinter import Image
import numpy as np


class NeuralNet:
    '''
    home build neural net that contains a multiple of ReLU layers and ends with 1 Sigmoid layer.
    Can be used to detect cats....
    '''
    def __init__(self, layer_dims):
        self.parameters = []
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims) # number of layers, including input and output!
        self.m = None # the batch set size. Set during training and prediction

    def train(self, X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (n_x, m)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, m)
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        self.m = X.shape[-1]
        np.random.seed(1)
        costs = []                         # keep track of cost
        # Parameters initialization.
        parameters = self.initialize_parameters_deep()
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward(X, parameters)
            # Compute cost.
            cost = self.compute_cost(AL, Y)
            # Backward propagation.
            grads = self.L_model_backward(AL, Y, caches)
            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate)
            # Print the cost every 100 iterations and for the last iteration
            if print_cost and (i % 100 == 0 or i == num_iterations - 1):
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0:
                costs.append(cost)
        
        self.parameters = parameters
        return parameters, costs
    
    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """
        
        L = self.L # number of layers in the neural network

        # Update rule for each parameter. Use a for loop starting at 1 to L (not including the input layer and output layer)
        for l in range(1, L):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
            
        return parameters
    
    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads = {}
        L = self.L # the number of layers
        m = self.m
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation given that it is a sigmoid activation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # last layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-2] # the cache of the but last layer
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        grads["dA" + str(L-2)] = dA_prev_temp
        grads["dW" + str(L-1)] = dW_temp
        grads["db" + str(L-1)] = db_temp
        
        for l in reversed(range(1, L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l-1]
            dAl = grads["dA" + str(l)]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAl, current_cache, activation = "relu")
            grads["dA" + str(l-1)] = dA_prev_temp
            grads["dW" + str(l)] = dW_temp
            grads["db" + str(l)] = db_temp

        return grads
    
    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, m)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, m)

        Returns:
        cost -- cross-entropy cost
        """
        
        m = self.m

        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost
    
    def initialize_parameters_deep(self):
        """
        Arguments:
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        np.random.seed(1)
        parameters = {}
        layer_dims = self.layer_dims
        L = self.L            # number of layers in the network

        for l in range(1, L): # excluding input and output layer
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01  # dont do this because learning will slow down a lot
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters
    
    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, m)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = self.L                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L-1):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L-1)], parameters['b' + str(L-1)], activation = "sigmoid")
        caches.append(cache)
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, m)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            
        else:
            raise Exception(f"Unknown activation {activation}")
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache
    
    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        else:
            raise Exception(f"Unknown activation {activation}")
        
        return dA_prev, dW, db
    
    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, m)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        Z = W.dot(A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache
    
    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = self.m

        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
    
    def sigmoid(self, Z):
        """
        Implements the sigmoid activation in numpy
        
        Arguments:
        Z -- numpy array of any shape
        
        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache

    def relu(self, Z):
        """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache


    def relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def predict(self, X):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        self.m = X.shape[1]
        m = self.m
        p = np.zeros((1,m))
        # Forward propagation
        probas, caches = self.L_model_forward(X, self.parameters)
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p

    def predict_file(self, my_image, num_px):
        fname = "images/" + my_image
        image = np.array(Image.open(fname).resize((num_px, num_px)))
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        return self.predict(image)
