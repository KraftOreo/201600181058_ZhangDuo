from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses ReLU nonlinearities after the first and the second fully
    connected layers.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the third fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, H)
        b2: Second layer biases; has shape (H,)
        W3: Third layer weights; has shape (H, C)
        b3: Third layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # the first layer
        z1 = X.dot(W1) + np.tile(np.reshape(b1, (1, b1.shape[0])), (N, 1))
        a1 = np.where(z1 < 0, 0, z1)
        z2 = a1.dot(W2) + np.tile(np.reshape(b2, (1, b2.shape[0])), (a1.shape[0], 1))
        a2 = np.where(z2 < 0, 0, z2)
        scores = a2.dot(W3) + np.tile(np.reshape(b3, (1, b3.shape[0])), (a2.shape[0], 1))
        z3 = scores
        temp = np.sum(np.exp(z3), axis=1).reshape((z3.shape[0], 1))
        prob = np.exp(z3) / np.tile(temp, z3.shape[1])
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1, W2, and W3. Store the    #
        # result in the variable loss, which should be a scalar. Use the Softmax    #
        # classifier loss.                                                          #
        #############################################################################
        y_real = np.zeros_like(prob)
        for i in range(N):
            y_real[i, y[i]] = 1
        loss = -np.sum(np.sum(y_real * np.log(prob), axis=1),
                       axis=0) + 0.5 * reg * (np.sum(W3 * W3) + np.sum(W2 * W2) + np.sum(W1 * W1))
        loss /= N
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        delta3 = prob - y_real
        dw3 = a2.T.dot(prob - y_real) + reg * W3
        # dw3 /= N
        db3 = delta3.sum(axis=0)
        # db3 /= N
        delta2 = delta3.dot(W3.T) * np.where(z2 < 0, 0, 1)
        dw2 = a1.T.dot(delta2) + reg * W2
        # dw2 /= N
        db2 = delta2.sum(axis=0)
        # db2 /= N
        delta1 = delta2.dot(W2.T) * np.where(z1 < 0, 0, 1)
        dw1 = X.T.dot(delta1) + reg * W1
        # dw1 /= N
        db1 = delta1.sum(axis=0)
        # db1 /= N
        grads['dW3'] = dw3
        grads['dW2'] = dw2
        grads['dW1'] = dw1
        grads['db3'] = db3
        grads['db2'] = db2
        grads['db1'] = db1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        import random
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            loc = random.sample(range(X.shape[0]), batch_size)
            X_batch = X[loc, :]
            y_batch = y[loc]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            '''
            adam
            '''
            L = 3  # number of layers in the neural networks
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            v = {}
            s = {}
            v_corrected = {}  # Initializing first moment estimate, python dictionary
            s_corrected = {}  # Initializing second moment estimate, python dictionary
            # Perform Adam update on all parameters
            for l in range(L):
                # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
                v["dW" + str(l + 1)] = np.zeros_like(self.params["W" + str(l + 1)])
                v["db" + str(l + 1)] = np.zeros_like(self.params["b" + str(l + 1)])
                s["dW" + str(l + 1)] = np.zeros_like(self.params["W" + str(l + 1)])
                s["db" + str(l + 1)] = np.zeros_like(self.params["b" + str(l + 1)])
                v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
                v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
                v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1)
                v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1)
                s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads["dW" + str(l + 1)],
                                                                                             2)
                s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads["db" + str(l + 1)],
                                                                                             2)
                s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2)
                s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2)
                self.params["W" + str(l + 1)] -= learning_rate * (
                        v_corrected["dW" + str(l + 1)] / (
                        np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon * np.ones_like(
                    s_corrected["dW" + str(l + 1)])))
                self.params["b" + str(l + 1)] -= learning_rate * (
                        v_corrected["db" + str(l + 1)] / (
                        np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon * np.ones_like(
                    s_corrected["db" + str(l + 1)])))
            # self.params['W3'] -= learning_rate * grads['dW3']
            # self.params['W2'] -= learning_rate * grads['dW2']
            # self.params['W1'] -= learning_rate * grads['dW1']
            # self.params['b1'] -= learning_rate * grads['db1']
            # self.params['b2'] -= learning_rate * grads['db2']
            # self.params['b3'] -= learning_rate * grads['db3']
            # learning_rate *= learning_rate_decay
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this three-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3 = self.params['W3']
        b1 = self.params['b1']
        b2 = self.params['b2']
        b3 = self.params['b3']
        z1 = X.dot(W1) + np.tile(np.reshape(b1, (1, b1.shape[0])), (X.shape[0], 1))
        a1 = np.where(z1 < 0, 0, z1)
        z2 = a1.dot(W2) + np.tile(np.reshape(b2, (1, b2.shape[0])), (a1.shape[0], 1))
        a2 = np.where(z2 < 0, 0, z2)
        z3 = a2.dot(W3) + np.tile(np.reshape(b3, (1, b3.shape[0])), (a2.shape[0], 1))
        temp = np.sum(np.exp(z3), axis=1).reshape((z3.shape[0], 1))
        scores = np.exp(z3) / np.tile(temp, z3.shape[1])
        # scores = np.exp(h3) / np.tile(np.sum(np.exp(h3), axis=1).reshape((1, h3.shape[1])), (X.shape[0], 1))
        y_pred = np.zeros(scores.shape[0])
        for i in range(scores.shape[0]):
            temp1 = np.where(scores[i] == np.max(scores[i]))[0]
            y_pred[i] = np.where(scores[i] == np.max(scores[i]))[0]
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
