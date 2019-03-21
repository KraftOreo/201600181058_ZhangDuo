import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3710, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (500, 3710)
    - y: A numpy array of shape (N, #10) containing training labels; y[i] = c means (500, 10)
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    prob (500, 10)
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    # Calculate the loss
    # Calculate the partial derivative of Cost function which is introduced as cross-entropy
    prob = np.exp(X.dot(W))
    y_real = np.zeros_like(prob)
    for i in range(prob.shape[0]):
        prob[i] /= np.sum(prob, axis=1)[i]
        y_real[i, y[i]] = 1
        loss -= np.sum(prob[i] * y_real[i])
    loss = loss + 0.5 * reg * np.sum(W * W)
    loss /= y.shape[0]
    dW = np.transpose(X).dot(prob - y_real) + reg * W
    dW /= y.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO:
    # Compute the softmax loss and its gradient using no explicit loops.        #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    prob = np.exp(X.dot(W))
    temp = np.sum(prob, axis=1).reshape((prob.shape[0], 1))
    prob = prob / np.tile(temp, prob.shape[1])
    y_real = np.zeros_like(prob)
    y_real[range(prob.shape[0], y)] = 1
    loss = -np.sum(np.sum(y_real * np.log(prob), axis=1), axis=0) + 0.5 * reg * np.sum(W * W)
    loss /= y.shape[0]
    dW = np.transpose(X).dot(prob - y_real) + reg * W
    dW /= y.shape[0]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW
