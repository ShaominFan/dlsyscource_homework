"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl
from needle import Tensor


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # ====== read image =====
    # image file: magic_n(32 bit), item_n(32 bit), rows_n(32 bit), cols_n(32 bit), pixel(unsigned byte)
    with gzip.open(image_filesname, 'rb') as f:
        magic_n, item_n, rows_n, cols_n = struct.unpack('>IIII', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(item_n, rows_n * cols_n).astype(np.float32)
        imgs /= 255.0

    # ====== read label =====
    # label file: magic_n(32 bit), item_n(32 bit), label(unsigned byte)
    with gzip.open(label_filename, 'rb') as f:
        magic_n, item_n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return imgs, labels
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    sum_z = ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1)))
    true_z = ndl.ops.summation(Z * y_one_hot, axes=(1))
    return ndl.ops.summation(sum_z - true_z) / y_one_hot.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    ### BEGIN YOUR SOLUTION
    y = y.reshape(y.shape[0], 1)
    m = X.shape[0]

    for idx in np.arange(0, m, batch):
        end_idx = min(idx + batch, m)
        batch_x = Tensor(X[idx: end_idx, :]) # batch_m * input_dim
        batch_y = Tensor(y[idx: end_idx, :]) # batch_m
        batch_m = batch_x.shape[0]

        z1 = ndl.ops.relu(ndl.ops.matmul(batch_x, W1)) # batch_m * hidden_dim
        z2 = ndl.ops.exp(ndl.ops.matmul(z1, W2)) / \
                ndl.ops.summation(ndl.ops.exp(ndl.ops.matmul(z1, W2)), axes=(1), keepdims=True) # batch_m * num_classes
        e_y = Tensor(np.array([range(0, z2.shape[1])] * z2.shape[0]) == batch_y.numpy(), dtype='uint8') # batch_m * num_classes
        
        g2 = z2 - e_y # batch_m * num_classes
        g1 = Tensor(z1.numpy() > 0, dtype='uint8') * ndl.ops.matmul(g2, W2.transpose())

        grad_w1 = ndl.ops.matmul(batch_x.transpose(), g1) / batch_m # input_dim * hidden_dim
        grad_w2 = ndl.ops.matmul(z1.transpose(), g2) / batch_m # hidden_dim * num_classes

        W1 = W1 - grad_w1 * lr
        W2 = W2 - grad_w2 * lr
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
