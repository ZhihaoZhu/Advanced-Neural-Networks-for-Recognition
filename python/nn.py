import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    '''

    这里要注意 W 是不是要resize
    :param in_size:
    :param out_size:
    :param params:
    :param name:
    :return:
    '''
    b = np.zeros((1, out_size))
    W = np.random.uniform(-np.sqrt(6)/np.sqrt(in_size+out_size),
                          np.sqrt(6)/np.sqrt(in_size+out_size), out_size*in_size).reshape((in_size,out_size))
    params['W' + name] = W
    params['b' + name] = b


def initialize_Momentum_weights(in_size,out_size,M_params,name=''):
    '''

    这里要注意 W 是不是要resize
    :param in_size:
    :param out_size:
    :param params:
    :param name:
    :return:
    '''
    b = np.zeros((1, out_size))
    W = np.zeros((in_size,out_size))
    M_params['W' + name] = W
    M_params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    sig = lambda p: 1/(1+np.exp(-p))
    res = sig(x)
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = X@W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    x = x.copy()
    max = -np.max(x,axis=1)
    expo = lambda p: np.exp(p)
    expo_x = expo(x)
    expo_max = expo(max).reshape((-1,1))

    expo_final = expo_x*expo_max

    sum = np.sum(expo_final, axis=1).reshape((-1,1))
    res = expo_final/sum
    
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    probs = probs.copy()
    log = lambda p: np.log(p)
    log_probs = log(probs)
    cross = y*log_probs
    loss = -np.sum(cross)
    n = y.shape[0]
    index = 0
    y_label = np.argmax(y, axis=1)
    probs_label = np.argmax(probs, axis=1)

    for i in range(n):
        if y_label[i] == probs_label[i]:
            index += 1
    acc = index/n

    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    delta_pre = delta * activation_deriv(post_act)
    grad_W = X.transpose() @ delta_pre
    grad_b = np.sum(delta_pre, axis=0, keepdims=True)
    # X is also D x in
    grad_X = delta_pre @ W.transpose()

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X


# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    N = x.shape[0]
    rand_index = np.random.permutation(N)
    num_batches = N//batch_size
    for i in range(num_batches):
        index = rand_index[i*batch_size:(i+1)*batch_size]
        x_batch = x[index,:]
        y_batch = y[index,:]
        batches.append((x_batch,y_batch))

    return batches
