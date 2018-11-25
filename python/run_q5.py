import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
print(valid_x.shape)

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()
M_params = Counter()


# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'hidden')
initialize_weights(hidden_size,hidden_size,params,'hidden2')
initialize_weights(hidden_size,1024,params,'output')

initialize_Momentum_weights(1024,hidden_size,M_params,'layer1')
initialize_Momentum_weights(hidden_size,hidden_size,M_params,'hidden')
initialize_Momentum_weights(hidden_size,hidden_size,M_params,'hidden2')
initialize_Momentum_weights(hidden_size,1024,M_params,'output')

loss_plot = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        output = forward(h3, params, 'output', sigmoid)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss = np.sum((output-xb)**2)
        total_loss += loss

        # backward
        delta = 2*(output-xb)
        delta1 = backwards(delta, params, name='output', activation_deriv=sigmoid_deriv)
        delta2 = backwards(delta1, params, name='hidden2', activation_deriv=relu_deriv)
        delta3 = backwards(delta2, params, name='hidden', activation_deriv=relu_deriv)
        backwards(delta3, params, name='layer1', activation_deriv=relu_deriv)

        # apply gradient
        for k, v in params.items():
            if 'grad' in k:
                name = k.split('_')[1]
                M_params[name] = 0.9*M_params[name] - learning_rate * v
                params[name] += M_params[name]
    loss_plot.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

'''
    Print Loss
'''
time_seq = np.arange(max_iters)
import matplotlib.pyplot as plt
plt.figure(2)
ax = plt.gca()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.plot(time_seq, loss_plot, color='r', linewidth=1, alpha=0.6)
plt.pause(1500)
plt.close()

'''
    Save Parameters 
'''
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
import pickle

with open('q5_weights.pickle', 'rb') as handle:
    params = pickle.load(handle)

h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(904,910):
    plt.subplot(2,1,1)
    plt.imshow(valid_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


# evaluate PSNR
# Q5.3.2
from skimage.measure import compare_psnr as psnr
psnr_sum = 0
for i in range(valid_x.shape[0]):
    psnri = psnr(valid_x[i], out[i])
    psnr_sum += psnri
psnr_avg = psnr_sum / valid_x.shape[0]
print(psnr_avg)
