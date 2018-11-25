import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

#
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']


# print(train_y.shape)
#
#
# max_iters = 100
# # pick a batch size, learning rate
# batch_size = 16
# learning_rate = 1e-3
# hidden_size = 64
#
# batches = get_random_batches(train_x,train_y,batch_size)
# batch_num = len(batches)
#
# params = {}
#
# # initialize layers here
# initialize_weights(1024,hidden_size,params,'layer1')
# initialize_weights(hidden_size,train_y.shape[1],params,'output')
#
#
# loss_plot = []
# acc_plot = []
# valid_acc_plot = []
# # with default settings, you should get loss < 150 and accuracy > 80%
# for itr in range(max_iters):
#     total_loss = 0
#     total_acc = 0
#     for xb,yb in batches:
#         # forward
#         h1 = forward(xb, params, 'layer1')
#         probs = forward(h1, params, 'output', softmax)
#
#         # loss
#         # be sure to add loss and accuracy to epoch totals
#         loss, acc = compute_loss_and_acc(yb, probs)
#         total_loss += loss
#         total_acc += acc
#
#         # backward
#         delta1 = probs
#         delta1 = delta1 - yb
#         delta2 = backwards(delta1, params, 'output', linear_deriv)
#         backwards(delta2, params, 'layer1', sigmoid_deriv)
#
#         # apply gradient
#         for k, v in params.items():
#             if 'grad' in k:
#                 name = k.split('_')[1]
#                 params[name] -= learning_rate * v
#
#     loss_plot.append(total_loss)
#     total_acc /= len(batches)
#     acc_plot.append(total_acc)
#
#     if itr % 2 == 0:
#         print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
#     # run on validation set and report accuracy! should be above 75%
#     h1 = forward(valid_x, params, 'layer1')
#     probs = forward(h1, params, 'output', softmax)
#     loss, acc = compute_loss_and_acc(valid_y, probs)
#     valid_acc = acc
#     valid_acc_plot.append(valid_acc)
#     print('Validation accuracy: ',valid_acc)
#
# '''
# Plot the loss and accuracy
# '''
# time_seq = np.arange(max_iters)
# import matplotlib.pyplot as plt
#
# plt.figure(1)
# ax = plt.gca()
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss')
# ax.plot(time_seq, acc_plot, color='r', linewidth=1, alpha=0.6, label = "acc_train")
# ax.plot(time_seq, valid_acc_plot, color='b', linewidth=1, alpha=0.6, label = "acc_val")
#
#
# plt.figure(2)
# ax = plt.gca()
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss')
# ax.plot(time_seq, loss_plot, color='r', linewidth=1, alpha=0.6)
# plt.pause(1500)
# plt.close()
#
#
#
#
#
# if False: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()
# import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Q3.1.3
# import pickle
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
#
# with open('q3_weights.pickle', 'rb') as handle:
#     saved_params = pickle.load(handle)
#
# weight = saved_params['Wlayer1']
# fig = plt.figure(1)
# grid = ImageGrid(fig, 111, (8, 8))
# for i in range(64):
#     weight_i = weight[:, i]
#     weight_i = weight_i.reshape(32, 32)
#     grid[i].imshow(weight_i)
#
# plt.show()
#
# initialize_weights(1024, 64, saved_params, 'original')
#
# weight = saved_params['Woriginal']
#
# fig = plt.figure()
# grid = ImageGrid(fig, 111, (8, 8))
# for i in range(64):
#     weight_i = weight[:, i]
#     weight_i = weight_i.reshape(32, 32)
#     grid[i].imshow(weight_i)
#
# plt.show()





# Q3.1.3
import pickle
with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))



h1 = forward(test_x, saved_params, 'layer1')
probs = forward(h1, saved_params, 'output', softmax)

y_gt = np.argmax(test_y, axis=1)
probs_gt = np.argmax(probs, axis=1)
print(y_gt)
print(probs_gt)


for i in range(test_x.shape[0]):
    confusion_matrix[y_gt[i]][probs_gt[i]] += 1


import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()