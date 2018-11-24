import numpy as np
import nn

acc_plot = np.arange(50)
valid_acc_plot = np.arange(50)*2
time_seq = np.arange(50)
import matplotlib.pyplot as plt

plt.figure(1)
ax = plt.gca()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax.plot(time_seq, acc_plot, color='r', linewidth=1, alpha=0.6)
ax.plot(time_seq, valid_acc_plot, color='b', linewidth=1, alpha=0.6)
plt.pause(1500)
plt.close()