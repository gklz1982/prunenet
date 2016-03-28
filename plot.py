import numpy as np
import matplotlib
matplotlib.use('GTK')
import matplotlib.pyplot as plt
import dill as pickle

with open('conv1.model', 'r') as f:
    obj = pickle.load(f)
    plt.figure(1)
    plt.plot(obj['sparsity'])
    plt.title('sparsity')
    plt.show()
