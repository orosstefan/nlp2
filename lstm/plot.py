import numpy as np
from matplotlib import pyplot as plt

# Plot loss
def plot_loss():
    history = np.load('lstm/model_history.npy', allow_pickle=True).item()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
def plot_rouge(values):
    plt.hist(values, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7 ,0.8, 0.9,1])
    plt.title('Rouge values')
    plt.ylabel('Loss')
    plt.xlabel('Rouge intervals')
    plt.show()

import pickle
with open('lstm/rouge_score.txt', 'rb') as f:
    values = pickle.load(f)
    plot_rouge(values)

