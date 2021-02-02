import numpy as np
from matplotlib import pyplot as plt

# Plot loss

history = np.load('lstm/model_history.npy', allow_pickle=True).item()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
