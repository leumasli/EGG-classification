## Plots
import matplotlib.pyplot as plt

def plot_accuracy_loss(history, desc='Model'):
  # summarize history for accuracy
  plt.plot(history['accuracy'])
  plt.plot(history['val_accuracy'])
  plt.title(f'{desc} accuracy with data preprocessing')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

  # summarize history for loss
  plt.plot(history['loss'],'o')
  plt.plot(history['val_loss'],'o')
  plt.title(f'{desc} loss with data preprocessing')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()