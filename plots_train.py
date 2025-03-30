import matplotlib.pyplot as plt
import numpy as np

epochs = range(1, 501)

data_dagger0 = np.loadtxt('training_error_0dagger.csv', delimiter=",")
plt.plot(epochs, data_dagger0, marker='o', linestyle='-')
plt.title('Training Error During DAgger Iteration 0')
plt.ylabel('Training Error')
plt.xlabel('Epochs')
plt.show()

data_dagger1 = np.loadtxt('training_error_1dagger.csv', delimiter=",")
plt.plot(epochs, data_dagger1, marker='o', linestyle='-')
plt.title('Training Error During DAgger Iteration 1')
plt.ylabel('Training Error')
plt.xlabel('Epochs')
plt.show()

data_dagger2 = np.loadtxt('training_error_2dagger.csv', delimiter=",")
plt.plot(epochs, data_dagger2, marker='o', linestyle='-')
plt.title('Training Error During DAgger Iteration 2')
plt.ylabel('Training Error')
plt.xlabel('Epochs')
plt.show()