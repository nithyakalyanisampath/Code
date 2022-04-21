
from neuralNetwork import neuralNetwork
input_nodes =3
output_nodes =3
hidden_nodes =3

learning_rate =.3

n= neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

n.query([1.0,0.5,-1.5])



import numpy
import matplotlib.pyplot as plt
data_file = open("mnist_dataset/mnist_train_100.csv",'r')
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array,cmap='Greys',interpolation = 'None')

plt.show()
