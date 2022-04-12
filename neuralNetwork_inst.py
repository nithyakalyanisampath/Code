
from neuralNetwork import neuralNetwork
input_nodes =3
output_nodes =3
hidden_nodes =3

learning_rate =.3

n= neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

n.query([1.0,0.5,-1.5])