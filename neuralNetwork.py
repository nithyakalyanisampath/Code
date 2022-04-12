import numpy
import scipy.special
#Neural Network Class definition
class neuralNetwork:
    #initilise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate) :
        #set the number in each input, hiddedn and output layers
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #learning rate
        self.lr = learningrate

        #link weight matrices wih and who
        self.wih = (numpy.random.rand(self.hnodes,self.inodes)-0.5)
        self.who = (numpy.random.rand(self.onodes,self.hnodes)-0.5)

        #activation fucntion is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #train the neural network
    def train(self,inputs_list,targets_list):
        #convert inputs list into 2d array 
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        #calculate the signals into the hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)

        #calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate the signals into final output layer
        final_inputs =  numpy.dot(self.who, hidden_outputs)
        #calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        #error is 
        output_errors = targets - final_outputs

        # hidden layer errors
        hidden_errors = numpy.dot(self.who.T,output_errors)

        pass

    #query the neural network
    def query(self,inputs_list):
        #convert inputs into 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)

        #calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)

        #calculate the signals emerging from the final output layer 
        final_outputs = self.activation_function(final_inputs)

        return final_outputs




