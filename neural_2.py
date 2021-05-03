import numpy as np
import math
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

#eps = 7./3 - 4./3 -1
#assert sigmoid(0) <= 0.5+eps and sigmoid(0) >= 0.5-eps

def sigmoid_der(a):
    return a*(1-a)

sigmoid_der_v = np.vectorize(sigmoid_der)



W_RANGE = [-0.5, 0.5] #inicializa os pesos entre esses valores
class Layer:
    def __init__(self,a = None, z = None, theta = None, bias = None, delta = None):
        self.a = a
        self.z = z #maybe its not needed <<<, delete later
        self.w = theta #weights (neurons) from w_ij
        self.b = bias #bias weights
        self.delta = delta #error of neuron i at layer, nx1 matrix (n nodes)
        self.w_grad = None #partial derivatives of cost (error) wrt weights
        self.b_grad = None #partial derivatives of cost fucntion wrt biases


class MultiLayer:
    def __init__(self,num_layers):
        self.layer = []
        self.num_layers = num_layers #neuron numbers on each layer
        for i in range(len(num_layers)):
            self.layer.append(Layer())
    def initialize_network(self,weights = None, bias = None): #weights (o theta)
        if weights is None: #initialize random weights
            for i in range(len(self.num_layers)-1):
                self.layer[i].w = np.random.default_rng().uniform(W_RANGE[0],W_RANGE[1],
                size=(self.num_layers[i+1], self.num_layers[i]))
                self.layer[i].b = np.random.default_rng().uniform(W_RANGE[0],W_RANGE[1],
                size=(self.num_layers[i+1], 1))
        elif len(weights) == len(self.num_layers)-1:
            for i in range(len(self.num_layers)-1):
                self.layer[i].w = weights[i]
                self.layer[i].b = bias[i]
    
    def forward_propagate(self,input_):
        self.layer[0].a = input_ #a = x, camada de entrada
        for i in range(1,len(self.num_layers)):
            self.layer[i].z = np.dot(self.layer[i-1].w,self.layer[i-1].a) #a*theta
            self.layer[i].a = sigmoid(self.layer[i].z + self.layer[i-1].b)

    def back_prop_error(self, y):
        self.layer[-1].delta = self.layer[-1].a - y
        for i in reversed(range(len(self.num_layers)-1)):
            if i == 0:
                break
            self.layer[i].delta = np.multiply(np.dot(self.layer[i].w.transpose(),self.layer[i+1].delta),
            sigmoid_der_v(self.layer[i].a))
    
    def gradient_eval(self):
        for i in reversed(range(len(self.num_layers)-1)):
            self.layer[i].w_grad = np.dot(self.layer[i+1].delta,self.layer[i].a.transpose())
            self.layer[i].b_grad = self.layer[i+1].delta

    def update_weights(self,l_rate):
        for i in reversed(range(len(self.num_layers)-1)):
            self.layer[i].w = self.layer[i].w - l_rate*self.layer[i].w_grad
            self.layer[i].b = self.layer[i].b - l_rate*self.layer[i].b_grad

    def train(self,inputs_,outputs):
        for k in range(len(inputs_)):
            self.forward_propagate(inputs_[k].reshape(inputs_[0].size,1))
            #print(inputs_[k].reshape(inputs_[0].size,1).shape)
            self.back_prop_error(outputs[k].reshape(outputs[0].size,1))
            self.gradient_eval()
            self.update_weights(0.05)


################ ESTA COM PROBLEMA NA ATUALIZACAO DOS PESOS, VERIFICAR GRADIENTE

t = MultiLayer([2,3,1])
t.initialize_network()
#print(t.layer[0].w)
#print(t.layer[1].w)

a = MultiLayer([2,1])
w = np.array([[20,20]]) #bias-10, w1 and w2 = 20
b = np.array([[-10]]).transpose() #MUST BE A COLUMN
a.initialize_network(w,b)
input_ = np.array([[1,1]]).transpose() #column
a.forward_propagate(input_)
print(a.layer[-1].a)
a.back_prop_error(np.array([[1]]).reshape(1,1))
print(a.layer[1].delta)

t.forward_propagate(input_)

t.back_prop_error(np.array([[1]]).reshape(1,1))
print(t.layer[2].a)
print(t.layer[2].delta)
print(t.layer[1].delta)
print(t.layer[0].delta)



def read_data(data,inputs):
    file = open(data,"r")
    x = []
    y = []
    for line in file:
        dummy = line.split(',')
        x.append([float(dummy[0]),float(dummy[1])])
        y.append(float(dummy[2]))
    file.close()
    x = np.array(x)
    y = np.array(y)
    return x,y


x,y = read_data("classification2.txt",2)

net = MultiLayer([2,2,2,1])
net.initialize_network()
'''
print(net.layer[0].w)
print(net.layer[0].b)

print('antes')
print('weights')
print(net.layer[0].w)
print(net.layer[1].w)
print('bias')
print(net.layer[0].b)
print(net.layer[1].b)
'''

net.train(x,y)

'''
print('depois')
print('weights')
print(net.layer[0].w)
print(net.layer[1].w)
print('bias')
print(net.layer[0].b)
print(net.layer[1].b)
'''

for i in range(len(x)):

    net.forward_propagate(np.array(x[i].reshape(2,1)))
    print(math.fabs(net.layer[-1].a-y[i]))
