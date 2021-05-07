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



W_RANGE = [-10, 10] #inicializa os pesos entre esses valores
class Layer:
    def __init__(self):
        self.a = None
        self.w = None #weights (neurons) from w_ij
        self.delta = None #error of neuron i at layer, nx1 matrix (n nodes)
        self.w_grad = None #partial derivatives of cost (error) wrt weights


class MultiLayer:
    def __init__(self,num_layers):
        self.layer = []
        self.num_layers = num_layers #neuron numbers on each layer
        for i in range(len(num_layers)):
            self.layer.append(Layer())
    def initialize_network(self,weights = None): #weights (o theta)
        if weights is None: #initialize random weights
            for i in range(len(self.num_layers)-1):
                self.layer[i].w = np.random.default_rng().uniform(W_RANGE[0],W_RANGE[1],
                size=(self.num_layers[i+1], self.num_layers[i]+1))
        elif len(weights) == len(self.num_layers)-1:
            for i in range(len(self.num_layers)-1):
                self.layer[i].w = weights[i]
    
    def forward_propagate(self,input_):
        self.layer[0].a = input_ #a = x, camada de entrada, entrar com um vetor coluna!!!!
        for i in range(1,len(self.num_layers)):
            x = np.insert(self.layer[i-1].a,0,1,axis = 0)
            z = np.dot(self.layer[i-1].w,x) #theta*a
            self.layer[i].a = sigmoid(z).reshape(z.size,1)

    def back_prop_error(self, y):
        self.layer[-1].delta = self.layer[-1].a - y
        for i in reversed(range(len(self.num_layers)-1)):
            if i == 0:
                break
            w = self.layer[i].w[:,1:] #adeus bias
            self.layer[i].delta = np.multiply(np.dot(w.transpose(),self.layer[i+1].delta),
            sigmoid_der_v(self.layer[i].a))
    
    def gradient_eval(self):
        for i in reversed(range(len(self.num_layers)-1)):
            a = np.insert(self.layer[i].a,0,1,axis = 0)
            self.layer[i].w_grad = np.dot(self.layer[i+1].delta,a.transpose())

    def update_weights(self,l_rate): ###nao esta sendo usado
        for i in reversed(range(len(self.num_layers)-1)):
            self.layer[i].w = self.layer[i].w - l_rate*self.layer[i].w_grad #errado
            self.layer[i].b = self.layer[i].b - l_rate*self.layer[i].b_grad #errado

    def train(self,inputs_,outputs): #adicionar loop de epochs
        for i in range(100): #epochs
            delta_w = []
            delta_b = []
            #print(self.layer[0].w)
            for s in range(len(self.num_layers)-1):
                delta_w.append(np.zeros((self.num_layers[s+1],self.num_layers[s]+1)))
            for k in range(len(inputs_)):
                self.forward_propagate(inputs_[k].reshape(inputs_[0].size,1))
                self.back_prop_error(outputs[k].reshape(outputs[0].size,1))
                #print(outputs[k].reshape(outputs[0].size,1))
                #print(self.layer[-1].delta[-1])
                self.gradient_eval()
                #print(self.layer[0].w_grad)
                #print('---')
                for g in range(len(self.num_layers)-1):
                    delta_w[g] = delta_w[g] + self.layer[g].w_grad #soma dos gradientes

            for h in range(len(self.num_layers)-1):
                d_weights = delta_w[h]
                r = 0.5*self.layer[h].w
                r[:,0] = 0.0 #no regularization for bias
                self.layer[h].w -= (0.001 * (d_weights/len(inputs_)) + r)



                    


################ ESTA COM PROBLEMA NA ATUALIZACAO DOS PESOS, VERIFICAR GRADIENTE

''' porta or, descomentar para ver os valores
a = MultiLayer([2,1])
w = np.array([[-10,20,20]]) #bias-10, w1 and w2 = 20
a.initialize_network(w)
input_ = np.array([[0,1]]).transpose() #column
a.forward_propagate(input_)
print(a.layer[-1].a)
a.forward_propagate(np.array([[0,0]]).reshape(-1,1))



a.back_prop_error(np.array([[1]]).reshape(1,1))
print(a.layer[1].delta)
'''



''' not xor
w = []
w1 = np.array([[-30,20,20],[10,-20,-20]]).reshape(2,-1)
w2 = np.array([[-10,20,20]]).reshape(1,-1)
w.append(w1)
w.append(w2)
xor = MultiLayer([2,2,1]) #rede neural com 2 neuronios na entrada, 2 no meio, 1 saida
xor.initialize_network(w)
entrada = np.array([[0,1]]).reshape(-1,1)
xor.forward_propagate(entrada)
print(xor.layer[-1].a) #saida da rede neural
'''


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


net = MultiLayer([2,4,4,1])
net.initialize_network()


#nao foram separados o conjunto (entre teste e para treinamento)
#so pra ver se o treino esta funcionando (nao esta)
net.train(x,y)

acertos = 0
for i in range(len(x)):

    net.forward_propagate(x[i].reshape(2,1))
    #print(x[i].reshape(2,1))
    d = 0
    if net.layer[-1].a < 0.5:
        d = 0
    elif net.layer[-1].a > 0.5:
        d = 1
    if math.fabs(d-y[i]) < 0.1:
        acertos +=1
    print("%d esperado %f" %(d,y[i]))

print("acertos %d" %acertos)
print(acertos/128)
