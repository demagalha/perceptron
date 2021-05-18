#ANDRE DEMETRIO DE MAGALHAES E BRUNO HASHINOKUTI IWAMOTO
#TRABALHO DE INTELIGENCIA ARTIFICIAL APLICADA A CONTROLE E AUTOMACAO
#TREINAMENTO DE REDES NEURAIS
####
import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_der(a):
    return a*(1-a)

sigmoid_der_v = np.vectorize(sigmoid_der)
l_rate = 0.05


W_RANGE = [-5, 5] #inicializa os pesos entre esses valores

class Layer:
    def __init__(self):
        self.a = None
        self.w = None #weights (pesos) w_ij, para i de j
        self.b = None #bias weights (pesos dos bias)
        self.delta = None #erro do neuronio i na camada, nx1 matrix (n nodes)
        self.w_grad = None #derivadas parciais em relacao aos weights
        self.b_grad = None #derivadas parciais em relacao aos biases


class Network:
    def __init__(self,num_layers):
        self.layer = []
        self.num_layers = num_layers #numero neuronios por camada 
        #eg: [2,10,2], 2 na primeira, 10 na segunda, 2 na ultima
        for i in range(len(num_layers)):
            self.layer.append(Layer())
    def initialize_network(self,weights = None, bias = None):
        if weights is None: #random weights
            for i in range(len(self.num_layers)-1):
                self.layer[i].w = np.random.uniform(W_RANGE[0],W_RANGE[1],
                size=(self.num_layers[i+1], self.num_layers[i]))
                self.layer[i].b = np.random.uniform(W_RANGE[0],W_RANGE[1],
                size=(self.num_layers[i+1], 1))
        elif len(weights) == len(self.num_layers)-1:
            for i in range(len(self.num_layers)-1):
                self.layer[i].w = weights[i]
                self.layer[i].b = bias[i]
    
    def forward_propagate(self,input_):
        self.layer[0].a = input_ #a = x, camada de entrada
        for i in range(1,len(self.num_layers)):
            z = np.dot(self.layer[i-1].w,self.layer[i-1].a) #a*theta
            self.layer[i].a = sigmoid(z + self.layer[i-1].b) #add bias, ja que w e b sao separados

    def back_prop_error(self, y):
        erro = -(y/self.layer[-1].a + (-1+y)/(1-self.layer[-1].a))
        deriv = self.layer[-1].a * (1-self.layer[-1].a) #sigmoid_derv_v(self.layer[-1].a)
        self.layer[-1].delta = erro * deriv
        #delta da ultima camada de acordo com ultima aula
        #descomentar a baixo caso necessario
        #self.layer[-1].delta = self.layer[-1].a - y
        for i in reversed(range(len(self.num_layers)-1)):
            if i == 0:
                break
            self.layer[i].delta = np.multiply(np.dot(self.layer[i].w.transpose(),self.layer[i+1].delta),
            self.layer[i].a * (1 - self.layer[i].a))
    
    def gradient_eval(self):
        for i in reversed(range(len(self.num_layers)-1)):
            self.layer[i].w_grad = np.dot(self.layer[i+1].delta,self.layer[i].a.transpose())
            self.layer[i].b_grad = self.layer[i+1].delta

    def train(self,inputs_,outputs):
        custo = []
        for epoch in range(1000): #epochs
            delta_w = []
            delta_b = []
            for k in range(len(self.num_layers)-1):
                delta_w.append(np.zeros((self.num_layers[k+1],self.num_layers[k])))
                delta_b.append(np.zeros((self.num_layers[k+1],1)))
            for k in range(len(inputs_)):
                self.forward_propagate(inputs_[k].reshape(-1,1))
                self.back_prop_error(outputs[k].reshape(-1,1))
                self.gradient_eval() #a_j * delta_i para todas as camadas
                for g in range(len(self.num_layers)-1):
                    delta_w[g] += self.layer[g].w_grad
                    delta_b[g] += self.layer[g].b_grad
            for k in range(len(self.num_layers)-1):
                D_w = ((1/len(inputs_))*delta_w[k])
                D_b = ((1/len(inputs_))*delta_b[k])
                self.layer[k].w -=  l_rate*D_w
                self.layer[k].b -= l_rate*D_b
            custo.append(Jent(self,inputs_,outputs))
        plt.plot(custo)
        plt.show()




 
def cost(h,y):
    return -(y*np.log(h) + (1-y)*np.log(1-h))  

def Jent(net,x,y):
    soma = 0
    for i in range(len(x)):
        net.forward_propagate(x[i].reshape(-1,1))
        soma += cost(net.layer[-1].a,y[i])
    return ((1/len(x)) * soma).item(0)


def fronteira(net):
    x1s = np.linspace(-1,1.5,50)
    x2s = np.linspace(-1,1.5,50)
    z=np.zeros((len(x1s),len(x2s)))

    for i in range(len(x1s)):
        for j in range(len(x2s)):
            net.forward_propagate(np.array([[x1s[i],x2s[j]]]).reshape(-1,1))
            z[i,j] = net.layer[-1].a[0]

    df=pd.read_csv("classification2.txt", header=None)
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1)
    plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
    plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
    plt.contour(x1s,x2s,z.T,0)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc=0)
    plt.show()


'''
#porta logica (not x1) AND (not x2)
#apenas para testar o forward_propagate
#esta ok
a = Network([2,1]) #cria uma Network (rede neural) com 2 neuronios na primeira e 1 na ultima camada
w = np.array([[-20,-20]]) #bias10, w1 and w2 = 20
b = np.array([[10]]).transpose() #tem que ser coluna
a.initialize_network(w,b) #inicializa os pesos com w e b especificados
input_ = np.array([[0,0]]).transpose() #coluna
a.forward_propagate(input_)
print(a.layer[-1].a) #saida da rede neural
'''




#funcao para ler os dados fornecidos
#poderia usar o pandas, mas desse jeito consegui fazer um embaralhar os dados
#para o treinamento
def read_data(data):
    file = open(data,"r")
    x = []
    for line in file:
        dummy = line.split(',')
        x.append([float(dummy[0]),float(dummy[1]), float(dummy[2])])
    file.close()
    x = np.array(x)
    return x


x_ = read_data("classification2.txt")
np.random.shuffle(x_) #embaralhar
x = []
y = []
for i in range(len(x_)):
    x.append([x_[i,0],x_[i,1]])
    y.append([x_[i,2]])

x = np.array(x)
y = np.array(y)


net = Network([2,30,30,30,1]) #cria nossa rede neural
net.initialize_network() #inicializa os pesos aleatoriamente

net.train(x[0:90],y[0:90]) #treina nossa rede com os dados de 0:90, ou seja, nosso conjunto de treinamento
#vou validar com x[90:] e y[90:]
#ie, separando os dados entre conjunto de treinamento e validacao


acertos = 0
x_ = x[90:] #validacao
y_ = y[90:]
for i in range(len(x_)):

    net.forward_propagate(x_[i].reshape(-1,1))
    d = 0
    if net.layer[-1].a < 0.5:
        d = 0
    elif net.layer[-1].a > 0.5:
        d = 1
    if math.fabs(d-y_[i]) < 0.1:
        acertos +=1
    #print("%d esperado %f" %(d,y[i]))
    #print(net.layer[-1].a)

#print("acertos %d" %acertos)
print(acertos/len(x_))

fronteira(net)
