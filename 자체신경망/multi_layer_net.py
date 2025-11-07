import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

# (Affine, Relu, SoftmaxWithLoss 클래스가 import 되었다고 가정)

class MultiLayerNet:
    """
    input_size : 입력 크기 (예: 784)
    hidden_size_list : 은닉층 뉴런 수를 담은 리스트 (예: [100, 50])
    output_size : 출력 크기 (예: 10)
    """
    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list) # 은닉층 개수 (예: 2)
        self.params = {} # 가중치(W), 편향(b)을 보관하는 딕셔너리

        # -----------------------------------------------------------------
        # 1. 가중치(W, b) 초기화
        # -----------------------------------------------------------------
        
        # [784, 100, 50] 리스트 생성 (입력층 + 은닉층)
        all_sizes_list = [self.input_size] + self.hidden_size_list 
        
        # W1, b1, W2, b2 ... 생성 (for문 사용)
        for idx in range(1, len(all_sizes_list)): # 1부터 2까지 (idx=1, idx=2)
            scale = weight_init_std
            
            # W1, b1 (idx=1) : all_sizes_list[0](784) -> all_sizes_list[1](100)
            # W2, b2 (idx=2) : all_sizes_list[1](100) -> all_sizes_list[2](50)
            self.params['W' + str(idx)] = scale * np.random.randn(all_sizes_list[idx-1], all_sizes_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_sizes_list[idx])

        # 마지막 W3, b3 (출력층) 생성
        # (idx=3) : all_sizes_list[2](50) -> output_size(10)
        idx = self.hidden_layer_num + 1 # 3
        self.params['W' + str(idx)] = scale * np.random.randn(all_sizes_list[-1], self.output_size)
        self.params['b' + str(idx)] = np.zeros(self.output_size)

        # -----------------------------------------------------------------
        # 2. 계층(Layer) 생성
        # -----------------------------------------------------------------
        self.layers = OrderedDict() # 순서가 있는 딕셔너리
        
        # Affine1 -> Relu1 -> Affine2 -> Relu2 ... 생성 (for문 사용)
        for idx in range(1, self.hidden_layer_num + 1): # 1부터 2까지 (idx=1, idx=2)
            # Affine1, Affine2
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # Relu1, Relu2
            self.layers['Relu' + str(idx)] = Relu()

        # 마지막 Affine3 (출력층) 생성
        idx = self.hidden_layer_num + 1 # 3
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        
        # 마지막 SoftmaxWithLoss 계층
        self.lastLayer = SoftmaxWithLoss()
        
    

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        # t = np.argmax(t, axis = 1) # t가 원핫인코딩 숫자 레이블이 아님

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    
    def gradient(self, x, t):
        # 순전파
        self.loss(x,t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        
        # 예: hidden_layer_num = 2 (3층) -> range(1, 4) -> idx는 1, 2, 3
        # 예: hidden_layer_num = 1 (2층) -> range(1, 3) -> idx는 1, 2
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads