### XOR 문제를 3 layer neural network 로 구
import numpy as np

#### 시그모이드 함수 구현
def sigmoid(x) :
    return  1 / (1 + np.exp(-x))


def costf(target , out) :
    return np.sum((target - out) ** 2)


def forwordpass(data_in, W):
    return data_in.dot(W)


input_layer_n = 2
hidden_layer1_n = 3
hidden_layer2_n = 3

output_layer_n = 1

### XOR 로직을 테스트 할수 있게 입력값 제공 , X[0] 는 vectorization 구현시 bias 값을 위한 셋팅
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#Y = np.array([[0], [1], [1], [1]])
Y = np.array([[0], [1], [1], [0]])
#Y = np.array([[0], [0], [0], [1]])

W1 = np.random.rand(input_layer_n, hidden_layer1_n)
W2 = np.random.rand(hidden_layer1_n, hidden_layer2_n )
W3 = np.random.rand(hidden_layer2_n , output_layer_n)

learn_rate = 0.1


for t in range(5000) :
    h1 = forwordpass(X, W1)
    h1 = sigmoid(h1)
    h2 = forwordpass(h1, W2)
    h2 = sigmoid(h2)
    output = forwordpass(h2, W3)

    cost = costf(Y, output)

    print(cost, np.round(output, 4))

    ### partial derivative of error with respect to output
    gred_output = output - Y
    gred_w3 = h2.T.dot(gred_output)
    gred_w2 = h1.T.dot(gred_output.dot(W3.T) * ((1 - h2) * h2))
    gred_w1 = (gred_output.dot(W3.T).dot(W2.T) * ((1 - h1) * h1))
    gred_w1 = X.T.dot(gred_w1)
    #print(gred_w1.shape , W1.shape, X.shape)
    ##gred_w2 = gred_output * W3 * (1 - h2) * h2 * h1
    #gred_w1 = gred_output.dot(W3.T).dot(W2) * (1 - h1) * h1
    #gred_w1 = X.T.dot(gred_w1)
    ###print(gred_w1)
    W1 -= learn_rate * gred_w1
    W2 -= learn_rate * gred_w2
    W3 -= learn_rate * gred_w3




#print(gred_w1.shape, W1.shape, W3.shape)




#print(gred_w2)

#print(gred_w2.shape, W2.shape, h1.shape)

#gred_h2 =




