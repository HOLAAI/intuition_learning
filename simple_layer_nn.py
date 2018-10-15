##### 2 layer (1 hidden layer) Neural Network 구현
##### XOR 문제를 학습하는 예제 구성
#### vectorization 구현  하며 이론은
##### https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ , https://en.wikipedia.org/wiki/Backpropagation 기준으로 구현
##### TODO 1. layer 더 추가해 보기 2. bais 값을 각 neural 별로 주었는데 layer 별로 공통된 값 사용할수 있게 변경

import numpy as np

#### 시그모이드 함수 구현
def sigmoid(x) :
    return  1 / (1 + np.exp(-x))


input_layer_n = 2
hidden_layer_n = 3
output_layer_n = 1

### XOR 로직을 테스트 할수 있게 입력값 제공 , X[0] 는 vectorization 구현시 bias 값을 위한 셋팅
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
#Y = np.array([[0], [1], [1], [1]])
Y = np.array([[0], [1], [1], [0]])

W1 = np.random.rand(input_layer_n + 1, hidden_layer_n)
W2 = np.random.rand(hidden_layer_n, output_layer_n)

learn_rate = 0.03

for t in range(10000) :
    dot1 = np.dot(X, W1)
    layer1 = sigmoid(dot1)
    dot2 = np.dot(layer1, W2)
    output = sigmoid(dot2)

    cost = np.sum((output - Y) ** 2)

    ### 에러 값과 결과 값을 출력해 본다 .
    print(cost, output[0], output[1], output[2], output[3])

    gred_output = output - Y
    gred_sigmoid = output * (1 - output)

    gred_w2 = layer1.T.dot(gred_output * gred_sigmoid)

    grad_h = gred_output.dot(W2.T)

    gred_w1 = X.T.dot((grad_h * layer1 * (1 - layer1)))

    W1 -= learn_rate * gred_w1
    W2 -= learn_rate * gred_w2

