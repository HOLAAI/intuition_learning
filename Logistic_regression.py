### logistic regression 을 구현한 코드
### 해당 코드에서 사용된 수식과 이론은 https://www.coursera.org/learn/machine-learning 을 바탕으로 한다
### 해당코드는 코드의 효율성 보다는 logistic regression 의 cost 함수  , 최적화 부분에서 직관력을 가지기 위해 작성한 코드이다
### vectorized implementation 에 대한 이해와 연습을 위주로 한다

import numpy as np

### 시그모이드 함수를 정의한다 classification 에서는 시그모이드 함수를 active function 으로 사용한다
def sigmoid(x) :
    return  1 / (1 + np.exp(-x))

### liner hypothesis function  + active function
def hypothesis(X, theta) :
    return sigmoid(np.dot(X, theta.T))



### cost function 과 gradient descent 기능 구현
def cost(X, y, theta, learning_rate) :
    m = X.shape[0]

    ## cost 구현 시작
    ### 1/m * (-y.T*log(h) - (1 - y).T*log(1-h))
    h = hypothesis(X, theta)
    log_values = np.log(h)
    true_term = np.dot(-y.T, log_values)
    temp = (1 - y)
    false_term = np.dot(temp.T, np.log(1- h))
    cost = 1/m * (true_term - false_term)
    ## cost 구현 완료

    ## gradient Descent 구현
    ## θ = θ - α/m * X.T * (g(X * θ) - y)
    grad  = np.dot(learning_rate * 1/m * X.T, (h - np.reshape(y, (4, 1))))
    return (cost, grad.T)



###  AND LOGIC 을 구현
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 0, 0, 1])
theta = np.random.rand(1, 3)

### learning
for i in range(5000) :
    cost_v, grad = cost(X, y, theta, 0.05)
    theta = theta - grad
    print(i, "=>", cost_v, grad, theta)


##predict 1 AND 0 = 0, 예측값은 maybe  < 0.5, 0에 가까운 값
print(hypothesis([[1, 1, 0]], theta))