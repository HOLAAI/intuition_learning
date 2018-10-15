### 해당 코드에서 사용된 수식과 이론은 https://www.coursera.org/learn/machine-learning 을 바탕으로 한다
### 해당코드는 코드의 효율성 보다는 regression 의 cost 함수, 최적화 부분에서 직관력을 가지기 위해 작성한 코드이다
### vectorized implementation 에 대한 이해와 연습을 위주로 한다

import numpy as np


### test 를 위하여 실제 맞는 y 값을 생성하는 함수
def real_h(X) :
    results = []
    for row in X :
        results.append(row[1] * 2 + row[2] * 3 + row[3] * 4 + row[4] * 5 + 2)
    return results



### x[0] is theta0,  bias 를 만들어야 하기 때문에 X[0] = 1
X = np.array([
    [1, 1, 2, 3, 4],
    [1, 5, 6, 7, 8],
    [1, 3, 2, 1, 2],
    [1, 5, 1, 1, 2]
])

y = real_h(X)

### theta random 초기화
theta = np.random.rand(1, 5)

### liner 함수로 가설 함수 구현
hypothesis = lambda  theta, X :  np.dot(theta, X.T)


### learning
### epoch start
learning_rate = 0.0250
for i in range(1000) :
    # 예측값 생성
    Y_hat = hypothesis(theta, X)
    Y_diff= Y_hat - y
    temp = np.dot(Y_diff, X)
    theta = theta -  learning_rate * (temp / X.shape[0])
    print("==> error", np.sum(Y_diff) ** 2, theta)

### epoch end

### 학습된 theta 값으로 예측 테스트
print("predict test : ", hypothesis(theta, np.array([[1, 1, 2, 3, 4]])))

