# ##### 혼공머신 챕터 3 - 1<br>
# 예측자료를 추측 해 내기 위한 회귀 (regression) 방법 사용

# +
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
# -
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
perch_length, perch_weight, random_state=42)

# numpy 에서는 reshape를 편하게 할 수 있기 때문에, 필요한 데이터가 있을 경우 알아서 변경 해서 사용하도록 하자.

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)

# +
knr = KNeighborsRegressor()

# k-최근접 이웃 회귀 모델을 훈련함
knr.fit(train_input, train_target)
# -

print(knr.score(test_input, test_target))

from sklearn.metrics import mean_absolute_error
# 테스트 세트에 대한 예측을 만듭니다
test_prediction = knr.predict(test_input)
# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

print(knr.score(train_input, train_target))

# +
knr.n_neighbors = 3

knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))

# -

print(knr.score(test_input, test_target))

# #### 핵심 포인트
# 회귀는 임의의 수치를 예측하는 문제.
# k-최근접 이웃 회귀는 k-최근접 이웃 알고리즘을 사용해 회귀 문제를 풉니다. 가장 가까운 샘플을 찾고 이 샘플들의 타깃값을 평균하여 예측으로 삼습니
# **결정계수(R의 제곱)**는 대표적인 회귀 문제의 성능 측정 도구
#


