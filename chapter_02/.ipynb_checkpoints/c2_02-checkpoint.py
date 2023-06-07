# ### Chapter 2-2 섹션의 의미<br>
# <br>
# 이 곳에서는 도미와 빙어를 구분하는 코드를 작성하였는데,<br>
# predict를 한 값이 도미가 아닌 빙어로 나오는 경우에 대한 문제 해결을 다루고 있다<br>
# <br>
# 원인은, 서로 다른 특성에 대한 값을 구분지을 경우, 최근접 로직을 이용 할 경우 문제가 생기는 데 있다<br>
# 이러한 경우에 대해 해결하는 방법을 순차적으로 보여주고 있는 코드이다.<br>
# 순서대로 내용을 읽어보면 어떤 내용인지 확용이 가능할 것 같다.

# #### Import Section

# +
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
# -

# Fish Data

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


# numpy 활용. 이렇게 편한기능인 걸 보면, tensorflow는 어떤 정도일까?<br>
# 아래 코드는 기존에 이렇게 활용했었다.<br>
# <br>
# `fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
# fish_target = [1]*35 + [0]*14`
#

# fish_data = np.column_stack(fish_length, fish_weight)

# 위 내용 중에서 주의해야 할 내용이 있는데,<br>
# `fish_data = np.column_stack(fish_length, fish_weight)`<br>
# 위 처럼 괄호 하나를 하면, 입력값이 하나여야 하는데 두개나 된다면서 오류를 낸다.<br>
# <br>
# 위 내용에 대해서는 나중에 확인이 되는대로 내용 보충하도록 하자.

fish_data = np.column_stack((fish_length, fish_weight))

print(fish_data[:5])

fish_target = np.concatenate((np.ones(35), np.zeros(14)))

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

print(test_target)

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

print(kn.predict([[25, 150]]))

# 아래 코드 중 scatter에 인자값을 보면 다음과 같다.<br>
# `train_input[:,0], train_input[:,1]`<br>
# <br>
# 위 내용 중, ',' 다음에 들어가는 자료는 열(column)을 지칭하는 것이다.<br>
# traininput 데이터는 [길이, 무게] 형태의 행렬로 이루어져 있으므로<br>
# scatter 함수의 x, y 값에 대응하도록 패턴을 나누어 입력 한 내용이다

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

distance, indexes = kn.kneighbors([[25, 150]])

# #### 실제 그래프 상에서는 도미에 가까우나, 최근접으로 보면 빙어가 더 많이 선택 됨

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(train_input[indexes])
print(train_target[indexes])

print(distance)

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# ##### 데이터 전처리 작업
# <br>
# - 위의 경우 weight와 length에 대한 두 가지 자료의 scale이 다른 면이 있음<br>
# - 이런 경우 각 특성치를 전처리 하여 동일한 scale을 가지도록 유도 해 주는 것이 중요함<br>
# - 위 내용을 처리하는 가장 흔한 방법 중 하나는 '표준 점수'임<br>
# - 표준점수는 각 특성값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 나타내어줌<br>

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

# 위에서 중요한 내용은, axis=0 을 지정했을 경우 각 열에 대한 값을 계산 해 준다는 의미이다.
# 그러므로, mean, std의 값은 각 열에 대해 따로 추출이 된다.
# 아래 코드를 확인 해 보자<br>
# 위 함수를 수행 할 경우, 각 열에 대해서 평균값 2개, 표준편차값 2개가 표시된다.

print(mean, std)

train_scaled = (train_input-mean) / std

print(train_scaled)

new = ([25, 150] -mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) / std

kn.score(test_scaled, test_target)

#print([new])
print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
#plt.xlim(0, 1000)
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# #### 핵심 포인트
# ##### 데이터 전처리
# 데이터 전처리는 머신러닝 모델에 훈련 데이터를 주입하기 전에 가공하는 단계를 말함. 때로는 전처리에 많은 시간이 소요되기도 한다.
# #### 표준 점수
# 훈련 세트의 스케일을 바꾸는 대표적인 방법 중 하나입니다. 표준 점수를 얻으려면 특성의 평균을 빼고 표준편차로 나눕니다. 반드시 훈련 세트의 평균과 표준편차로 테스트 세트를 바꿔야 합니다.
# #### 브로드캐스팅
# 크기가 다른 넘파이 배열에서 자동으로 사칙 연산을 모든 행이나 열로 확장하여 수행하는 기능입니다


