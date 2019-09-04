from keras.datasets import boston_housing

(train_data, train_targets), (test_data,test_targets) = boston_housing.load_data()

print(train_data.shape) #13개의 특성 1인당 범죄율, 주택당 평균 방의 개수, 고속도로 접근성 등등
print(test_data.shape)

train_targets #주택의 중간가격 천단위달러

#정규화 하는 과정
mean = train_data.mean(axis=0) #축0 기준으로 0의 기준으로 평균값을 잼
train_data -= mean # 평균을 0에 맞추도록 평균을 빼줌
std = train_data.std(axis=0) #축0 기준으로 0의 기준으로 분산값을 잼
train_data /= std#최대값이 1이 되도록 분산값을 나눔

test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',
                          input_shape=(train_data.shape[1],))) #특성의 갯수 만큼 크기
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1)) #활성화 함수가 없음, 선형층이라고 부르고 전형적인 스칼라 회귀라고 함
    #만일 sigmoid 로 활성화함수를 두면 0~1사이의 값을 예측하도록 할것입니다.
    
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae']) #평균제곱오차 손실함수를 사용하여 컴파일함.(회귀문제에서 널리사용)
    #훈련하는 동안은 모니터링을 위해 새로운 지표인 평균절대오차를 측정(예측과 타깃사이 거리의 절대값)
    #MAE가 0.5면 예측이 평균적으로 500달러 정도 차이난다는 뜻 target 기준이 1000달라이므로
    return model



# K-겹 교차검증
# 모델을 평가하기 위해 이전 예제에서 했던 것 처럼 데이터를 훈련셋과 검증셋으로 나눔.
# 데이터 갯수가 많지 않기에 검증 셋도 매우 작아짐. 
# 검증 셋과 훈련 셋의 어떤 데이터가 선택 되었는지에 따라 검증 점수가 크게 달라짐
# 각 검증셋 분할에 대한 검증 점수 분산이 높아짐 
#이렇게되면 신뢰있는 평가가 아님

# 이런 상황에서 좋은 방법이 K-겹 교차검증
# 데이터를 K개로 나누어 K개의 모델을 각각 만든 후 K-1개의 분할에서 훈련
# 나머지 분할에서 검증 평가하는 방법 

# 모델의 검증 점수는 K개의 검증 점수의 평균이 됨

import numpy as np

k = 4

num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('처리중인 폴드 #', i)
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples] # 검증 데이터 인덱스의 1/4 만큼 잘라 순차적으로 적용
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
    
    partial_train_data = np.concatenate( # 훈련데이터 인덱스의 3/4만큼을 비중 두기위함
    [train_data[ : i*num_val_samples],
     train_data[(i+1)*num_val_samples :]],
     axis=0)
    
    partial_train_targets = np.concatenate(
    [train_targets[ : i*num_val_samples],
     train_targets[(i+1)*num_val_samples :]],
     axis=0)
    
    model = build_model() #모델을 만드는 함수
    model.fit(partial_train_data, partial_train_targets,
             epochs=num_epochs, batch_size=1, verbose=0) #각 K교차마다 훈련시킴
    #epochs는 100이고 batch_size는 1개씩, verbose=0이므로 훈련 과정이 출력되지 않음
    
    val_mse, val_mae = model.evaluate(val_data,val_targets, verbose=0)#각 완료 될때마다 검증세트 모델 평가
    
    all_scores.append(val_mae)#모델 평가한 평균절대오차값을 all_scores리스트에 푸쉬함


print(all_scores)

print(np.mean(all_scores)) #약 평균 2531달러 정도 오차가 있음

num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('처리중인 폴드 #', i)
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples] # 검증 데이터 인덱스의 1/4 만큼 잘라 순차적으로 적용
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
    
    partial_train_data = np.concatenate( # 훈련데이터 인덱스의 3/4만큼을 비중 두기위함
    [train_data[ : i*num_val_samples],
     train_data[(i+1)*num_val_samples :]],
     axis=0)
    
    partial_train_targets = np.concatenate(
    [train_targets[ : i*num_val_samples],
     train_targets[(i+1)*num_val_samples :]],
     axis=0)
    
    model = build_model() #모델을 만드는 함수
    history = model.fit(partial_train_data, partial_train_targets,
              validation_data=(val_data, val_targets), #평가할 데이터
              #model.evaluate(val_data,val_targets, verbose=0) 의 옵션
             epochs=num_epochs, batch_size=1, verbose=0) #각 K교차마다 훈련시킴
    #epochs는 500이고 batch_size는 1개씩, verbose=0이므로 훈련 과정이 출력되지 않음
    
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)#모델 평가한 평균절대오차값을 all_mae_histories리스트에 푸쉬함

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# 각 epoch 마다 모든 폴드의 MAE평균값을 나열
# 500epochs 각각 4개의 폴드에 대한 MAE 평균

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous =smoothed_points[-1]
            smoothed_points.append(previous*factor + point*(1-factor)) #지수 이동 평균으로 대체
        else:
            smoothed_points.append(point)
            
    return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:]) #변동이 너무 심한 10개의 epochs는 제외

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)

plt.xlabel('Epochs')
plt.ylabel('Validation MAE')

plt.show()

model = build_model()
model.fit(train_data,train_targets,
         epochs=80, batch_size=16,verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data,test_targets)

test_mae_score

#회귀에서는 분류에서 사용했던 것과는 다른 손실 함수를 사용합니다.
#평균제곱오차는 회귀에서 자주 사용되는 손실함수 입니다.
#입력 데이터의 특성이 서로 다른 범위를 가지면 전처리 단계에서 각 특성을 개별적으로 스케일 조정해야함(정규화)
#정확도 개념은 회귀에 적용되지 않음 , 일반적으로 회귀지표는 평균절대오차MAE입니다.
#가용한 데이터가 적으면 K-겹 검증을 사용
#가용한 훈련 데이터가 적으면 과적합을 피하기 위해 은닉 층의 수를 줄인 모델이 좋음(1~2개)

#2445달러 차이로 줄어듬
