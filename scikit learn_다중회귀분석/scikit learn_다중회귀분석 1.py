from sklearn import linear_model # scitkit learn 모듈 내 linear_model 을 불러옴

import numpy as np # numpy 모듈을 np로 이름 변경
#파이썬 언어를 위한 행렬, 벡터 등의 수학 계산을 위한 자료구조와 계산 함수를 제공하는 패키지

import pandas as pd # pandas 모듈을 pd로 이름 변경
#데이터 분석, 가공, 처리 등을 쉽게 하기 위한 자료구조와 처리 함수들을 제공하는 패키지

import matplotlib
#플롯(그래프)를 그릴 때 주로 쓰이는 2D,3D 플롯팅 패키지

import matplotlib.pyplot as plt
#matplotlib의 서브 패키지로 Matlab 처럼 플롯을 그려주는 패키지

from sklearn.metrics import mean_squared_error
#sklearn 패키지에서 제공하는 MSE를 구하기 위한 모듈

%matplotlib inline 
# matplotlib의 시각화 결과를 ipython notebook 내에서 출력하게 하는 함수
matplotlib.style.use('ggplot') #matplotlib 패키지에서 제공하는 스타일 중 ggplot 을 지정

data = {'x1' : [13, 18, 17, 20, 22, 21],
        'x2' : [9, 7, 17, 11, 8, 10],
        'y' :  [20, 22, 30, 27, 35, 32]}
# data set 생성

data=pd.DataFrame(data) # 컬럼 데이터 x를 pandas dataframe으로 적용함
X = data[['x1','x2']]
y = data['y']

data

linear_regression = linear_model.LinearRegression() #선형회귀분석 모델을 저장
linear_regression.fit(X=pd.DataFrame(X), y=y) # X와 y를 선형회귀분석 모델에 학습시킴
prediction = linear_regression.predict(X=pd.DataFrame(X)) #X값에 따른 새로운 값을 예측하여 저장
print('a value = ', linear_regression.intercept_)#선형회귀분석식의 a계수 출력
print('b value = ', linear_regression.coef_)#선형회귀분석식의 b계수 출력

# y= a + bX1 + bX2 + e

#y는 특정 관측치(예상값)에 대한 종속변수의 실제값
#X는 이미 알려진 독립변수의 값
#a는 X값이 변해도 Y값에는 영향을 주지 않는 회귀 계수
#b는 X의 영향력을 크기와 부호로 나타내 주는 회귀 계수, 독립변수X의 기울기
#e는 특정 관측치(예상값)과 실제값의 오차항

residuals = y - prediction # 실제 값에서 예측값을 뺀 잔차값을 저장
residuals.describe() #잔차값들의 통계를 요약

SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_sqiared = ', R_squared)#R의 제곱 즉 결정계수 제곱근 (1에 가까울수록 실제값을 예측하는 정확성 높아짐)

#numpy.sum()
#numpy.mean()

#SSE = 오차 제곱 합 (각 오차(실제값-예측값)의 제곱들의 합), 즉 실제값과 예측값이 어느정도 오차가 있는지의 정도
#SST = (실제값-실제평균값)의 제곱 합, 즉 실제값들이 실제평균으로부터 흩어진 정도
#SSR = (예측치-실제평균값)의 제곱 합, 즉 예측값들이 실제평균으로부터 흩어진 정도
#결정계수 R^2 = SSR/SST = 1 - (SSE/SST)

#from sklearn.metrics import mean_squared_error
print('score = ', linear_regression.score(X=pd.DataFrame(X),y=y))
#2차원 X의 dataFrame 형태와 y로 지정하여 학습한 모델을 통해 성능 평가(결정계수)
print('Mean_squared_Error = ', mean_squared_error(prediction, y))
#학습한 모델을 통해 나온 예측값과 실제값의 평균제곱오차을 구함 
print('RMSE = ', mean_squared_error(prediction, y)**0.5)
#MSE를 루트 적용한 값 RMSE
