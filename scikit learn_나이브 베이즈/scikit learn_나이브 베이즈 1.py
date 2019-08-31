#라플라스 스무딩 : 새로운 입력벡터를 입력 했을 때 조건부 확률이 0이 되어 계속 0인 상태의 오류를 고치기 위해 정상적으로 분류 되지않는 경우를 방지하기 위해 확률 값을 보정하는데 사용하는 기법
#log 변환 : 확률의 값은 항상 1보다 작은 소수점값이므로 입력벡터를 구성하는 요소가 많아질 수록 조건부확률에서의 곱셈에 의해 확률이 매우 작아져 조건부 확률 값의 비교 불가능한 underflow 발생 이를 위해 조건부 확률식에 log를 적용하여 확률 값을 연산

from sklearn.model_selection import train_test_split #dataset을 train과 test set으로 분리하기 위한 모듈
from sklearn.naive_bayes import GaussianNB #나이브베이즈 기반 모델ㅇ리 있는 서브 패키지
#가우시안은 연속적인 값을 지닌 데이터를 처리할 때 각 클래스의 연속적인 값들이 가우스분포(정규분포)를 가정하는 나이브 베이즈 모듈 중 하나

import pandas as pd
import numpy as np

tennis_data = pd.read_csv('dataset/playtennis.csv')
tennis_data

tennis_data.outlook = tennis_data.outlook.replace('sunny', 0)
tennis_data.outlook = tennis_data.outlook.replace('overcast', 1)
tennis_data.outlook = tennis_data.outlook.replace('rainy', 2)

tennis_data.temp = tennis_data.temp.replace('hot', 3)
tennis_data.temp = tennis_data.temp.replace('mild', 4)
tennis_data.temp = tennis_data.temp.replace('cool', 5)

tennis_data.humidity = tennis_data.humidity.replace('high', 6)
tennis_data.humidity = tennis_data.humidity.replace('normal', 7)

tennis_data.windy = tennis_data.windy.replace('weak', 8)
tennis_data.windy = tennis_data.windy.replace('strong', 9)

tennis_data.play = tennis_data.play.replace('no', 10)
tennis_data.play = tennis_data.play.replace('yes', 11)

tennis_data

X = np.array(pd.DataFrame(tennis_data, columns = ['outlook', 'temp','humidity','windy']))
y = np.array(pd.DataFrame(tennis_data, columns = ['play']))
#tennis_data의 각 column값들을 추출하여 X와 y에 저장
print(pd.DataFrame(tennis_data, columns = ['outlook', 'temp','humidity','windy']))
#pandas의 dataframe에 맞춰 변환시키고 np.array의 배열 형태로 변환 후 저장
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y) #train set과 test set을 분리 7.5:2.5

gnb_clf=GaussianNB() #가우시안 나이브베이즈 모듈을 변수에 저장
gnb_clf=gnb_clf.fit(X_train,y_train) #모델에 학습시켜 다시 저장

gnb_prediction =gnb_clf.predict(X_test) #X_test값을 테스트하여 예측값 저장

print(X_test)
print(gnb_prediction)

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
#classfication_report 주요분류 측정 항목을 보여주는 보고서 모듈
#confusion_matrix 분류의 정확성을 평가하기 위한 오차행렬 계산 모듈
#f1_score F-measure을 계산하는 모듈
#accuracy_score 정확도를 수치로 계산

print('Confusion Matrix')
print(confusion_matrix(y_test,gnb_prediction))
#       예측값  yes     no
#실제값
#  yes           1       1
#   no           1       1

print('Classification Report')
print(classification_report(y_test,gnb_prediction))

fmeasure = round(f1_score(y_test,gnb_prediction, average='weighted'),2)
#실제 값과 예측값을 입력 후 평균을 weighted로 설정 (weighted는 클래스별로 가중치를 적용하겠다는 뜻) 
#f1_score를 계산하고 round함수를 사용해 소수점 아래 2번째 자리까지 표현한 값을 저장
accuracy = round(accuracy_score(y_test,gnb_prediction, normalize=True),2)
#실제 값과 예측값을 입력 후 normalize를 True로 설정(True는 정확도를 계산해서 출력해주는 역할)
#accuracy_score 계산하고 round함수를 사용해 소수점 아래 2번째 자리까지 표현한 값을 저장

df_nbclf = pd.DataFrame(columns=["classifier","F-measure","Accuracy"]) #각각의 열을 만들어 pandas의 dataframe에 생성 후 저장
df_nbclf.loc[len(df_nbclf)] = ["Naive Bayes",fmeasure, accuracy] #각 값들을 열에 맞게 저장
df_nbclf
