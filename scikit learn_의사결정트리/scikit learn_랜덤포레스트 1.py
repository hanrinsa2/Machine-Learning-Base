from sklearn.datasets import load_iris #datasets 모듈 내 load_iris 함수를 불러옴
from sklearn.metrics import accuracy_score #metrics 모듈 내 성능평가 함수를 불러옴
import numpy as np
import pandas as pd

iris = load_iris()

x_train = iris.data[:-30] #데이터 시작 부터 끝에서 30번째 하나 전까지 
y_train = iris.target[:-30]

x_test = iris.data[-30:] #끝에서 30번째 부터 끝까지
y_test = iris.target[-30:]


print(y_train)
print()
print(y_test)

#분리가 합리적으로 이루어지지 않은 결과를 볼 수 있음

from sklearn.ensemble import RandomForestClassifier
#random forest분류기 생성을 위한 모듈

rfc = RandomForestClassifier(n_estimators=10)
rfc
#n_estimators : Decision tree의 개수 
#max_features : 최대 고려하는 feature의 개수 (기본 자동)
#oob_score : oob사용여부 (기본 사용 X)

rfc.fit(x_train,y_train)
#train data를 random forest 모델에 학습시킴

prediction = rfc.predict(x_test)
#test 데이터를 넣어 예측값을 저장

print(prediction == y_test)
#예측값과 실제값을 비교하여 나열


rfc.score(x_test, y_test)
#rfc의 정확도를 측정해줌(Accuracy)

#성능평가 2
from sklearn.metrics import accuracy_score #분류결과의 accuracy를 계산해주는 함수
from sklearn.metrics import classification_report #분류결과의 종합적인 성능평가를 계산해줌

print("Accuracy is : ", accuracy_score(prediction, y_test))
print("=======================================================")
print(classification_report(prediction,y_test))

from sklearn.model_selection import train_test_split
x = iris.data
y = iris.target

X_train,X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2) 
#데이터 무작위 혼합 8:2로 train test set 분할
print(y_test)
print(Y_test)

clf = RandomForestClassifier(n_estimators=10)#트리 10개
clf.fit(X_train,Y_train)
prediction_1 = clf.predict(X_test)


print("Accuracy is : ", accuracy_score(prediction, y_test))
print("=======================================================")
print(classification_report(prediction,y_test))

clf_2 = RandomForestClassifier(n_estimators=200,
                              max_features=4,
                              oob_score=True) 
clf_2.fit(X_train, Y_train)
prediction_2 = clf_2.predict(X_test)
print(prediction_2 == Y_test)
print("Accuracy is : ",accuracy_score(prediction_2, Y_test))
print("========================================")
print(classification_report(prediction_2,Y_test))
