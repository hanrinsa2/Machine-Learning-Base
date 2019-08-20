from sklearn import datasets #scikit learn에서 제공하는 data set을 사용하기 위한 모듈

from sklearn.tree import DecisionTreeClassifier #decision tree 기계학습 모델을 위한 모듈

from sklearn.model_selection import train_test_split #데이터셋을 train set과 test set으로 분리하기위한 모듈
from sklearn.model_selection import StratifiedKFold #분리한 데이터셋을 stratified K fold cross validation 을 사용하기 위한 모듈
from sklearn.model_selection import cross_val_score #cross validation결과의 정확도를 측정하기 위한 모듈

from sklearn.metrics import confusion_matrix #분석 결과의 confusion_matrix를 보여주기 위한 모듈
from sklearn.metrics import accuracy_score #분석결과의 accuracy 를 측정하기 위한 모듈
from sklearn.metrics import classification_report #분석결과의 recall, precision, f-measure 을 측정하기 위한 모듈
from sklearn.metrics import roc_auc_score #ROC curve(TPR , FPR) 곡선 아래 면적 AUC를 구하기 위한 모듈(1에 가까울수록좋음)
from sklearn.metrics import mean_squared_error #분석 결과의 MSE를 구하기 위한 모듈(평균제곱오차)

data = datasets.load_breast_cancer() #dataset에 있는 유방암 데이터를 불러오는 함수를 이용해 data에 저장
X = data.data #속성 데이터를 X에 저장( 특징들 )
y = data.target #클래스 데이터를 y에 저장 (악성, 양성)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 
# train_test_split 함수로 train set과 test set중 test set을 20퍼로 설정

clf = DecisionTreeClassifier() #결정트리분류 모델을 clf에 할당
clf.fit(X_train, y_train) #train set을 fit함수로 clf에 할당된 모델로 훈련
clf

y_pred = clf.predict(X_test) #predict 함수를 사용해 test의 X값으로 이전에 만들어진 모델에 넣어 예측 값을 추론하고 y_pred에 저장

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred)) # test의 실제 y값과 방금 예측한 y_pred값을 비교해 confustion_matrix 를 보여줌
#                       예측값 
#                 class=0    class=1
#실제값  class=0     47         5
#        class=1     2          60

print('Accuracy')
print(accuracy_score(y_test,y_pred,normalize=True)) # False일시 건수  /  True 일 시 비율
# y_test와 y_pred 값을 비교해 정확도 출력 [ (a+d) / (a+b+c+d) ]

#                       예측값 
#                 class=0    class=1
#실제값  class=0     a           b
#        class=1     c           d

print('Classification Report')
print(classification_report(y_test, y_pred))
# precision     [ a / (a+c) ]
# recall       [ a / (a+b) ]
# F-measure   [ 2a / (2a+b+c) ]  <<= [ 2rp / (r+p) ]
# ==f1-score
# support -> 데이터건수

#                        예측값 
#                  class=0    class=1
# 실제값  class=0     a           b
#         class=1     c           d

print('AUC')
print(roc_auc_score(y_test, y_pred)) 
# roc 곡선 아래 면적 구하기 (1에 가까울수록 좋음)
# roc(TPR 과 FPR 의 그래프)
# TPR=recall=[ a/(a+b) ]
# FPR=[ c/(c+d) ]

#                        예측값 
#                  class=0    class=1
# 실제값  class=0     a           b
#         class=1     c           d

print('Mean Squared Error')
print(mean_squared_error(y_test, y_pred))
# MSE를 출력 (평균 제곱 오차 -> 실제값의 회귀선과 모델 예측값 사이의 오차(residual)을 사용)

skf = StratifiedKFold(n_splits=10) #10개의 fold cross validation 모듈을 skf에 생성
skf.get_n_splits(X,y) #data set을 10개의 fold(train, test set)로 구성하도록 적용
print(skf)

for train_index, test_index in skf.split(X,y): #데이터 셋 구성을 봐봄
    print('Train set : ', train_index)
    print('Test set  : ', test_index) # 대략 50개 씩 나누어 바뀌는걸 볼수 있음

    clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=skf) 
# X,y 데이터셋을 10개 fold한 skf 모듈로 만들어진 train set과 test set을 clf모듈을 통해 각 10개의 셋에 대한 트리들의 정확도를 출력
print('K Fold Cross validation Score')
print(scores)
print("Average accuracy") #10개 트리에 대한 정확도의 평균
print(scores.mean())

skf_sh = StratifiedKFold(n_splits=10, shuffle=True) #10개의 fold cross validation 모듈을 skf에 생성 
# (단 현재는 데이터를 랜덤으로 섞음)
skf_sh.get_n_splits(X,y) #data set을 10개의 fold(train, test set)로 구성하도록 적용
print(skf_sh)

for train_index, test_index in skf_sh.split(X,y): #데이터 셋 구성을 봐봄
    print('Train set : ', train_index)
    print('Test set  : ', test_index) # 대략 50개 씩 나누어 바뀌는걸 볼수 있음

    clf = DecisionTreeClassifier()
scores_sh = cross_val_score(clf, X, y, cv=skf_sh) 
# X,y 데이터셋을 10개 fold한 skf 모듈로 만들어진 train set과 test set을 clf모듈을 통해 각 10개의 셋에 대한 트리들의 정확도를 출력
print('K Fold Cross validation Score')
print(scores_sh)
print("Average accuracy") #10개 트리에 대한 정확도의 평균
print(scores_sh.mean())
