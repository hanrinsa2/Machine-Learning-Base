from sklearn.metrics import classification_report, confusion_matrix
#classfication_report 주요분류 측정 항목을 보여주는 보고서 모듈
#confusion_matrix 분류의 정확성을 평가하기 위한 오차행렬 계산 모듈

from sklearn.model_selection import train_test_split
#train set과 test set을 분할해주는 모듈

from sklearn.tree import DecisionTreeClassifier
#의사결정트리 모듈을 불러옴

from sklearn import tree
#분류 및 회귀를 위한 의사결정 트리 기반 모델이 있는 서브 패키지

from IPython.display import Image
#IPython내 정보를 보여주는 도구용 공용 API
#Image : raw 데이터가 있는 PNG JPEG 이미지 객체를 만드는 모듈

import pandas as pd
import numpy as np

import pydotplus
#그래프를 생성하는 graphviz의 Dot언어를 파이썬 인터페이스에 제공하는 모듈

import os
#운영체제와 상호작용하기위한 기본적 기능이 제공(경로생성,변경)

tennis_data = pd.read_csv('dataset/playtennis.csv') #pandas read_csv함수를 사용해 playtennis.csv 파일을 불러옴
tennis_data

#데이터 내 문자열을 int형으로 대치하여 전처리
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

X = np.array(pd.DataFrame(tennis_data, columns=['outlook','temp','humidity','windy']))
#tennis_data의 컬럼 값들을 데이터 프레임 형태로 추출하여 np.array를 사용해 배열형태로 저장
y=  np.array(pd.DataFrame(tennis_data, columns=['play']))
#tennis_data의 class 값을 데이터 프레임 형태로 추출하여 np.array를 사용해 배열형태로 저장

X_train, X_test, y_train, y_test = train_test_split(X,y) # data set을 train과 test set 으로 분리
#일반적으로 7.5:2.5

dt_clf = DecisionTreeClassifier()# 의사결정트리 분류 모델을 호출 후 저장
dt_clf = dt_clf.fit(X_train,y_train)# train set을 의사결정트리 분류 모델에 훈련시킴
dt_prediction = dt_clf.predict(X_test)#test set의 X값에 대한 예측값을 저장

print(confusion_matrix(y_test,dt_prediction)) 
#오차 행렬을 print문으로 출력

print(classification_report(y_test, dt_prediction)) 
#실제값과 예측값 사이의 성능평가

os.environ["PATH"]+=os.pathsep+'C:/Program Files (x86)/Graphviz2.38/bin'
#Ipython 내 그래프 생성할 수 있는 인터페이스 경로 추가 설정
#https://graphviz.gitlab.io/_pages/Download/Download_windows.html

feature_names = tennis_data.columns.tolist() 
#tennis_data의 각 컬럼을 list형태로 변수 feature_names에 저장
#트리표현함수 (tree.export_graphviz())의 각 파라미터(feature_names)에 넣기위함

feature_names = feature_names[0:4]
#저장된 feature_names를 슬라이싱(0:4)하여 outlook, temp,humidity,windy 의 컬럼만을 추출하여 다시 저장

target_name = np.array(['Play No', 'Play Yes'])
#list형태로 변수 target_name에 저장
#트리표현함수 (tree.export_graphviz())의 각 파라미터(class_names)에 넣기위함

dt_dot_data = tree.export_graphviz(dt_clf, out_file = None, feature_names = feature_names,
                                   class_names = target_name,filled = True, rounded = True,
                                  special_characters = True)
#tree 패키지 중 의사결정 트리를 dot형식으로 내보내는 함수인 export_graphviz()를 이용해 트리 표현을 변수 dt_dot_data에 저장
#dt_clf : 의사결정트리 분류기(위에서 선언)
#out_file : 의사결정트리를 파일 또는 문자열로 반환(기본 : tree.dot   None : 문자열)
#feature_names : 각 features의 이름
#class_names : 각 대상 class의 이름을 오름차순으로 정렬 (True일 경우 class 이름의 symbol 표현)
#filled : True일 경우 분류를 위한 다수 클래스 , 회귀 값의 극한 또는 다중 출력의 노드 순도를 나타내기 위해 노트를 색칠
#rounded : True 일 경우 둥근 모서리 노드상자를 그리고 , times-roman 글꼴대신 helvetica 글꼴 사용
#special_characters : true일 경우 특수 문자 표시

dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
#pydotplus 모듈중 dot 형식의 데이터로 정의된 그래프를 로드하는 함수인 graph_from_dot_data()변수에 dt_dot_data를 입력후 저장

Image(dt_graph.create_png())
#변수 dt_graph에 대한 정보를 png파일로 생성 후 Image모듈을 통해 그래프 표현
#gini 계수가 1일수록 불순도가 높음
#가장 중요한 요소로 outlook이 뽑혔음 (sunny)
#https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2-3-5-%EA%B2%B0%EC%A0%95-%ED%8A%B8%EB%A6%AC/
