from sklearn.datasets import load_iris #아이리스 데이터 셋 로드 모듈

iris = load_iris() #아이리스 데이터셋 저장

iris.keys() #아이리스 데이터셋의 키값 확인

iris['data'][0:10] #아이리스 데이터 셋의 'data'키값의 0~9번째를 슬라이싱하여 나타냄

X = iris['data'] #X값에 150X4 크기 특징 데이터를 저장
y = iris['target']#Y값에 클래스 데이터 저장 0, 1, 2로 표시

from sklearn.model_selection import train_test_split #X와y로 나눈 데이터를 학습할 training set과 test set으로 나누기위한 모듈

X_train, X_test, y_train, y_test = train_test_split(X, y) #set 분할, 7.5:2.5 로 분할
X_train

from sklearn.preprocessing import StandardScaler #정규화를 시켜주는 함수를 불러오는 모듈

scaler = StandardScaler() #정규화 생성자를 저장 , 모든 데이터를 평균0 표준편차1 로 좁혀줌

scaler.fit(X_train) #X_train 의 데이터 값으로 기준을 세워 정규화를 학습함.

#위의 X_train 기준으로 정규화 된 scaler함수로 모든 데이터를 정규화
X_train = scaler.transform(X_train) #정규화하여 재할당
X_test = scaler.transform(X_test) #정규화 하여 재할당
X_train

from sklearn.neural_network import MLPClassifier #인공신경망 분류알고리즘 중 MLPClassifier(다중인공신경망=다층신경망) 모듈을 불러옴

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10)) #히든레이어를 3계층(10,10,10)으로 할당

mlp.fit(X_train,y_train)
#training set을 mlp에 학습시킴 
#기본적인 활성화함수는 relu임
#relu 함수 : 음수일 경우 0 양수일경우 선형적 데이터

#activation : 활성화 함수
#alpha : 신경망 내 정규화 파라미터
#batch_size : 최적화를 시키기 위한 학습 최소 크기 (메모리 내 모든 데이터가 들어갈 수없으니 들어갈 데이터 양을 정해줌)
#총 데이터 100개  batch_size가 10 이면
#1 iteration = 10 개 데이터에 대해 학습
#1 Epoch = 10 batch_size = 10 iteration
#1 epoch마다 10개의 데이터가 10번으로 나뉘어 들어가 100개의 총 데이터를 순전파 역전파함

#epsilon : 수치 안정성을 위한 오차 값
#learning_rate_init : 가중치 업데이트 할때의 크기
#max_iter : 최대 반복 횟수
#hidden_layer_sizes : 히든레이어 크기
#shuffle : 데이터 학습 시 데이터 위치를 임의적으로 변경하는지의 여부 (섞는지)
#solver : 가중치 최적화를 위해 사용하는 함수(역전파)
#validation_fraction : training데이터 학습시 validation의 비율
#validation : training데이터 학습시 데이터가 유의미 한지 검증하는 데이터
#epoch : https://blog.naver.com/qbxlvnf11/221449297033 순방향 + 역방향 패스 시 1 epoch (underfitting, overfitting을 조절)

predictions = mlp.predict(X_test) #mlp로 학습된 모델로 X_test에 대해 예측하여 predictions 변수에 저장

from sklearn.metrics import classification_report, confusion_matrix
#confusion matrix : 실제값과 예측값 간을 비교해 행렬로 나타냄

print(confusion_matrix(y_test,predictions))
#           예측값 setosa(0)      versicolor(1)   virginica(2)
# 
#   실제값
# setosa(0)           9              0               0
# versicolor(1)       0              12              2
#virginica(2)         0              0               15

print(classification_report(y_test,predictions))
#정확률 재현율 fesure-score 갯수
