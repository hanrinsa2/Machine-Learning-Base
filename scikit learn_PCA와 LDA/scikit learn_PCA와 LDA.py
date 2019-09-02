import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA #PCA를 실행시키기 위한 패키지
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA를 실행 시키기 위한 패키지
%matplotlib inline

data = datasets.load_breast_cancer() #데이터 셋 로드
data.feature_names #데이터셋의 속성종류 확인

x = data.data[:,:2] #2번째 속성들만 저장
y = data.target #class 데이터 저장
target_names = data.target_names #악성과 양성 정보를 저장
print(x)
print(target_names)

plt.figure(figsize=(10,10))
colors = ['red','blue']

for color, i, target_name in zip(colors, [0,1], target_names): #zip을 사용하면 각배열을 인덱스마다 묶어 리스트로 나타내줌. 
    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target_name) 
    #x인덱스0 즉 속성의 첫번째는 가로x축, x인덱스1 즉 속성의 두번째는 가로 y축, 각 클래스 i 에 대해 색상을 달리함
    
plt.legend()
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.show()


x = data.data
y = data.target
target_names = data.target_names #위와 동일한 방법이지만 x 를 전부 포함

pca = PCA(n_components=2) #주성분 2개를 추출하기 위함
x_p = pca.fit(x).transform(x) #훈련 후 차원 축소 진행
print('가장 큰 주성분 두개에 대한 분산 : %s' %str(pca.explained_variance_ratio_))

plt.figure(figsize=(10,10))
colors = ['red','blue']
print(x_p)
for color, i, target_name in zip(colors, [0,1], target_names): #zip을 사용하면 각배열을 인덱스마다 묶어 리스트로 나타내줌. 
    plt.scatter(x_p[y==i, 0], x_p[y==i, 1], color=color, label=target_name) 
    #x인덱스0 즉 속성의 첫번째는 가로x축, x인덱스1 즉 속성의 두번째는 가로 y축, 각 클래스 i 에 대해 색상을 달리함
    
plt.legend()
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

x = data.data
y = data.target
target_names = data.target_names #위와 동일한 방법이지만 x 를 전부 포함

lda = LinearDiscriminantAnalysis(solver='eigen', n_components=2) #고유값을 사용해 클래스를 구분하는 벡터를 구하기 위해 eigen으로 설정
x_l = lda.fit(x,y).transform(x) #속성 데이터와 클래스 데이터y를 훈련시키고 차원축소 진행

plt.figure(figsize=(10,10))
colors = ['red','blue']
print(x_l)
for color, i, target_name in zip(colors, [0,1], target_names): #zip을 사용하면 각배열을 인덱스마다 묶어 리스트로 나타내줌. 
    plt.scatter(x_l[y==i, 0], x_l[y==i, 1], color=color, label=target_name) 
    #x인덱스0 즉 속성의 첫번째는 가로x축, x인덱스1 즉 속성의 두번째는 가로 y축, 각 클래스 i 에 대해 색상을 달리함
    
plt.legend()
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.show()

#결론 LDA의 모델이 현재 데이터에 대해 1차원으로 인식해버리기에 점을 찍을 수 없음.
