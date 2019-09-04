from keras.datasets import imdb

(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words =10000)
#훈련 데이터에서 가장 자주 나타나는 단어 10000개만 사용


print(train_data[0]) #train_data[0]문장 중 각 단어들의 해당되는 번호를 나열( 총 10000개의 feature)

print(train_labels[0]) #긍정

word_index = imdb.get_word_index() #단어와 정수 인덱스를 매핑한 딕셔너리
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()]) #정수 인덱스와 단어를 매핑하도록 뒤집음
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])#리뷰를 디코딩함 0,1,2는 '패딩','문서시작','사전에없음'을 위한 인덱스로 3을 뺌
#https://blog.naver.com/shwotjd14/221531924188
#https://blog.naver.com/yisu0407/221508385142


print(decoded_review)

#신경망에는 리스트를 입력 값으로 둘 수 없음 -> 리스트를 텐서로 바꾸는 작업을 해야함
#패딩하는 방식과 원-핫 인코딩 방식이 있음

#원-핫 인코딩 방식
#리스트를 0과 1 벡터로 변환함 
# [3,5] => [0, 0, 1, 0, 1, 0, 0, .... ,0] (10000차원벡터)

import numpy as np
print(len(train_data)) #25000개의 문장

def vectorize_sequences(sequences, dimension=10000):
    results=np.zeros((len(sequences),dimension))
    
    for i, sequence in enumerate(sequences): #각 문장마다 나타나는 문자번호 i번째 문장의 sequence 문자번호를 인덱스로 하여 1로 변경
        results[i,sequence]=1.
    return results

x_train = vectorize_sequences(train_data)# 25000개의 문장데이터를 10000개의 많이 등장하는 단어 필드에 매핑
print(x_train.shape)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32') #asarray 는 참조본 생성으로 원본이 바뀌면 참조본도 동시에 데이터가 변경됨
y_test = np.asarray(test_labels).astype('float32')  #array 는 복사본 생성으로 원본이 바뀌어도 바뀌지 않음
print(y_train.shape)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])


x_val = x_train[:10000] #검증 데이터
partial_x_train= x_train[10000:]
y_val = y_train[:10000]
partial_y_train= y_train[10000:]


history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data=(x_val,y_val)) #training set 내에서 검증함

import matplotlib.pyplot as plt
%matplotlib inline
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() #그래프 초기화
acc = history_dict['acc']
val_acc=history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

model= models.Sequential()
model.add(layers.Dense(16, activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test) #test set을 가지고 예측 정도를 봄

results

model.predict(x_test)
