from keras.datasets import reuters

(train_data,train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000) # training set과 test set으로 나눔
#가장 자주 등장하는 단어 1만개

import numpy as np

def vectorize_sequences(sequences, dimension=10000): #이거도 원핫 인코딩에 해당
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data) # x_train 에 해당하는 부분을 원핫인코딩
x_test = vectorize_sequences(test_data) #x_test 에 해당하는 부분을 원핫 인코딩

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)# 원핫 인코딩으로 만들어줌 y_train 해당
one_hot_test_labels = to_categorical(test_labels) # y_test 해당

from keras import models
from keras import layers

model = models.Sequential()# 모델 층 생성
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))#총 10000개의 특징이 들어 갈 예정(데이터의 갯수가 아님)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
              #두 확률 분포 사이의 거리를 측정 (네트워크가 출력한 확률 분포와 진짜 레이블의 분포 사이 거리)
             metrics=['accuracy'])

x_val = x_train[:1000] #검증을 위해 데이터셋을 검증용 데이터와 훈련용 데이터로 나눔
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, #훈련용 데이터로 모델에 적용
                   partial_y_train,
                   epochs=20, #20회 반복
                   batch_size=512, #한번 들어가는 데이터 셋 크기가 512개씩
                   validation_data=(x_val,y_val)) #검증은 검증용 데이터를 사용하여 함

import matplotlib.pyplot as plt
%matplotlib inline
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss') #훈련용 loss 값
plt.plot(epochs, val_loss, 'b', label='Validation loss') #검증용 loss 값

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# 9epochs 쯤에서 과대적합이 진행되는 것을 볼 수 있음

model = models.Sequential()# 모델 층 생성
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))#총 10000개의 특징이 들어 갈 예정(데이터의 갯수가 아님)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
              #두 확률 분포 사이의 거리를 측정 (네트워크가 출력한 확률 분포와 진짜 레이블의 분포 사이 거리)
             metrics=['accuracy'])

model.fit(partial_x_train, #훈련용 데이터로 모델에 적용
        partial_y_train,
        epochs=9, #9회 반복 , 20회는 너무 과함
        batch_size=512, #한번 들어가는 데이터 셋 크기가 512개씩
        validation_data=(x_val,y_val)) #검증은 검증용 데이터를 사용하여 함

results = model.evaluate(x_test, one_hot_test_labels)

results

import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels)==np.array(test_labels_copy)
print(hits_array)
float(np.sum(hits_array)) / len(test_labels)#섞어서 그대로인 부분의 확률

predictions = model.predict(x_test)

print(np.argmax(predictions[0]))
print(predictions.shape)# 2246개의 뉴스기사에 대한 46개(데이터갯수)의 토픽(클래스)에 대한 확률 분포
print(np.sum(predictions[0]))
print(np.sum(predictions[1]))
print(predictions[0])

model = models.Sequential()# 모델 층 생성
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))#총 10000개의 특징이 들어 갈 예정(데이터의 갯수가 아님)
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
              #두 확률 분포 사이의 거리를 측정 (네트워크가 출력한 확률 분포와 진짜 레이블의 분포 사이 거리)
             metrics=['accuracy'])

model.fit(partial_x_train, #훈련용 데이터로 모델에 적용
        partial_y_train,
        epochs=20, #9회 반복 , 20회는 너무 과함
        batch_size=512, #한번 들어가는 데이터 셋 크기가 512개씩
        validation_data=(x_val,y_val)) #검증은 검증용 데이터를 사용하여 함

results = model.evaluate(x_test, one_hot_test_labels)


results

#은닉층에서 출력층의 노드가 48인데 은닉층을 4로 하여 손실이 많이 일어남 이로써 정확도가 떨어지는걸 볼 수 있음
#즉 은닉층의 노드 수는 출력층보다 많은 것이 기본적이론( 너무적어도 안된다는 말)
#병목현상이 일어나지 않게 하기위함
