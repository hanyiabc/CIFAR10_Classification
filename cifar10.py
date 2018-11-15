import tensorflow as tf
from tensorflow import keras as ks
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batchSize = 300
numIter = 150

#(x_train, y_train), (x_test, y_test) = ks.datasets.cifar10.load_data()

cifar10_meta = unpickle("cifar10\\batches.meta")
cifar10_classes = cifar10_meta[b'label_names']

data=np.empty((1,1024*3), dtype='float32')
labels = np.empty((1), dtype='int')
for i in range(1,6):
    batch = unpickle("cifar10\\data_batch_" + str(i))
    data = np.append( data, batch[b'data'].astype('float32'), axis=0 )
    labels = np.append(labels, batch[b'labels'])

data = np.delete(data, 0, axis=0)
labels = np.delete(labels, 0, axis=0)

testBatch = unpickle("cifar10\\test_batch")
testData = testBatch[b'data'].astype('float32')

data /=255
testData/=255

data = data.reshape((len(data), 3, 32, 32)).transpose(0, 2, 3, 1)
testData = testData.reshape((len(testData), 3, 32, 32)).transpose(0, 2, 3, 1)

labels = ks.utils.to_categorical(labels, 10)
testLabels = ks.utils.to_categorical(batch[b'labels'], 10)

network = ks.Sequential()
network.add(ks.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
network.add(ks.layers.Conv2D(64, (3,3), activation='relu'))
network.add(ks.layers.MaxPooling2D(pool_size=(2,2)))
network.add(ks.layers.Dropout(0.15))

network.add(ks.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
network.add(ks.layers.Conv2D(128, (3,3), activation='relu'))
network.add(ks.layers.MaxPooling2D(pool_size=(2,2)))
network.add(ks.layers.Dropout(0.15))

network.add(ks.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
network.add(ks.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
network.add(ks.layers.Conv2D(256, (3,3), activation='relu'))
network.add(ks.layers.MaxPooling2D(pool_size=(2,2)))
network.add(ks.layers.Dropout(0.15))

#network.add(ks.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
#network.add(ks.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
#network.add(ks.layers.Conv2D(512, (3,3), activation='relu'))
#network.add(ks.layers.MaxPooling2D(pool_size=(2,2)))
#network.add(ks.layers.Dropout(0.15))

#network.add(ks.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
#network.add(ks.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
#network.add(ks.layers.Conv2D(512, (3,3), activation='relu'))
#network.add(ks.layers.MaxPooling2D(pool_size=(2,2)))
#network.add(ks.layers.Dropout(0.15))

network.add(ks.layers.Flatten())

network.add(ks.layers.Dense(2048, activation='relu'))
network.add(ks.layers.Dropout(0.35))
network.add(ks.layers.Dense(2048, activation='relu'))
network.add(ks.layers.Dropout(0.35))
network.add(ks.layers.Dense(10, activation='softmax'))
network.summary()

network.compile(
    loss='categorical_crossentropy', 
    optimizer = 'adam', 
    metrics=['accuracy']
    )


network.fit(
    data, labels, 
    batch_size = batchSize, 
    epochs = numIter, 
    validation_data=(testData, testLabels)
    )

model_json = network.to_json()
with open("networkStructure.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
network.save_weights("model.h5")
print("Saved model to disk")