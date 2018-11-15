import tensorflow as tf
from tensorflow import keras as ks
import pickle
import numpy as np
import matplotlib.pyplot as plt
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
cifar10_meta = unpickle("cifar10\\batches.meta")
cifar10_classes = cifar10_meta[b'label_names']

json_file = open('networkStructure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
network = ks.models.model_from_json(loaded_model_json)

network.load_weights("model.h5")
print("Loaded model from disk")
image2Load = 'cat6.jpg'
img =ks.preprocessing.image.load_img(image2Load, target_size=(32,32))
#plt.imshow(img)
#plt.show()
img = ks.preprocessing.image.img_to_array(img)
arrayofImg = np.expand_dims(img, axis=0)
arrayofImg /= 255
results = network.predict(arrayofImg)
result = results[0]
classIdx = np.argmax(result)
chance = result[classIdx]
class_label = cifar10_classes[classIdx]

print("This is an image of a", class_label, "With a chance of", chance, '\n')
img =ks.preprocessing.image.load_img(image2Load)
plt.imshow(img)
plt.show()
