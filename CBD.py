from google.colab import drive
drive.mount('/content/drive')

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.utils.vis_utils import plot_model
from glob import glob
from keras.models import Model
from keras.layers import Flatten,Dense,Dropout,Softmax
from tensorflow.keras.optimizers import Adam

Image_size = [224,224]
valid_path = "/content/drive/MyDrive/cbddataset/dataset/Images/Test"
train_path = "/content/drive/MyDrive/cbddataset/dataset/Images/Train"

train_path

resnet = ResNet50(include_top=False,input_shape=Image_size+[3],weights='imagenet')
plot_model(resnet)
resnet.summary()

for layer in resnet.layers:
  layer.trainable = False
  
folders = glob("/content/drive/MyDrive/cbddataset/dataset/Images/Train/*")
folders

x=Flatten()(resnet.output)

prediction=Dense(len(folders),activation='softmax')(x)

model = Model(inputs = resnet.input , outputs = prediction)

plot_model(model)
model.summary()

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("/content/drive/MyDrive/cbddataset/dataset/Images/Train",
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                 class_mode='categorical')


test_set = train_datagen.flow_from_directory("/content/drive/MyDrive/cbddataset/dataset/Images/Test",
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode='categorical')
                                             
          
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),validation_steps=len(test_set)
)

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

from tensorflow.keras.models import load_model
model.save('model_resnet50.h5')

y_pred = model.predict(test_set)

y_predimport numpy as np
y_pred = np.argmax(y_pred, axis=1)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('model_resnet50.h5')

img=image.load_img("/content/drive/MyDrive/cbddataset/dataset/Images/Train/audi/15.jpg",target_size=(224,224))

x=image.img_to_array(img)

x=np.expand_dims(x,axis=0)
x =x /25
preds = model.predict(x)

preds=np.argmax(preds, axis=1)

if preds==1:
  preds="The Car IS a"
elif preds==2:
  preds="The Car IS Mercedes"
elif preds==3:
  preds="The Car IS Lambhorgini"
elif preds==3:
  preds="The Car IS Hyundai"
elif preds==5:
  preds="The Car IS Skoda"
else:
  preds="The Car Is Audi"
print(preds)
