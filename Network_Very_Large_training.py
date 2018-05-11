
# coding: utf-8

# # Exemplo de rede neural MLP e Convolucional com Keras

# ## Importando as bibliotecas

# In[1]:


import keras
import csv


# In[2]:


#creating train labels list and images list: train_labels and train_images
with open('sign_mnist_train.csv','r') as dataset_train:
    reader=csv.reader(dataset_train)
    next(dataset_train) #jump the first line of the csv
    train=list(reader)
    print("Number of train exemples:",len(train),"\n")
    a=len(train)
    print("Size of one exemple, with label:",len(train[1]),"\n")
    b=len(train[1])
    train_labels=[]
    train_images=[]
    for i in range(0,a):
        train_labels.append(train[i][0])
    print("train[3]:",train[3],"\n")
    print("Label of exemple 3:",train_labels[3],"\n")
    for i in range(0,a):
        train_images.append(train[i][1:b])
    print("Exemple 3 Image:",train_images[3])


# In[3]:


#creating test labels list and images list: test_labels and test_images
with open('sign_mnist_test.csv','r') as dataset_test:
    reader=csv.reader(dataset_test)
    next(dataset_test)
    test=list(reader)
    print("Number of train exemples:",len(test),"\n")
    a=len(test)
    print("Size of one exemple, with label:",len(test[1]),"\n")
    b=len(test[1])
    test_labels=[]
    test_images=[]
    for i in range(0,a):
        test_labels.append(test[i][0])
    print("test[3]:",test[3],"\n")
    print("Label of test exemple 3:",test_labels[3],"\n")
    for i in range(0,a):
        test_images.append(test[i][1:b])
    print("Test exemple 3 Image:",test_images[3])


# ## Analisando os dados

# In[4]:


import matplotlib.pyplot as plt
import math
import numpy as np


# In[5]:


dim=(b-1)
dim=math.sqrt(dim)
print("images are:", dim,"x",dim)


# In[6]:


print(len(train_labels))
print(len(test_labels))


# In[7]:


print(np.shape(train_images))
print(np.shape(test_images))


# In[8]:


#convert data to np array and float
train_img=np.array(train_images)
train_img=train_img.astype(np.float)

test_img=np.array(test_images)
test_img=test_img.astype(np.float)

train_l=np.array(train_labels)
train_l=train_l.astype(np.float)

test_l=np.array(test_labels)
test_l=test_l.astype(np.float)


# In[9]:


#reshapes data to list of 2D images
train_img=train_img.reshape((27455,28,28))
test_img=test_img.reshape((7172,28,28))


# In[10]:


train_img[3]


# In[11]:


test_img[3]


# In[12]:


plt.imshow(train_img[3], cmap='gray')


# In[13]:


plt.imshow(test_img[3], cmap='gray')


# In[14]:


print(train_l[3])
print(test_l[3])
#a=0,b=1,c=2,d=3,e=4,f=5,g=6,h=7,i=8,j=9,k=10,l=11,m=12,n=13,o=14,p=15,q=15,r=17,t=18,u=19,v=20,w=21,x=22,y=23


# ## Normalizando os dados

# In[15]:


train_img = train_img.astype('float32') / 255
test_img = test_img.astype('float32') / 255


# In[16]:


train_img[3]


# ### Entrada da rede convolucional

# In[17]:


from keras import backend as K


# In[18]:


img_rows = 28
img_cols = 28

if K.image_data_format() == 'channels_first':
    train_img = train_img.reshape(train_img.shape[0], 1, img_rows, img_cols)
    test_img = test_img.reshape(test_img.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_img = train_img.reshape(train_img.shape[0], img_rows, img_cols, 1)
    test_img = test_img.reshape(test_img.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# In[19]:


print('input shape:', input_shape)


# In[20]:


np.shape(test_img)


# ### Transformando rótulos em dados categóricos

# In[21]:


from keras.utils import to_categorical

train_l = to_categorical(train_l)
test_l = to_categorical(test_l)


# In[22]:


train_l[3]


# In[23]:


np.shape(train_l)


# ## Definindo a rede

# model: https://keras.io/models/model/
# 
# layers: https://keras.io/layers/about-keras-layers/

# In[24]:


from keras import models
from keras import layers


# ### Rede Convolucional

# In[51]:


network = models.Sequential()
network.add(layers.Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
network.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Dropout(0.25))                                                  
network.add(layers.Flatten())
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dropout(0.5))                                                
network.add(layers.Dense(25, activation='softmax'))


# In[52]:


network.summary()


# In[53]:


#allows two gpu usage (2 GTX1080)
network = keras.utils.multi_gpu_model(network,gpus=2)


# ## Compilando a rede

# optimizers: https://keras.io/optimizers/
# 
# loss funcitons: https://keras.io/losses/
# 
# metrics: https://keras.io/metrics/

# In[54]:


sgd=keras.optimizers.SGD(momentum=0.8)
network.compile(optimizer=sgd, 
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])


# ## Treinando

# In[55]:


earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=20)
history = network.fit(train_img, train_l,
                          batch_size=128,
                          epochs=500,
                          validation_split=0.2, #0.3?
                          callbacks=[earlystop])


# ## Avaliando a rede Convolucional

# In[56]:


test_loss, test_acc = network.evaluate(test_img, test_l)


# In[57]:


print('test accuracy:', test_acc)
#max:93.5%, dropout maior


# In[58]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'b--', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# ## Saving Network

# In[63]:


import h5py


# In[64]:


#create simple model, not parallel
network2 = models.Sequential()
network2.add(layers.Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
network2.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
network2.add(layers.MaxPooling2D(pool_size=(2, 2)))
network2.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network2.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
network2.add(layers.MaxPooling2D(pool_size=(2, 2)))
network2.add(layers.Dropout(0.25))                                                  #mudar para 0.25
network2.add(layers.Flatten())
network2.add(layers.Dense(256, activation='relu'))
network2.add(layers.Dense(256, activation='relu'))
network2.add(layers.Dropout(0.5))                                                  #mudar para 0.5
network2.add(layers.Dense(25, activation='softmax'))

network2.summary()


# In[65]:


#compiling new model
#sgd=keras.optimizers.SGD(momentum=0.8)
network2.compile(optimizer='sgd',
                 loss=keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])


# In[66]:


#copy the trained parameters
network2.set_weights(network.get_weights()) 


# In[67]:


#save the network
network2.save('large_network_2.h5')


# In[68]:


#test saved network
test_loss2, test_acc2 = network2.evaluate(test_img, test_l)
print('test accuracy:', test_acc2)

