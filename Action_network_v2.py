
# coding: utf-8

# In[1]:


import keras
import csv

#creating train labels list and images list: train_labels and train_images
with open('sign_mnist_train.csv','r') as dataset_train:
    reader=csv.reader(dataset_train)
    next(dataset_train) #jump the first line of the csv
    train=list(reader)
    #print("Number of train exemples:",len(train),"\n")
    a=len(train)
    #print("Size of one exemple, with label:",len(train[1]),"\n")
    b=len(train[1])
    train_labels=[]
    train_images=[]
    for i in range(0,a):
        train_labels.append(train[i][0])
    #print("train[3]:",train[3],"\n")
    #print("Label of exemple 3:",train_labels[3],"\n")
    for i in range(0,a):
        train_images.append(train[i][1:b])
    #print("Exemple 3 Image:",train_images[3])

#creating test labels list and images list: test_labels and test_images
with open('sign_mnist_test.csv','r') as dataset_test:
    reader=csv.reader(dataset_test)
    next(dataset_test)
    test=list(reader)
    #print("Number of train exemples:",len(test),"\n")
    a=len(test)
    #print("Size of one exemple, with label:",len(test[1]),"\n")
    b=len(test[1])
    test_labels=[]
    test_images=[]
    for i in range(0,a):
        test_labels.append(test[i][0])
    #print("test[3]:",test[3],"\n")
    #print("Label of test exemple 3:",test_labels[3],"\n")
    for i in range(0,a):
        test_images.append(test[i][1:b])
    #print("Test exemple 3 Image:",test_images[3])


# In[2]:


import random
i=random.randint(0,7171)


# In[3]:


#convert data to np array and float
import numpy as np

test_img=np.array(test_images[i])
test_img=test_img.astype(np.float)



test_l=np.array(test_labels[i])
test_l=test_l.astype(np.float)
print(test_l)




#train_img=np.array(train_images)
#train_img=train_img.astype(np.float)

#test_img=np.array(test_images)
#test_img=test_img.astype(np.float)

#train_l=np.array(train_labels)
#train_l=train_l.astype(np.float)

#test_l=np.array(test_labels)
#test_l=test_l.astype(np.float)


# In[4]:


#reshapes data to list of 2D images
#train_img=train_img.reshape((28,28))
test_img=test_img.reshape((28,28))


# In[5]:


image=test_img


# In[6]:


image


# In[7]:


import matplotlib.pyplot as plt
print(plt.imshow(image, cmap='gray'))


# In[8]:


## Normalizando os dados

#train_img = train_img.astype('float32') / 255
test_img = test_img.astype('float32') / 255


# In[9]:


### Entrada da rede convolucional

from keras import backend as K

img_rows = 28
img_cols = 28

if K.image_data_format() == 'channels_first':
    #train_img = train_img.reshape(train_img.shape[0], 1, img_rows, img_cols)
    test_img = test_img.reshape(1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    #train_img = train_img.reshape(train_img.shape[0], img_rows, img_cols, 1)
    test_img = test_img.reshape(img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('input shape:', input_shape)

np.shape(test_img)


# In[10]:


#atualizando entrada para entrada normalizada
image_norm=test_img


# In[11]:


#image_norm


# ## Importando rede

# In[12]:


import h5py


# In[13]:


network=keras.models.load_model('large_network_2.h5')


# In[14]:


network.summary()


# In[15]:


#test_loss, test_acc = network.evaluate(test_img, test_l)    #check do modelo


# In[16]:


#print('test accuracy:', test_acc)


# ## Execution

# In[17]:


import tensorflow as tf
result=network.predict(np.array([image_norm,]))


# In[18]:


print(result)


# In[19]:


result=np.round(result,1)


# In[20]:


print(result)


# In[21]:


#from keras.utils import to_categorical

#train_l = to_categorical(train_l)
print(type(test_l))
test_l=test_l.astype('int')
test_l.item(0)


# In[22]:


for j in range(25):
    if result.item(j)==0:
        continue
    else:
        result_number=j
        break
list_of_classes=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
result_class=list_of_classes[result_number]       


# In[23]:


print("\n\n")
print("test exemple selected at random:", i)
print("label:", test_l)
list_of_classes=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
print("label class:",list_of_classes[test_l.item(0)])
print("a=0,b=1,c=2,d=3,e=4,f=5,g=6,h=7,i=8,j=9,k=10,l=11,m=12,n=13,o=14,p=15,q=16,r=17,s=18,t=19,u=20,v=21,w=22,x=23,y=24")


# In[24]:


print("predicted label:", result_number)
print("predicted class:", result_class)

