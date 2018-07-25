
# coding: utf-8

# In[2]:


import numpy as np
import os
import pandas as pd

from PIL import Image

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from random import shuffle

from tensorboard.backend.event_processing import event_accumulator

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import collections

base_dir = "D:/Projects/CrazyDataScience/Metal-o-meter/"
os.chdir(base_dir)

tblog_dir = os.path.join(base_dir,"tflogs")
if not os.path.isdir(tblog_dir):
    os.makedirs(tblog_dir)


# In[3]:


convnet = input_data(shape=[None, 128, 128, 1], name='input')

convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='elu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 2, activation='softmax') # Set number to number of unique genres
convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

model = tflearn.DNN(convnet, tensorboard_dir=tblog_dir,tensorboard_verbose=0)


# In[8]:


# Load our pre-trained model (0.97 accuracy on training data)
model.load(base_dir+"/model/metalyzer_model.tflearn")


# In[5]:


def getImageData(filename,imageSize):
    img = Image.open(filename)
    imgData = getProcessedData(img, imageSize)
    return imgData

def getProcessedData(img,imageSize):
    img = img.resize((imageSize,imageSize), resample=Image.ANTIALIAS)
    imgData = np.asarray(img, dtype=np.uint8).reshape(imageSize,imageSize,1)
    imgData = imgData/255.
    return imgData


# In[7]:


# Load and prepare the training data
genres = os.listdir("D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/Slices/Train/")
genres = [filename for filename in genres if os.path.isdir("D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/Slices/Train/"+filename)]

data = []

for genre in genres:
        print("-> Adding {}...".format(genre))
        #Get slices in genre subfolder
        filenames = os.listdir("D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/Slices/Train/"+genre)
        filenames = [filename for filename in filenames if filename.endswith('.png')]
        
        #Randomize file selection for this genre
        shuffle(filenames)

        #Add data (X,y)
        for filename in filenames:
            imgData = getImageData("D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/Slices/Train/"+genre+"/"+filename, 128)
            label = [1. if genre == g else 0. for g in genres]
            data.append((imgData,label))
            
#Shuffle data
shuffle(data)

#Extract X and y
X,y = zip(*data)

validationNb = int(len(X)*0.3) # use 30% of the data for validation
trainNb = len(X)-(validationNb)

print("Total amount of spectrograms: " + str(len(X)))
print("Size of the training dataset: " + str(trainNb))
print("Size of the validation dataset: " + str(validationNb))

train_X = np.array(X[:trainNb]).reshape([-1, 128, 128, 1])
train_y = np.array(y[:trainNb])
validation_X = np.array(X[trainNb:trainNb+validationNb]).reshape([-1, 128, 128, 1])
validation_y = np.array(y[trainNb:trainNb+validationNb])


# In[ ]:


# Train the model
model.fit(train_X, train_y, n_epoch=10, batch_size=32, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True)


# In[9]:


# Let's take a look how the training went, copy/paste the location of the tfevents file here
ea = event_accumulator.EventAccumulator("D:/Projects/CrazyDataScience/Metal-o-meter/tflogs/A5BP9V/events.out.tfevents.1531163553.DESKTOP-DQS30QC",
  size_guidance={ 
  event_accumulator.SCALARS: 0
})

ea.Reload()

hist = {
  'Accuracy' : [x.value for x in ea.Scalars('Accuracy')],
  'Validation Accuracy' : [x.value for x in 
                           ea.Scalars('Accuracy/Validation')],
  'Loss' : [x.value for x in ea.Scalars('Loss')],
  'Validation Loss' : [x.value for x in ea.Scalars('Loss/Validation')]
}

fig = plt.figure()
keys = ['Accuracy', 'Loss', 'Validation Accuracy', 'Validation Loss']
for i,thing in enumerate(keys):
  trace = hist[thing]
  plt.subplot(2,2,i+1)
  plt.plot(range(len(trace)),trace)
  plt.title(thing)

fig.set_tight_layout(True)
fig


# In[ ]:


# (optional) save the model
# model.save(base_dir+"/model/metalyzer_model.tflearn")


# In[14]:


# Let's get some test data and throw it at the model
data_test = []

filenames = os.listdir("D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/Slices/Test")
filenames = [filename for filename in filenames if filename.endswith('.png')]        

for filename in filenames:
    imgData = getImageData("D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/Slices/Test/"+filename, 128) 
    label = 0
    data_test.append((imgData, label))
    print(filename)
            
#Extract X and y
X,y = zip(*data_test)

testNb = len(X)

print("Size of the test dataset: " + str(testNb))
test_X = np.array(X[:testNb]).reshape([-1, 128, 128, 1])


# In[15]:


# Now that we have some test data in, let's use it to predict
# We will get a predicted label pack for each slice in the form of
# most likely label to the most unlikely label
predict = model.predict(test_X)


# In[16]:


# Let's return the result of the prediction, each slice has a predicted class ranging from 0 to 3 (4 classes in total)
# based on the folder order we used when importing the training data
# in our case:
# 0 = Non-metal
# 1 = Metal
predict_readable = np.argmax(predict, axis=1)
predict_readable


# In[17]:


# Lets do a count on the number of times a genre was classified for the entire song on a per slice basis
collections.Counter(predict_readable)


# In[18]:


# Now we can do some calculations to add the predictions per slice and classify how metal a song is!
pred_nonmetal = int(collections.Counter(predict_readable)[0])
pred_metal = int(collections.Counter(predict_readable)[1])

total_classifications = len(predict_readable)
metal_percentage = int(pred_metal/total_classifications*100)
nonmetal_percentage = 100-metal_percentage

# How metal is this song? 
print("Metal'o Meter Score")
print("-------------------")
if metal_percentage < 10:
   print("This song scored "+ str(metal_percentage) + "/100 on the Metal 'o Meter - 😴")
   print("A tree is more metal than this...")
elif metal_percentage > 10 and metal_percentage < 50:
   print("This song scored "+ str(metal_percentage) + "/100 on the Metal 'o Meter - 🤨")
   print("Is this Nothing Else Matters?")
elif metal_percentage > 50 and metal_percentage < 80:
   print("This song scored "+ str(metal_percentage) + "/100 on the Metal 'o Meter - 🤘")
   print("Do I see a mosh pit forming?")
elif metal_percentage > 80 and metal_percentage < 90:
   print("This song scored "+ str(metal_percentage) + "/100 on the Metal 'o Meter - 🤘🤘")
   print("Yes! Bang your head! Things are gonna get rough!")
elif metal_percentage > 90:
   print("This song scored "+ str(metal_percentage) + "/100 on the Metal 'o Meter - 🤘🤘")
   print("O my f**** god! This is the f****** loudest, nastiest, metalest metal evah!")


# In[19]:


# Let's look at some graphs for some more insight!
plt.gcf().clear()

labels = 'Metal', 'Non-metal'
sizes = [metal_percentage, nonmetal_percentage]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[20]:


# Can we see which parts of the song were the most metal?

# Convert the numpy array to a dataframe
# This shows us the percentage the model generated whether the slice was 0 - NonMetal or 1 - Metal
# Keep in mind each slice is ~ 2,57 sec
predict_percentage = pd.DataFrame(predict)
predict_percentage['counter'] = range(len(predict_percentage))
predict_percentage.columns = ['Not Metal', 'Metal', 'SliceNr']
predict_percentage['Slice Start Time'] = predict_percentage['SliceNr']*2.57
predict_percentage = predict_percentage[['SliceNr','Slice Start Time', 'Not Metal', 'Metal']]
predict_percentage


# In[21]:


# And let's plot the percentages
plt.gcf().clear()

plt.style.use('seaborn-darkgrid')
my_dpi=96
plt.figure(figsize=(800/my_dpi, 480/my_dpi), dpi=my_dpi)

plt.plot('Slice Start Time', 'Metal', data=predict_percentage, marker='', markerfacecolor='', markersize=10, color='skyblue', linewidth=4, alpha=0.8)
plt.plot('Slice Start Time', 'Not Metal', data=predict_percentage, marker='', color='grey', linewidth=1, alpha=0.4)
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.xlabel("Time (sec)")
plt.ylabel("Metal percentage")


plt.show()

