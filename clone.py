import csv
import cv2
import numpy as np

lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)
        
images = []
measurements = []
for line in lines:
    image = cv2.imread('/opt/carnd_p3/data/' + line[0])
    measurement = float(line[3])
    
    images.append(image)
    measurements.append(measurement)
    images.append(np.fliplr(image))
    measurements.append(-1.0*measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, Lambda, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x : (x / 255) - 0.5))

model.add(Conv2D(24, kernel_size=5, strides=2, activation='relu'))
model.add(Conv2D(36, kernel_size=5, strides=2, activation='relu'))
model.add(Conv2D(48, kernel_size=5, strides=2, activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
print("success")
    