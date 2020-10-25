import csv
import cv2
import numpy as np

lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
print("size of lines is: ", len(lines))
print("steering is: ", lines[1][3])

lines.pop(0)
        
images = []
measurements = []
for line in lines:
    print('/opt/carnd_p3/data/' + line[0])
    image = cv2.imread('/opt/carnd_p3/data/' + line[0])
    measurement = float(line[3])
    
    images.append(image)
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
    