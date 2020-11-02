# Behavioral Cloning Project
## Objective
The challenge in this project was to train a network in a way that it is able to steer a vehicle in a way, that it follows the path. The input for the network is an image from the vehicle's front camera. The output is a steering angle.

## Training data
To train the network, I used the data provided by the project. The main reason is, that I wanted to start with what is there, instead of immedately taking new data. Also, I find that the data collection task can be difficult, depending on what one wants: 
The best data would show the network how to always steer back to the middle and align in the center of the road. However, to get that data, you also have to maneuver the car away from the center. Including these maneuvers into the training data would be faulty. So before training the network, one would have to filter the data.
Another way is to drive perfect laps and hope that the network replicates exactly that behavior. For this use case this approach is okay, but in real world scenarios you always have disturbances and this training approach would not give you a robust result. I don't think it would be able to handle strong disturbances, like noise strong noise in the image data.

## Data preprocessing
to generalize a bit, I took every image in the data set, copied it, flipped it around its vertical axis, multiplied the expected steering angle with -1 and added those to the training set.

## Network architecture

I used the CNN described in Nvidias [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) paper, because it tackles a comparable, though even more challenging problem. The CNN's basic structure is:
* Normalization layer 
* 3 convolutional layer with 5x5 kernels and 2x2 strides
* 2 convolutional layer with 3x3 kernels and 1x1 strides
* 3 fully connected layers

The input is an image and the output is a steering angle.

### Why I didn't handle overfitting
Firstly, because also the paper did not include regularization, dropout or any other measures to prevent overfitting. Secondly, because the projects use case is one where a bit of overfitting might be desirable. As mentioned above the aim is that the net replicates driving behavior on the path. So actually it might even be a good thing here, if the model overfits, since it is supposed to replicate what is given in the training data.

## Result
After looking at different trainings, I saw that the network hardly improves after roundabout five epochs. So I let it learn for five epochs, using MSE as a loss function and _Adam_ as the optimization algorithm. Surprisingly, already after my first training, the vehicle made it through the whole round. When I tried training a second time, the network did not perform as well, so I stuck to the first solution.