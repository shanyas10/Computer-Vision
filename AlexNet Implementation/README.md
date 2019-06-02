# AlexNet


## Introduction

Alex Krizhevsky, Geoffrey Hinton and Ilya Sutskever created a neural network architecture called ‘AlexNet’ and won Image Classification Challenge (ILSVRC) in 2012. They trained their network on 1.2 million high-resolution images into **1000 different classes** with 60 million parameters and 650,000 neurons. 

## Architecture

It contains **5 convolutional layers** and **3 fully connected layers**. Relu is applied after very convolutional and fully connected layer. **Dropout** is applied before the first and the second fully connected year. **The image size in the following architecutre chart should be 227 * 227 (instead of 224 * 224, pointed out by Andrei Karpathy in his famous CS231n Course)**

### Convolutional Layers
#### Layer 1:
The input for AlexNet is a **227x227x3 RGB image** which passes through the first convolutional layer with **96 feature maps** or filters having **size 11×11** and a **stride of 4**. 
Then the AlexNet applies **maximum pooling layer** or sub-sampling layer with a **filter size 3×3** and a **stride of two**.

#### Layer 2:
Next, there is a second convolutional layer with **256 feature maps** having size 5×5 and a stride of 1. Then there is again a **maximum pooling layer** with **filter size 3×3** and a **stride of 2**. 

#### Layer 3,4,5:
The third, fourth and fifth layers are convolutional layers with **filter size 3×3** and a **stride of one**. The first **two used 384 feature maps** where the **third used 256 filters**.
The three convolutional layers are followed by a **maximum pooling layer** with **filter size 3×3**, a **stride of 2** and have **256 feature maps**.

### Fully Connected Layers
#### Layer 6, 7, 8:
Fully connected layers with **4096 units**
