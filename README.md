# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

#### Model Architecture

I use 4 convoluted layers with 64, 96, and 128 filters respectively, and each of which is followed by a maximum pooling with keep probability of 0.8. 

|               | input size    | kernel size | filters | keep probability |
| ------------  |:-------------:|:-----------:|:-------:|:----------------:|
|    conv2D     | 160 x 320 x 1 |    5 x 5    |    64   |      0.8         |
|    conv2D     | 5 x 5 x 128   |    3 x 3    |    96   |      0.8         |
|    conv2D     | 3 x 3 x 128   |    3 x 3    |   128   |      0.8         |
|  full connect | flatten       |             |         |      1           |
|  full connect | 128           |             |         |      0.7         |
|  full connect | 64            |             |         |      0.7         |
|  full connect | 1             |             |         |      1           |


