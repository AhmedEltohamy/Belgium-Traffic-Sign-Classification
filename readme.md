# Belgium Traffic Sign Classification 

## About

<p>Apply different classification techniques using machine learning and deep learning models on belgium traffic sign for classification dataset</p>


***

## Dataset:

Belgium Traffic Sign for Classification **[Dataset](https://people.ee.ethz.ch/~timofter/traffic_signs/)**

### Number of classes and their labels

Dataset contain 62 different classes labelled as [0:61]

### Dataset Images Numbers and size

Dataset contain 7095 images but with different size

### Training, Validation and Testing

Dataset already split as 4575 images for training data and 2520 images for testing data 


***

## Non-Deep Models

### Extracted Features 

In preprocessing step we resize all image to 64x64 px and use grayscale raw pixel values as features so there are 4096 features to use in the model

***

### Logistic Regression (LR)

Hyper-parameters

<ul>
  <li>the cost function is equal to the sum of - Y times the national log of prediction</li>
  <li>optimizer is GradientDescentOptimizer with learning rate equal to 0.01</li>
  <li>number of epochs equal to 1000 and batch size equal to 900</li>
  <li>number of batches depending on training data equal to 5 </li>
</ul>

#### LR Results

the LR model labeled 82.58% of the testing images correctly

confusion matrix: 

<img alt="LR CM" src="./ReadMe/screenshots/LR CM.jpg">

***


### Support Vector Machine (SVM)

Hyper-parameters

<ul>
  <li>the cost function is multiclass support vector machine loss with delta equal to 1</li>
  <li>add regularization of type L2 regularization to cost function with alpha equal to 0.0001</li>
  <li>optimizer is GradientDescentOptimizer with learning rate equal to 0.01</li>
  <li>number of epochs equal to 1000 and batch size equal to 900</li>
  <li>number of batches depending on training data equal to 5</li>
</ul>

#### SVM Results

the SVM model labeled 85.04% of the testing images correctly

confusion matrix:

<img alt="SVM CM" src="./ReadMe/screenshots/SVM CM.jpg">

***

## Deep Models

### Preprocessing Step

resize all images to 64x64 px and use images with 3 channels and convert labels from class vector to binary class matrix

***

### Convolutional Neural Network (CNN)

### CNN Number of Layers

Number of layers in CNN model equal to 12

### CNN Model Architecture

<img alt="CNN Arc" src="./ReadMe/screenshots/CNN Arc.png">

Hyper-parameters

<ul>
  <li>in compile stage stochastic gradient descent otpimizer is used with learning rate equal to 0.01 and loss function equal categorical crossentropy</li>
  <li>in fiting stage number of epochs set to 10 and batch size equal to 64 </li>
</ul>

#### CNN Results and Loss Screen Shots

the CNN model labeled 95.2% of the testing images correctly

loss curve:

<img alt="Model Loss CNN" src="./ReadMe/screenshots/Model Loss CNN.jpg">

confusion matrix: 

<img alt="CNN CM" src="./ReadMe/screenshots/CNN CM.jpg">

***


###	Pre-Trained Model

### Chosen Pre-trained Model

I used VGG16 model with 4 additional layers

### Pre-Trained Model Architecture

<img alt="VGG Arc1" src="./ReadMe/screenshots/VGG Arc1.jpg">

<img alt="VGG Arc2" src="./ReadMe/screenshots/VGG Arc2.jpg">

Hyper-parameters

<ul>
  <li>in compile stage rmsprop otpimizer is used with learning rate equal to 0.0001 and loss function equal categorical crossentropy</li>
  <li>in fiting stage number of epochs set to 10 and batch size equal to 64</li>
</ul>

### Tuning Layers

I only unfrozen the last layer in VGG16 model

#### Pre-trained Results and Loss Screen Shots

the pre-trained model labeled 87.1% of the testing images correctly

loss curve:

<img alt="Model Loss VGG" src="./ReadMe/screenshots/Model Loss VGG.jpg">


confusion matrix: 

<img alt="VGG CM" src="./ReadMe/screenshots/VGG CM.jpg">


***

## ⚡ Technologies and libraries
* Python
* TensorFlow
* Keras
* Matplotlib
* seaborn
* pandas
* NumPy
* scikit-image

***

## Tools
* Anaconda
* Jupyter
* Google Colab


