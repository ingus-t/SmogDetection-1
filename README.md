# Smog Detection Project

A project by **Planet-Earth** (#sg_planetearth) study group in Facebook Secure and Private AI Scholarship Challenge 2019.  
**_Check out our web-app we have deployed at https://smog4000.onrender.com/. At this site, you can upload your street/highway images and test the accuracy of our classification system._** Although it works on all images, you can get the best results on highway images.
___
## Part 1: Project summary

### Description:
Smog Detection project has been created as a collaborative team effort between 
Facebook Secure and Private AI Scholars on #sg_planetearth. The main focus of the group is to provide Deep Learning solutions to solve most concerning real-world problems, such as Climate Change, Food Security, Plastic Pollution, Deep Fake, Oceanic Pollution, Industrial Nuclear Waste, Clean Energy and more(using AI for social good). Currently, our team is focusing on Smog and Fog detection on highways.</br>

### Why is it important? Why should we care?
The sudden appearance of smog and/or fog on the highway more often than not causes serious and sometimes fatal accidents. Smog is the main agent for severe air pollution. It can aggravate health problems including problems with breathing and sleeping, as well as it can inversely damage plants and forest cover.</br> 

### Solution
Monitoring, early smog detection, and preventive action. By using traffic cameras and train a model to recognize the smog/fog patterns, we can automate the alert and send a notification promptly. When smog/fog appears, the system notifies drivers who are within specified range about an upcoming “obstacle”. </br>

In any practical situation, other components need to be taken into consideration as well. For example, the presence of flying birds and any type of material that will block the camera view. To identify the range of vision sensors need to be added to the solution.</br>

### Our approach
There are plenty of existing solutions and advancements in computer vision. Our approach is to use Machine Learning Image Detection and train a CNN model. Smog Detection is a smog classification project, where images are arranged into two main categories: "Smog/Fog/NotClearlyVisible Highways" and "Clearly Visible Highways".</br>

### Project objective
Smog is a byproduct of the global climate change scenario. As we move further into the industrialized age, Smog continues to pollute our air, reduce visibility on roads and is a leading cause of accidents on roads.  
**The primary goal of the project is to avoid and help reduce the rate of accidents in self-driving vehicles. That can be achieved by using this classifier model as one of the key components attached to traffic cameras**. This way vehicles can automatically be alerted about smog on the roads, streets or highways, no matter if it's heavy, medium or low traffic to adjust its dynamics like speed, steering rotation, lanes, etc. We can use the prediction output labels from the model to alert vehicles and drivers. 

### Dataset
Since inception, people in the group have taken their task most seriously. They contributed towards building a unique dataset which now has more than 4,000 images divided evenly between the two groups mentioned above. Our collection consists of:</br>
![DataSet1](https://user-images.githubusercontent.com/7014697/63082087-25652d80-befa-11e9-9ccd-f49b5476010f.JPG)
![DataSet2](https://user-images.githubusercontent.com/7014697/63082090-272ef100-befa-11e9-8d7a-2296e22aa7c3.JPG)

### Rules and regulations for selecting pictures
- Images need to have a view of highways or streets;
- Avoid a large number of people;
- Avoid traffic accidents;
- Avoid night pictures;
- Avoid bird's-eye view;
- Avoid images that are copyrighted or have a watermark;
- Acceptable image formats: jpg, jpeg, png, jfif, webp.  

Notes:
- Synthetic images we used were taken from [SFSU synthetic dataset](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)
- To detect and remove duplicates (regardless of format and resolution), we were using [DupeGuru](https://dupeguru.voltaicideas.net/). It is an open-source tool for duplicate image detection that works across different platforms (Linux, OS X, Windows).

### Contributors:

No | Name | Slack Handle 
--- | --- | ---
1 | Shudipto Trafder | @Shudipto Trafder
2 | Berenice Terwey | @Berenice
3 | Agata Gruza | @Agata [OR, USA]
4 | Ingus Terbets | @Ingus Terbets
5 | Akash Antony | @Akash Antony
6 | Alexander Villasoto | @Alexander Villasoto
7 | Pooja Vinod | @Pooja Vinod
8 | Ramkrishna Acharya | @Viper
9 | Sourav Kumar | @sourav kumar
10 | George Christopoulos | @George Christopoulos
11 | Sayed Maheen Basheer | @Sayed Maheen Basheer
12 | Abhishek Lalwani | @Abhishek Lalwani
13 | Laura Truncellito | @LauraT

## Screenshots from the User Interface:

**'Smog' Detection**  
![](screenshots/Capture.JPG)

**'Clear' Detection**  
![](screenshots/Capture1.JPG)
___
# Part 2: Implementation with PyTorch

This is a descriptive summary of the Model Architecture we have implemented, and the Data Augmentations we have applied.  
[Find the notebook for this implementation here](https://www.kaggle.com/berenice18/smogdetection).

**Input to model:**  
Images captured by traffic cameras.  
**Output from model :**  
Prediction as label '0'(clear view) or '1'(smog detected).

### Data Augmentations and Transformations

* [transforms.RandomRotation(30),](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomRotation)
* [transforms.RandomHorizontalFlip(),](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip)
* [transforms.Resize(256),](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Resize)
* [transforms.ColorJitter(0.1),](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ColorJitter)
* [transforms.CenterCrop(224),](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#CenterCrop)
* [transforms.ToTensor(),](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ToTensor)
* [transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])](https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Normalize)
        
### Model name 
*[model.pt](https://www.kaggleusercontent.com/kf/18699045/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NV6L2pV_K0u5eS5JKcovWg.dESzIjhRmqdZvcrYJfWc0PEm1naJnTT7yU1E7IgRn7awT7yClc4NMtb5YCbWr_IlYpz5ZdbEs2sM9aKKpu2W3hgkilV2jo3AYjxI3lTCKe4nt25kA5dvh0lq4qw8cPDiYKz6ROFWKnnl9yd_Fw12dQ.Mb5NRyZispSEz3dY5gs7dA/model.pt)

### Description of the model 
Five modules contain deeper sublayers.
Let's go through them one by one:

* CNN layers : 7 
* Linear layers : 2
* Pooling layers : 4
* Batch normalization layers : 7

**conv1 :**
* CNN layers : Conv2d(3, 32, 3, padding=1), Conv2d(32, 32, 3, stride=2, padding=1)
* Pooling layer : MaxPool2d(2, 2)
* Batch normalization layer : BatchNorm2d(32), nn.BatchNorm2d(32),

**conv2 :**
* CNN layers : Conv2d(32, 64, 3, padding=1), Conv2d(64, 64, 3, stride=2, padding=1)
* Pooling layer : MaxPool2d(2, 2)
* Batch normalization layer : BatchNorm2d(32), nn.BatchNorm2d(64),

**conv3 :**
* CNN layers : Conv2d(64, 128, 3, padding=1), Conv2d(128, 128, 3, stride=2, padding=1)
* Pooling layer : MaxPool2d(2, 2)
* Batch normalization layer : BatchNorm2d(128), BatchNorm2d(128)

**conv4 :**
* CNN layers : Conv2d(128, 256, 3, padding=1)
* Pooling layer : MaxPool2d(2, 2)
* Batch normalization layer : BatchNorm2d(256)

**fc :**
* Linear layers : Linear(256, 128), Linear(128, 2)

**Dropout has been applied in conv4 and linear layer where probability = 0.35 in conv4 and 0.5 in fc.**

### Activation functions:
[Mish](https://github.com/digantamisra98/Mish) activation has been used.  
[Mila](https://github.com/digantamisra98/Mila) activation has been used.

### Optimizer : 
[Adam](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html) optimizer is used here.

### Loss function : 
[CrossEntropyLoss](https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html) is used.

### Learning rate (with lr scheduler) :
* lr schedular = StepLR(optimizer, step_size=10, gamma=0.5)
* lr = 0.001

### Epochs : 
epoch : 50

## RESULTS

![](screenshots/PT_train_test_loss_plot.png)

### Test loss :
Test Loss: 36.734721 

### Test Accuracy :
Accuracy: 99.0000

### Class wise loss :
* Test Accuracy of     0: 98% (389/393)
* Test Accuracy of     1: 98% (378/382)

### Overall testing accuracy : 
Test Accuracy (Overall): 98% (767/775)

We decided to create a user-friendly web app so everyone can test and experiment with our project. We hope that this will be useful for any road-alerts related project that may require such a smog detection/classification facility. 

___
# Part 3: Implementation with KERAS:

### Description
* Libraries used: keras 2.2.4, numpy, matplotlib
* Trained: on google Colab
* Dataset Used: [Smog4000](https://www.kaggle.com/berenice18/smog4000)
* [Find the notebook for this implementation here](https://github.com/q-viper/SmogDetection/blob/master/Smog4000/keras_Smog4000.ipynb). 

### About dataset:
[Smog4000](https://www.kaggle.com/berenice18/smog4000) dataset was created by our team. 

### Preprocessing of data
* Rescale range: 1./255
* shear and zoom: 0.2
* color: RGB
* Outshape: 224 X 224

### About the model
Custom CNN model is used here. Complete summary of model architecture is given below:
<img src = 'https://github.com/q-viper/SmogDetection/blob/master/Smog4000/Assets/model.JPG'>
<br/>
The Sequential model has 3 blocks of CNN layers and one final linear layer. In each CNN block we have:
* Convolution layers of same out filters(32, 32, 64, 64, 128, 128)
* Filter shape: (3, 3)
* Activation Function: ReLU(Rectified Linear Unit)
* MaxPooling: pool size(3, 3)
* BatchNormalization
* Dropout: 0.25

The Final block is for linear layers. It has:
* Flatten
* Dense of out 256 and ReLU
* BatchNormalization
* Dropout of 0.5
* Classification layer with sigmoid

Total parameters: 1,469,346

### Model Compilation
* Optimizer: Adam(Adaptive Momentum) Optimizer is used on this model.
* Learning rate: default(0.001)
* Loss function: Categorical_crossentropy (categories: clear, smog)

### Training
* Epochs: 10
* Batch Size: 32

### Performance
* Train Accuracy: 0.9968
* Validation Accuracy: 0.9921
* Test Accuracy: 0.985
* Train time: 750 seconds per epoch

### Future Implementations:
* Using fewer parameters
* Test with different optimizer and loss functions

## Application of Secure & Private AI in this project
The images collected in this project are carefully chosen to avoid any copyright issues.  For images that contain intellectual property issues or private images, we can consider incorporating federated learning and secure prediction into our classification models. This will allow for training in a private cloud while minimizing the risk of leaking intellectual property or private training data. 
