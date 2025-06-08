COMPARISON TABLE



SUMMARY REPORT

VGG models

It is a typical deep Convolutional Neural Network (CNN) design with numerous layers, and the abbreviation VGG stands for Visual Geometry Group. The term “deep” describes the number of layers, with VGG-16 or VGG-19 having 16 or 19 convolutional layers, respectively.


VGG16 model 

VGG16 is a type of CNN that is considered to be one of the best computer vision models to date. The creators of this model evaluated the networks and increased the depth using an architecture with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations. They pushed the depth to 16–19 weight layers making it approx — 138 trainable parameters.


An accuracy of 92% was achieved by the model after training on the CIFAR-10 dataset. 


VGG19 model

The VGG19 model has the same basic idea as the VGG16 model, with the exception that it supports 19 layers. The numbers “16” and “19” refer to the model’s weight layers. In comparison to VGG16, VGG19 contains three extra convolutional layers.


An accuracy of 75.39% was achieved by the model after training on the CIFAR-10 dataset 


CNN Models 

A Convolutional Neural Network (CNN), also known as ConvNet, is a specialized type of deep learning algorithm mainly designed for tasks that necessitate object recognition, including image classification, detection, and segmentation.

We have used CNN models with different layers – 8, 12, 14 and 19. It was expected that the accuracy would increase on adding more layers but that turned out to be false. At first the accuracy increased from 70.20% in 8-layer to 86.84% in 12-layer since more features are getting extracted. But then it declines to 67.29% in 14-layer and 10% in 19-layer. Due to overfitting, additional stacked layers slow network speed. Specifically the model has reduced training error but larger testing error. As the network depth rises, the accuracy saturates and declines quickly.

ResNet models

ResNet (Residual Network) models are deep convolutional neural networks that leverage residual connections to enable training of very deep networks without vanishing gradients. 

ResNet-50
Depth: 50 layers
Architecture: Uses bottleneck structures (3-layer blocks) for efficiency.
Performance: Achieves good accuracy with a balance between depth and computational cost.
ResNet-152 
Depth: 152 layers
Architecture: Also uses bottleneck structures for efficiency.
Performance: Provides higher accuracy, but at the cost of increased computational complexity.
ResNet-50 is generally faster for inference due to its lower complexity.
ResNet-152 will have a larger model size due to the additional parameters.
We achieved an accuracy of 94% on ResNet-50 and 95% on ResNet-152 after training them on the CIFAR-10 dataset.


CIFAR-10 Classification Report using Fine-Tuned AlexNet
1. Model and Training Setup
Model Used: Pre-trained AlexNet from torchvision.models
Classifier Modification: Final layers replaced to output 10 classes
Input Size: 224×224 pixels
Loss Function: CrossEntropyLoss
Optimizer: Adam (learning rate = 1e-4)
Regularization: Dropout layers added to the classifier
Gradient Clipping: Max norm of 1.0
Learning Rate Adjustment: Reduced by 10% after epoch 5
Epochs: 10
Batch Size: 32
Augmentations: Horizontal flip applied during training
2. Performance Metrics
2.1 Accuracy
After training for 10 epochs, the model achieved the following performance on the test dataset:
Test Accuracy: 91.10%
2.2 Classification Report
Below is the detailed classification report showing precision, recall, and F1-score for each class:
(Insert your actual classification\_report output here)
2.3 Confusion Matrix
The confusion matrix summarizes the model's predictions for each class:

2.4 ROC Curve and AUC
Using a one-vs-rest approach, ROC curves were plotted for each class using the model's predicted probabilities (softmax outputs). The average AUC was also computed.
average AUC: 0.9956

3. Training and Validation Trends
Below is the plot showing the trend of training and validation losses over 10 epochs:




CIFAR-10 Classification Report using ANN (9 hidden layers)
1. Dataset
Dataset Used: CIFAR-10
Number of Classes: 10
Image Dimensions: 32 × 32 × 3
Train/Validation Split: 80% training, 20% validation (from original training set)
Test Set Size: 10,000 images
2. Data Preprocessing and Augmentation
Training Transforms: Random horizontal flip, random cropping with padding, normalization
Test/Validation Transforms: Normalization only
Normalization Parameters: Mean = (0.5, 0.5, 0.5), Std = (0.5, 0.5, 0.5)
3. Model Architecture
A fully connected deep ANN was implemented with the following characteristics:
Input Size: 3 × 32 × 32 = 3072
Hidden Layers: 9 hidden layers with sizes:
2500 → 2000 → 2000 → 1000 → 1000 → 500 → 500 → 250 → 100
Activation: ReLU
Batch Normalization: Applied after each layer
Dropout: Applied progressively (0.2 in early layers, 0.1 in later layers)
Output Layer: Linear layer with 10 outputs (one per class)
4. Training Details
Loss Function: CrossEntropyLoss
Optimizer: Adam (learning rate = 0.001)
Learning Rate Scheduler: StepLR (step\_size = 10, gamma = 0.5)
Epochs: 30
Batch Size: 64
5. Performance Metrics
5.1 Training Progress
The model was trained for 30 epochs. Training loss, validation loss, and training accuracy were tracked.
(Insert training vs validation loss plot here)
(Insert training accuracy plot here)
5.2 Final Evaluation on Test Set
Evaluation Metric: Accuracy, Precision, Recall, F1-Score

Confusion Matrix:

5.3 ROC Curve and AUC (One-vs-Rest)
Multi-class ROC analysis was performed using the model's softmax output.
Micro-average AUC: (Insert AUC score here)

6. Observations
Batch normalization and dropout improved stability and generalization.
Training accuracy increased steadily, with validation decreased over epochs.
Comparatively lower accuracy.

















Report: Artificial Neural Network (ANN) for CIFAR-10 Classification (4 hidden layers)
Data Preparation
Dataset: CIFAR-10 (60,000 images: 50,000 training, 10,000 test)
Data Splitting:
Training set further split into 80% training (40,000 images) and 20% validation (10,000 images).
Data Augmentation (Training only):
Random horizontal flip
Random cropping with padding of 4 pixels
Normalization:
All images normalized to mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5) per channel to scale pixel values to roughly [-1, 1].
Model Architecture
The Improved ANN consists of:
Input layer: Flattened image vector (3 channels × 32 height × 32 width = 3072 features)
Hidden layers: Four fully connected (Linear) layers with sizes 2500, 2000, 1000, and 500 neurons respectively.
Activation: ReLU non-linearity after each hidden layer
Batch Normalization: After each linear layer to stabilize and accelerate training
Dropout: To reduce overfitting, dropout rates of 0.2 for first two hidden layers and 0.1 for the last two.
Output layer: Fully connected layer with 10 outputs corresponding to the CIFAR-10 classes.
Training Details
Loss function: CrossEntropyLoss (suitable for multi-class classification)
Optimizer: Adam optimizer with initial learning rate 0.001
Learning rate scheduler: StepLR reducing learning rate by half every 10 epochs
Batch size: 64
Epochs: 30

Training Performance
The model was trained for 30 epochs.
Both training and validation losses decreased steadily, indicating effective learning.
Training accuracy improved consistently, demonstrating the model's ability to fit the training data.
Dropout and batch normalization helped mitigate overfitting and stabilized training.
Loss Curves

Training Accuracy Curve


Evaluation on Test Data
Test Accuracy: The model achieved a test accuracy of 59% on unseen data.
Classification Report:

Confusion Matrix: 

ROC Curve and AUC:
One-vs-All ROC curves plotted for each class.
Macro-average ROC AUC score: 0.9234










LeNet-5 Performance on CIFAR-10 Dataset
1. Model Architecture
The LeNet-5 architecture used consists of:
Convolutional layers:
Conv1: 3 input channels → 6 output channels, kernel size 5x5
Conv2: 6 → 16 channels, kernel size 5x5
Conv3: 16 → 120 channels, kernel size 5x5
Pooling layers:
Average pooling (2x2) after Conv1 and Conv2 layers
Fully connected layers:
FC1: 120 → 84 neurons
FC2: 84 → 10 output neurons (classes)
Activation function: ReLU after each convolution and FC layer except the output layer.

2. Data Preprocessing and Augmentation
To improve generalization, several data augmentation techniques were applied during training:
Random horizontal flipping
Color jittering (brightness, contrast, saturation adjustments)
Random cropping with padding of 4 pixels
Normalization with mean and standard deviation of (0.5, 0.5, 0.5) per channel
The dataset was split into 90% training and 10% validation sets. The test set was kept separate.

3. Training Setup
Loss function: Cross-Entropy Loss
Optimizer: AdamW with initial learning rate 0.001
Learning rate scheduler: StepLR (decays LR by 0.5 every 10 epochs)
Batch size: 32
Epochs: 30
Device: GPU (if available), otherwise CPU


4. Results
4.1 Training and Validation Loss
Training and validation losses decreased steadily over epochs, indicating effective learning.
The learning rate scheduler helped stabilize training by reducing the learning rate midway, preventing overfitting.

4.2 Accuracy
Final training accuracy reached approximately 65.12% (insert exact % from last epoch)
Final validation accuracy reached approximately 63.28% (insert exact % from last epoch)
4.3 Test Set Evaluation
The model was evaluated on the CIFAR-10 test set.
The classification report shows precision, recall, and F1-score for each class.
The confusion matrix highlights the class-wise performance and misclassification trends.



5. Detailed Metrics
5.1 Classification Report
5.2 ROC Curves and AUC
Multi-class ROC curves were plotted using one-vs-rest approach.
Area Under Curve (AUC) values indicate strong discriminative power across classes.
Most classes achieved AUC > 0.90, with some variation.

5.3 Precision-Recall Curves
Precision-Recall (PR) curves were plotted for each class.
Average precision (AP) scores were computed.

6. Observations
Augmentation techniques helped improve generalization on the validation and test sets.
Using AdamW optimizer and learning rate scheduling contributed to stable training.


 CNN 14 CNN12


 
CNN19 CNN8

Resnet152 
 Alexnet 
 






ANN5 ANN10


 
LeNet VGG16




 


