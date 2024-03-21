# Fire-Detection-Model




## Technologies:
- **Convolutional Neural Networks (CNN):** The model architecture includes convolutional layers, which are well-suited for image classification tasks.
- **Transfer Learning:** The code uses a pre-trained ResNet50 model and fine-tunes it for fire detection. Transfer learning enables leveraging knowledge from pre-trained models to improve performance on a specific task.
- **Data Augmentation:** The ImageDataGenerator class is used to apply data augmentation techniques like width and height shifts, which increase the diversity of training samples and help prevent overfitting.
- **Adam Optimizer:** The model is optimized using the Adam optimizer, which adapts the learning rate during training to achieve faster convergence and better performance.

## Data Preparation:
- The code uses the ImageDataGenerator class from TensorFlow to generate augmented training data. It applies width and height shifts to the images and splits the data into training and validation sets.
- The flow_from_directory method is used to load the training and validation data, with a target size of (300, 300) and a batch size of 8.

## Model Architecture:
- The code utilizes a pre-trained ResNet50 model as the base model by loading it with pre-trained weights from the ImageNet dataset. The last 10 layers of the model are made trainable, while the remaining layers are frozen.
- The pre-trained ResNet50 model is added to a sequential model along with additional layers, including dense layers and dropout layers.
- The model is compiled with the Adam optimizer and binary cross-entropy loss.

## Training:
- The model is trained using the fit function, with the training and validation sets as inputs. The number of epochs is set to 100.
- Two callbacks are used during training: LearningRateScheduler and ModelCheckpoint. The learning rate scheduler adjusts the learning rate during training, while the model checkpoint saves the best model based on validation loss.






## Results:
- The training process shows that the model achieves a loss of 0.0569 and an accuracy of 0.9750 on the validation set.
- These results indicate that the model performs well in detecting fire based on the provided dataset.
- The model is trained for 100 epochs, and the training process shows a decreasing loss and increasing accuracy on the validation set over time.
