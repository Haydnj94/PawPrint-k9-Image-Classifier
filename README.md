# PawPrint-k9-Image-Classifier
- Desktop - https://pawprintk9.streamlit.app/
- Mobile - https://pawprintk9-mobile.streamlit.app/

This repository contains a deep learning model built using Convolutional Neural Networks (CNN) to classify images of dog breeds from the Stanford Dogs Dataset. The model uses the pre-trained MobileNet model for feature extraction and fine-tuning, with custom preprocessing and data augmentation to optimize performance.

## Dataset

The **Stanford Dogs Dataset** (http://vision.stanford.edu/aditya86/ImageNetDogs/) consists of 120 dog breeds from around the world. The dataset contains:

- **Number of categories**: 120
- **Number of images**: 20,580
- **Annotations**: Class labels and bounding boxes

The dataset was sourced from ImageNet and is designed for fine-grained image categorization. Each breed has a subfolder containing images. A dataframe was created by iterating through each image and defining the full image path in a column called `image_path`. The breed names from the subfolder names were cleaned and assigned to the `breed` column.

## Data Preprocessing

1. **Image Resizing & Normalization**: The original images varied in size and aspect ratio, so a function was created to resize all images to **224x224** while maintaining aspect ratios. Padding was added when necessary, and pixel values were normalized between 0 and 1 by dividing by 255.
   
2. **Data Split**: The dataset was split into 3 subsets:
    - **Training**: 80% of the images (16,464 samples)
    - **Validation**: 10% of the images (2,058 samples)
    - **Testing**: 10% of the images (2,058 samples)

   Additionally, the breed labels were encoded into integers, and a new column, `breed_index`, was created for model training.

## Model Architecture

The model was built using **MobileNet**—a lightweight, pre-trained deep learning model. MobileNet uses depthwise separable convolutions, which reduce computational complexity without sacrificing performance, making it ideal for this project given the computational limitations.

### Image Augmentation
To enhance the model's ability to generalize and deal with a small dataset, **image augmentation** techniques were applied. This increased the variety of images and allowed the model to learn more robust features.

### Transfer Learning Strategy
- **Frozen Layers**: Initially, all layers of MobileNet were frozen to keep the model lightweight and prevent overfitting. Only the final classification layer was trained.
- **Optimizer**: The Adam optimizer was used with a learning rate of **1e-4**, optimizing for accuracy.
- **Callbacks**: To prevent overfitting, callbacks such as **early_stopping**, **model_checkpoint**, and **reduce_lr** were used during training.

### Training Results (First Model)
- **Training time**: 1 hour and 31 minutes
- **Final Epoch Results**:
    - **Training Accuracy**: 77.15%
    - **Validation Accuracy**: 80.17%
    - **Test Accuracy**: 79.5%

Although these results were satisfactory, further optimization was attempted to improve accuracy.

### Advanced Model
In an attempt to handle the class imbalance and improve model performance, the following changes were made:
1. **Unfreezing the last 20 layers** of MobileNet to allow the model to learn from the images directly.
2. **Class balancing**: The model was adjusted to train more on underrepresented breeds.
3. **Training for 50 Epochs**.

However, this approach led to some overfitting, as seen by the drop in validation and test accuracy.

### Training Results (Advanced Model)
- **Training time**: 1 hour, 9 minutes, and 27 seconds
- **Final Epoch Results**:
    - **Training Accuracy**: 82.09%
    - **Validation Accuracy**: 73.66%
    - **Test Accuracy**: 76.58%

### Optimized Model
In the final model, the layers were frozen again to balance the dataset, and additional neurons and dense layers were added. This helped to improve the results.

### Final Results (Optimized MobileNet Model)
- **Training Accuracy**: 80.76%
- **Validation Accuracy**: 80.52%
- **Test Accuracy**: 80.61%

This result was satisfactory, as the target of 80% accuracy was achieved.

### Other Experiments
I also tried training the model using **ResNet50**, a more complex architecture. However, after 10 epochs, the model showed no learning progress, and I did not have sufficient time to optimize it further.

### Model Evaluation
I plotted the evaluation of the best model using:
- Line graphs
- Accuracy reports
- Confusion matrix

## Deployment

Finally, the model was deployed on **Streamlit Cloud**, where it was tested with unseen images. The predictions were accurate, with a confidence level of above **80%** for all tested images.

## Conclusion

The project demonstrates a successful implementation of a CNN-based image classifier for dog breeds using the Stanford Dogs Dataset. Despite the challenges of class imbalance and limited data, the final model achieved a solid performance and was successfully deployed for real-time predictions.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Streamlit

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/PawPrint-k9-Image-Classifier.git
cd PawPrint-k9-Image-Classifier
