# Fashion MNIST CNN Classifier

![Project Image](https://github.com/AmirFARES/Fashion-MNIST-CNN/blob/main/imgs/fashionLogo.png)

## Introduction 🌟

Welcome to the "Fashion MNIST CNN Classifier" project, a showcase of our capabilities in image classification using Convolutional Neural Networks (CNN). Our primary goal was to build an efficient model to classify fashion items from the Fashion MNIST dataset, a task that is far from trivial given the visual similarity between certain classes. We proudly report an impressive accuracy of 92.3% on the validation set. Notably, our project faced challenges in distinguishing between closely related fashion classes, such as sneakers and ankle boots, a testament to the intricacies of image classification. This project underscores our commitment to tackling real-world image recognition challenges with a focus on accuracy and robustness, demonstrating our ability to effectively leverage deep learning techniques to solve complex problems in the fashion domain.

## Key Objectives 🎯

Our primary objectives for this project were to recognize and classify each fashion piece within the Fashion MNIST dataset. We set specific criteria and goals, aiming to achieve an accuracy rate of over 90%. Remarkably, we met and exceeded this objective, achieving an accuracy of 92.3% on the first attempt. To approach the project, we followed a structured methodology, which included dataset preparation by splitting datasets, checking labels and their distribution, handling missing data, normalizing and reshaping, label encoding, and splitting into training and validation sets. Additionally, we meticulously designed and trained the model, accompanied by visualizing results to gain insights into its performance and areas of improvement.

## Data Sources 📊

All data for this project was obtained from the [**Fashion MNIST Dataset**](https://www.kaggle.com/datasets/zalando-research/fashionmnist). The dataset includes two primary files:

- **fashion-mnist_train.csv** - containing 60,000 examples, where each example is a 28x28 grayscale image associated with a label from 10 distinct fashion classes.
- **fashion-mnist_test.csv** - containing 10,000 examples for model evaluation. 

Additionally, for an in-depth look at the project and its development, you can access our [**Notebook on Kaggle**](https://www.kaggle.com/code/amirfares/fashion-mnist-cnn-92-3-accuracy) or view the code in [**notebook.ipnyb**](https://github.com/AmirFARES/Fashion-MNIST-CNN/blob/main/fashion-mnist-cnn-92-3-accuracy.ipynb).

## Methodology 🚀

Our approach to building and training the CNN model involved initially starting with a simple architecture and progressively refining it. We encountered challenges such as overfitting and the time-consuming training process due to the resource limitations of free cloud services. To address overfitting, we incorporated multiple dropout layers, which proved to be effective. The overall methodology for image classification comprised several key steps, including data preprocessing tasks like data splitting, label and distribution checks, handling missing values, normalization, reshaping, and label encoding. We then proceeded to design, train, and fine-tune the model, making it more robust and capable of tackling the complexities of fashion item classification. Visualization of results played an integral role in understanding the model's performance and identifying areas for improvement.

## Data Preprocessing 🛠️

Before diving into model training, we performed essential data preprocessing tasks to ensure the dataset was ready for our CNN model. Here's an overview of the steps we took:

- **Reshaping and Normalization**
- **Label Encoding**
- **Data Splitting**

Our data preprocessing ensured that our dataset was well-structured and ready for training our CNN model.

## Model Architecture 🏗️

Our CNN model's architecture was designed to effectively classify fashion items. The model's layers are as follows:

```plaintext
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D) (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 13, 13, 32)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling2D) (None, 5, 5, 64)         0         
 )                                                               
                                                                 
 dropout_1 (Dropout)         (None, 5, 5, 64)          0         
                                                                 
 flatten (Flatten)           (None, 1600)              0         
                                                                 
 dense (Dense)               (None, 64)                102464    
                                                                 
 dropout_2 (Dropout)         (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 121,930
Trainable params: 121,930
Non-trainable params: 0
_________________________________________________________________

```

## Training and Evaluation 📈

- **Optimization:** We used the Adam optimizer with default parameters.
- **Loss Function:** Categorical cross-entropy.
- **Epochs and Early Stopping:** The model was trained for 35 epochs, and we implemented early stopping.
- **Evaluation Metrics:** We assessed the model's performance using accuracy as the evaluation metric.

Our model's training and evaluation strategy aimed to find the right balance between accuracy and generalization after multiple tries.

## Conclusion 🎯

- Achieved an accuracy of 92.3% on the validation set.
- Successful in classifying fashion items, showcasing the effectiveness of CNN models.
- Some challenges observed with similar fashion categories like pullovers, shirts, and coats.
- Valuable insights for improvement gained from the confusion matrix and misclassified images.
<img src="https://github.com/AmirFARES/Fashion-MNIST-CNN/blob/main/imgs/confusionMatrix.png" alt="Line Chart" width="500" height="470">  

<img src="https://github.com/AmirFARES/Fashion-MNIST-CNN/blob/main/imgs/misclassified.png" alt="Line Chart" width="500" height="384">  

- Excited to share our findings with the community and continue improving our model.

## Future Work 🚧

- Further fine-tuning the model to enhance performance.
- Exploring data augmentation techniques for increased diversity.
- Focusing on addressing misclassified labels, especially for similar fashion categories observed in previous results.


## Connect with Me 📫

Feel free to reach out for any project-related inquiries, collaboration opportunities, or discussions. You can connect with me on [LinkedIn](https://www.linkedin.com/in/amir-f), explore more of my projects on [GitHub](https://github.com/AmirFARES), and check out my portfolio [here](https://amirfares.github.io/).

## Acknowledgments 🙏

I'd like to express my gratitude to the providers of the fashion MNIST dataset.

Thank you for visiting my project repository, and I'm excited to share more data-driven insights in the future!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/amir-f)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-orange)](https://amirfares.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-green)](https://github.com/AmirFARES)
