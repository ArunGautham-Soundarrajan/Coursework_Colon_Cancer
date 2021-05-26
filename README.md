# Coursework Overview
Colon cancer is the 3rd most common cancer in the world.

The normal colon has projections known as villi which absorb food and appear as circles of cells on a microscope image of the tissue (since the villi are cut through):

The cell nuclei of these images can be segmented and coloured according to the different cell types:
(Orange cells are normal, red cells are cancer, green cells are immune cells invading to help, and blue cells are connective tissue.)

Digital pathology aims to apply machine learning to digital pathology images like these to determine if the tissue is normal or cancerous. One approach is to segment out the nuclei of cells from these images and classify them into different cell types including malignant cell nuclei (i.e. cancerous cells).

The overall goal for this data science task is to train a deep neural network which can take a 64x64 image with a cell nuclei at the centre and classify it into one of the following types,

* Normal epithelial cells.
* Cancer epithelial cells.
* Immune Leukocyte cells.
* Connective fibroblast cells.

# Code and Resources Used
* Python Version : 3.7.6
* Packages: pandas, sklearn, pytorch, torchvision, matplotlib, seaborn.
* Data: Provided by the University of Glasgow. [Kaggle Competition Link](https://www.kaggle.com/c/deep-learning-for-msc-coursework-2021)

# About the Data

Images are 64 * 64 pixel and coloured. Lets look images for each type of classification,
![Normal](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/normal.png)

The above are the some of the images of normal cells.

![Immune](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/Immune.png)

The above are the some of the images of Immune cells.

![Cancer](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/cancer.png)

The above are the some of the images of cancer cells.

![Connective](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/connective.png)

The above are the some of the images of connective cells. The data is unbalanced as the normal cell type images were very limited
![Count Plot](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/count_plot.png)

# Data Preprocessing

These are the following things which I did to tackle the unbalanced data and to process it.

* ### Normalising 

I normalised the images, which is done by subratracting the mean and dividing it by the standard deviation of the data(*Standardising*). Here are the images before and after normalising

![Before Normalising](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/bf_normalising.png) ![After Normalising](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/af_normalising.png)

* ### Data Augmentation

We only had about 2190 images to work with, which is'nt sufficient for a high performing model. So we had to incorpate data augmentation techniques to *generalise* the model. 

* ### Sampling

Sampling is one of the methods to deal with unbalanced data. I used *Weighted Random Sampler* and assigned weights for each type, so that a type with less data is sampled much to meet the rest of the types.

* ### Class Weights 

One other method to deal with unbalanced data is to use class weights when assigning the optimiser. It mulitplies the loss of the mispredicted class with weight we specified. So loss will increased for mis prediction done on a class with heigher weights.

# Things which did'nt work well

1. Inverting the images 
2. Grayscale images
3. Making a copy of dataset with fully transformed data( Model started to overfit)

# Model Building

Splited the data into train and test set using RandomSplit. The model was built using pytorch.
* Implemented a simple model with few convolutional layers and linear layers.
* Then tried to increase the complexity of the model by increasing the filters and neurons and increasing the layers. Which didnt work well.
* So reverted the changes and added few dropout layers and batch normalisation. And removed batch normalisation as it didnt produce expected results.
* Implemented Adam optimiser with Scheduled Learning Rate and also added weight decay to it to avoid overfitting.

# Model Performance. 

The Overfitted model achieved an accuracy above 90 percent, which showed the maximum capability. So the final model achived an accuracy above *75 percent* on test data. The plot below shows the trend.

![Accuracy](https://github.com/ArunGautham-Soundarrajan/Coursework_Colon_Cancer/blob/main/images/acc.png).

Hyperparameter tuning could have increased the performce, but sadly couldnt find enough time to implement it. 


 


