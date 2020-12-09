# masked-face-classifier
Final project for CS4262. Group members: Chang Yu, Amy Chen, Hunter Wang

The scripts in this project assume the directory structure below:
```
/masked-face-classifier
    |_/CMFD <- correctly masked face images
        |_/00000
            |_00000_Mask.jpg
            |_00001_Mask.jpg
                ...
        |_/00010
          ...
    |_/IMFD <- incorrectly masked face images
        |_/00000
        |_/00010
           ...
```

To run the pca-svm pipeline, clone this repo, change to the corresponding directory, and simply type the command `python pca-svm.py`.

## PCA-SVM Pipeline:
### Preprocessing
Read images as gray scale images (1024, 1024). Downsample to (256,256)
by averaging the pixel values in each 4 by 4 pixel block. Finally,
flatten the 2d matrix into a 1d vector.

### Results
A straightforward grid search cross validation is used for selecting the hyperparameters for SVM. The parameter grid used is as below:
```
'C':     [1e3,    5e3,    1e4,   5e4,   1e5]
'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
```
Below is the classification report for the pca-svm model I just trained (~2000 images used):
```
                    precision    recall  f1-score   support
Incorrectly maksed       0.85      0.83      0.84       219
  Correctly masked       0.79      0.81      0.80       171
          accuracy                           0.82       390
         macro avg       0.82      0.82      0.82       390
      weighted avg       0.82      0.82      0.82       390
```

And below is the normalized confusion matrix:

![Confusion Matrix](/confusion_matrix_pca.png?raw=true)

Considering the fact that I only used a very small portion of the dataset, I'd say it's looking pretty good so far.

## VGG16 Model
### Preprocessing
Loaded the data in as target size (224,224) to construct a VGG16 model using Keras.

### Results
For the VGG16 model, the model was trained using 2540 images (equally masked and unmasked) and tested on a test set of 300 images each and a test set of 592 images. The training and validation accuracies and corresponding losses are shown in Figure 4 below. The training process stopped after 46 epochs and took about 1220 seconds with a validation accuracy of 99.7%. Then we used the best model to test on the test set of images and the final test accuracy is 98.0%. 

## Discussion
The results from both methods are very promising. The accuracy of the PCA-SVM method is higher than our previous expectation. The VGG16 Model works very well for this task as our expectation. However, there are still some limitations in our research and experiments:
- We only used a very small subset of the dataset. The hosting server of the dataset is located in western europe, so the downloading is very slow and unstable. Thus, it was unrealistic for us to download a larger subset of the data. 
- The VGG16 model requires a decent GPU. However, we didn’t have a good enough GPU to run the model, so we chose to use Google Colab to run our experiments. Because the training is very slow, we still used a very small subset of the original dataset.
- The dataset we used was simulated and does not come from real world photos of people wearing masks correctly/incorrectly. It would be interesting to further investigate how the models perform on real-world data. 
- The faces in the dataset always have masks on - they are just not worn correctly. It would be more interesting to see if our models could correctly detect faces without masks at all despite being only trained on a dataset whose subjects always had masks on.


