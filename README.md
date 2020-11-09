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

## Progress:
### Preprocessing
Read images as gray scale images (1024, 1024). Downsample to (256,256)
by averaging the pixel values in each 4 by 4 pixel block. Finally,
flatten the 2d matrix into a 1d vector.

### Results
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

## TODO
1. CNN: ResNet50 or VGG16
2. PCA-SVM pipeline: prettier visualizations
