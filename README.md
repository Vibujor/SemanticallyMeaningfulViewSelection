# Semantic View Selection
This project contains the code for semantically meaningful view selection. It aims to autonomously find an optimal view for a given object in a given pose.
The code for our IROS 2018 paper: [Semantically Meaningful View Selection](https://arxiv.org/abs/1807.10303).

## Dataset
The complete dataset generated for this paper can be found [here](https://github.com/jorisguerin/SemanticViewSelection_dataset). Please download it and uncompress before executing this code.
This dataset is composed of around 10k images organized by category, object, pose and view.

## Feature extraction and semantic score labelling
The code for feature extraction from pretrained CNNs  and semantic score labeling using Monte-Carlo sampling of clustering problems will be uploaded later. For now, the results from these computation have been gathered in h5 files and can be downloaded [here](https://drive.google.com/drive/folders/1qQCI0ITyAUqYdkio2VnpiHKx2idZwlwH?usp=sharing).

## Training
The script used for training is *train.py*.
* It uses the the train and test *.h5* files as input.
* It generates a *model.json* and *my_weights.h5* files as ouput.

## Testing
The testing script is the *testNN.py*.
* It uses the file *model.json* and *my_weights.h5* to select the optimum view for each image in the clustering problem (model trained from features extracted with vgg19, hard-coded). 
* The variable *feature_extractor* is used to choose the network used to obtain the features that will be used to compare the views.
* vgg19 ?

## Launching the application
###The code is launched using the following command : 
```
Command to write in terminal
``` 
###Explaining the different arguments
* arg 1 
* arg 2 
* ...
***
###Cases where features have not been computed
Other script to launch to extract the features for the chosen network.
