# Semantic View Selection
This project contains the code for semantically meaningful view selection. It aims to autonomously find an optimal view for a given object in a given pose.
The code for our IROS 2018 paper: [Semantically Meaningful View Selection](https://arxiv.org/abs/1807.10303).

## Dataset
The complete dataset generated for this paper can be found [here](https://github.com/jorisguerin/SemanticViewSelection_dataset). Please download it and uncompress before executing this code.
This dataset is composed of around 10k images organized by category, object, pose and view.

## Feature extraction and semantic score labelling
The code for feature extraction from pretrained CNNs  and semantic score labeling using Monte-Carlo sampling of clustering problems, contained in *feature_extraction.py*.
* Contains the function *create_input_features* calling *save_pickle_features*, which takes as inputs the chosen architecture used to extract the features the corresponding layer, and the set type (train or test) to compute the features from.
* If features for the chosen architecture, layer and set type have already been computed, the function *save_pickle_features* will not be called and features will not be recalculated.
* Stores features in .p files, in Features/architecture/set_type, organized similarly to the .png data.
The results from these computation have been gathered in h5 files and can be downloaded [here](https://drive.google.com/drive/folders/1qQCI0ITyAUqYdkio2VnpiHKx2idZwlwH?usp=sharing).

## Training
The script used for training is *train.py*.
* Contains function *training_model* used to train perceptrons.
* It uses the the train and test *.h5* files as input.
* It generates a *model.json* and *my_weights.h5* files as ouput.

## Testing
The testing script is the *testNN.py*.
* Contains the function *testing* with parameters architecture and layer (architecture and layer used to extract features).
* It uses the file *model.json* and *my_weights.h5* to select the optimum view for each image in the clustering problem (model trained from features extracted with vgg19, hard-coded). 
* The variable *feature_extractor* is used to choose the network used to obtain the features that will be used to compare the views.
* Saves results in a text file named results_model_layer

## Installation
### Data 
* Download the dataset [here](https://github.com/jorisguerin/SemanticViewSelection_dataset)
* Download extracted features (file vgg_block5pool_fullset.h5) [here]https://drive.google.com/drive/folders/1t8_EfyqK3B_7Q4Xo34uZfzjKzsp-Dbpg).
This file will be used to extract labels and create new features.
* Make sure to organize your data and features in folders "Data" and "Features" in your directory. 
### Python files
* feature_extraction.py
* create_h5.py
* train.py
* testNN.py
* purity.py
* clustering_problem_generator.py
* launch_skip 

These files will need to be in the same directory as the data folders.

## Launching the application
### The code is launched using the following command : 
```
./launch_skip -a "architecture" -l "layer"
or 
./launch_skip --architecture "architecture" --layer "layer"
``` 
### Command arguments
* -a --architecture : architecture used to extract features (inception, resnet, vgg16, vgg19, xception or mobilenet)
* -l --layer : layer used to extract features
***