# Semantic View Selection

The code for our IROS 2018 paper: [Semantically Meaningful View Selection](https://arxiv.org/abs/1807.10303).

## Dataset
The complete dataset generated for this paper can be found [here](https://github.com/jorisguerin/SemanticViewSelection_dataset). Please download it and uncompress before executing this code.

## Feature extraction and semantic score labelling
The code for feature extraction from pretrained CNNs  and semantic score labeling using Monte-Carlo sampling of clustering problems will be uploaded later. For now, the results from these computation have been gathered in h5 files and can be downloaded [here](https://drive.google.com/drive/folders/1qQCI0ITyAUqYdkio2VnpiHKx2idZwlwH?usp=sharing).

## Training
The script used for training is *train.py*.
* It uses the the train and test *.h5* files as input.
* It generates a *model.json* and *my_weights.h5* files as ouput.

## Testing
The testing script is the *evaluateNN.ipynb* jupyter notebook.
