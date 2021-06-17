import numpy as np

from feature_extraction import create_input_features, save_pickle_features
from create_h5 import h5_generator
from train import training_model
from testNN import testing

import argparse

# --help default

#argument parsing
#layer example : conv_pw_10_relu
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architecture', type=str,
                    help='architecture used to extract features (inception, resnet, vgg16, vgg19, xception or mobilenet)')
parser.add_argument('-l', '--layer', type=str, help='layer used to extract features')
args = parser.parse_args()

create_input_features(args.architecture, args.layer, "test")
create_input_features(args.architecture, args.layer, "train")
h5_generator(args.architecture, args.layer)

training_model(args.architecture, args.layer)
testing(args.architecture, args.layer)
