from data_manip import *
import numpy as np
import h5py
from os import listdir
import pickle


def h5_generator(architecture, layer):
    # loading labels from fullset
    vggfull = h5py.File("vgg_block5pool_fullset.h5", "r")
    vgg_labels = vggfull["labels"][:]
    vggfull.close()

    # selecting test classes spoon and clamp
    test_classes = [3, 17]

    data_description = get_description("train")

    # listcl = listdir("Features/mobilenet_global_average_pooling2d/train")
    #loading list of classes
    listcl = listdir("Features/%s_%s/train" % (architecture, layer))
    listcl = np.array(listcl)
    print(listcl)
    angles_test = []
    feat_test = []
    labs_test = []

    angles_train = []
    feat_train = []
    labs_train = []

    i = 0
    testind = 0
    trainind = 0

    for cl in range(len(data_description)):
        for ob in range(len(data_description[cl])):
            for po in range(len(data_description[cl][ob])):
                for vi in range(len(data_description[cl][ob][po])):
                    if cl in test_classes:
                        angles_test.append(data_description[cl][ob][po][vi])
                        # feat_string = "Features/mobilenet_global_average_pooling2d/train/%s/%d/%d/%d_%d.p" % (
                        # listcl[cl], ob + 1, po + 1, data_description[cl][ob][po][vi][0],
                        # data_description[cl][ob][po][vi][1])
                        feat_string = "Features/%s_%s/train/%s/%d/%d/%d_%d.p" % (architecture, layer,
                                                                                 listcl[cl], ob + 1, po + 1,
                                                                                 data_description[cl][ob][po][vi][0],
                                                                                 data_description[cl][ob][po][vi][1])
                        feat_fich = open(feat_string, "rb")
                        feat_test.append(pickle.load(feat_fich))
                        labs_test.append(vgg_labels[i])
                        feat_fich.close()
                        i += 1
                        testind += 1
                    else:
                        angles_train.append(data_description[cl][ob][po][vi])
                        feat_string = "Features/%s_%s/train/%s/%d/%d/%d_%d.p" % (architecture, layer,
                                                                                 listcl[cl], ob + 1, po + 1,
                                                                                 data_description[cl][ob][po][vi][0],
                                                                                 data_description[cl][ob][po][vi][1])
                        feat_fich = open(feat_string, "rb")
                        feat_train.append(pickle.load(feat_fich))
                        labs_train.append(vgg_labels[i])
                        feat_fich.close()
                        i += 1
                        trainind += 1

    #creating the files
    testset_name = ("%s_%s_testset.h5" % (architecture, layer))
    fitest = h5py.File(testset_name, "w")
    fitest.create_dataset("features", data=feat_test)
    fitest.create_dataset("th_ph", data=angles_test)
    fitest.create_dataset("labels", data=labs_test)
    fitest.close()

    trainset_name = ("%s_%s_trainset.h5" % (architecture, layer))
    fitrain = h5py.File(trainset_name, "w")
    fitrain.create_dataset("features", data=feat_train)
    fitrain.create_dataset("th_ph", data=angles_train)
    fitrain.create_dataset("labels", data=labs_train)
    fitrain.close()