import numpy as np

import keras
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input as prpc_vgg_res
from keras.applications.xception import preprocess_input as prpc_xce_inc
from keras.applications.mobilenet import preprocess_input as prpc_mob
from keras.applications.mobilenet_v2 import preprocess_input as prpc_mob2
from keras.applications.mobilenet_v3 import preprocess_input as prpc_mob3

from keras.models import Model
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.resnet50 as res
import keras.applications.inception_v3 as inc
import keras.applications.xception as xce
#networks added 21/04/2021 :
import keras.applications.mobilenet as mob
import keras.applications.mobilenet_v2 as mob2
import keras.applications.mobilenet_v3 as mob3

import pickle
import os
from glob import glob

from data_manip import *


def create_input_features(architecture, layer, set_type):
	
	if os.path.exists("Features/%s_%s/%s/" % (architecture, layer, set_type)):
		return

	else:
		print("Extracting features")
		os.makedirs("Features/%s_%s/%s/" % (architecture, layer, set_type))
		save_pickle_features(architecture, layer, set_type)


def save_pickle_features(architecture, layer, set_type):
	if architecture == "inception":
		base_model = inc.InceptionV3(weights='imagenet')
		tgt_sze = (299, 299)
		prpc_fct = prpc_xce_inc
	elif architecture == "resnet":
		base_model = res.ResNet50(weights='imagenet')
		tgt_sze = (224, 224)
		prpc_fct = prpc_vgg_res
	elif architecture == "vgg16":
		base_model = vgg16.VGG16(weights='imagenet')
		tgt_sze = (224, 224)
		prpc_fct = prpc_vgg_res
	elif architecture == "vgg19":
		base_model = vgg19.VGG19(weights='imagenet')
		tgt_sze = (224, 224)
		prpc_fct = prpc_vgg_res
	elif architecture == "xception":
		base_model = xce.Xception(weights='imagenet')
		tgt_sze = (299, 299)
		prpc_fct = prpc_xce_inc
	# 21/04/2021 adding mobilnet, mobilenetv1, mobilenetv2, densenet, efficientnet
	elif architecture == "mobilenet":
		base_model = mob.MobileNet(weights='imagenet')
		tgt_sze = (224, 224) #same for densenet
		prpc_fct = prpc_mob
	elif architecture == "mobilenetv2":
		base_model = mob2.MobileNetV2(weights='imagenet')
		tgt_sze = (224, 224)
		prpc_fct = prpc_mob2
	elif architecture == "mobilenetv3":
		base_model = mob3.MobileNetV3(weights='imagenet')
		tgt_sze = (224, 224)
		prpc_fct = prpc_mob3
	else:
		print("Error in Network name")
		return
	
	dir_feat = "Features/%s_%s/%s/" % (architecture, layer, set_type)
	dir_im = "Data/SemanticViewSelection_data/Images/"

	for l_ses_morts in base_model.layers:
		print(l_ses_morts.name)

	model = Model(inputs=base_model.input, outputs=base_model.get_layer(name=layer).output)

	features = []
	true_classes = []
	
	class_list = get_classes_list(set_type)
	data_description = get_description(set_type)

	for i_cl in range(len(data_description)):
		cat = class_list[i_cl]
		for i_ob in range(len(data_description[i_cl])):
			obj = 1 + i_ob
			for i_po in range(len(data_description[i_cl][i_ob])):            
				pose = 1 + i_po
				os.makedirs("Features/%s_%s/%s/%s/%s/%s/" % (architecture, layer, set_type, cat, obj, pose))
				for i_vi in range(len(data_description[i_cl][i_ob][i_po])):
					view = str(data_description[i_cl][i_ob][i_po][i_vi]).replace(", ", "_").strip("[").strip("]")
					file = "%s/%s/%s/%s" % (cat, obj, pose, view)

					pil_im = image.load_img(dir_im + file + ".png", target_size=tgt_sze)
					arr_im = image.img_to_array(pil_im)
					prpc_im = prpc_fct(np.expand_dims(arr_im, axis=0))
					features = np.ndarray.flatten(model.predict(prpc_im))
					pickle.dump(features, open(dir_feat + file + ".p", "wb"), protocol=4)

	keras.backend.clear_session()
