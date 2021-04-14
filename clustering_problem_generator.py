import numpy as np
from numpy.random import randint, shuffle
import pickle

class CPG:

	def __init__(self, class_list, data_description, feature_extractor, set_type):

		self.class_list = class_list
		self.data_description = data_description
		self.feature_extractor = feature_extractor
		self.set_type = set_type
        #max_classes max number of classes
		self.max_classes = len(self.class_list)

	def generate_cp(self):

		self.clustering_problem = []
        
        #n_cl between 3 and max_classes (max nb of classes)
		n_cl = randint(3, self.max_classes)

        #shuffle_classes ordered list 0:max_classes...
		shuffle_classes = list(range(self.max_classes))
        #... randomly shuffled
		shuffle(shuffle_classes)
        #we keep the n_cl first values in the list in problem_classes
		problem_classes = shuffle_classes[:n_cl]
        
		for cl in problem_classes:
			n_obj_tot = len(self.data_description[cl]) + 1
            #n_obj number of objects in the problem
			n_obj = randint(1, n_obj_tot)
            #shuffle_obj ordered list 0:n_obj_tot-1
			shuffle_obj = list(range(n_obj_tot - 1))
			shuffle(shuffle_obj)
            #keep n_obj randomly shuffled from all objects
			objects = shuffle_obj[:n_obj]
			for obj in objects:
                #randomly selecting a pose and a view
				n_pose_tot = len(self.data_description[cl][obj])
				pose = randint(n_pose_tot)
				n_view_tot = len(self.data_description[cl][obj][pose])
				view = randint(n_view_tot)
                #adding to clustering_problem a class, object, pose and view
				self.clustering_problem.append([cl, obj, pose, view])

		self.convert2folderDescription()
		self.convert2featuresLabels()

	def convert2folderDescription(self):

		self.cp_folderDescription = []

		for c in self.clustering_problem:
            #extract cat, obj, pos and view to create a descriptor
			cat = self.class_list[c[0]]
			obj = 1 + c[1]
			pose = 1 + c[2]
			view = self.data_description[c[0]][c[1]][c[2]][c[3]]

			self.cp_folderDescription.append([cat, obj, pose, "%d_%d" % (view[0], view[1])])

	def print_cp(self):
        #print all folder descriptions (representation of all clustering problems generated)
		for c in self.cp_folderDescription:
			print(c)


	def convert2featuresLabels(self):

		self.features, self.labels = [], []
        #create path
		base_path = "Features/" + self.feature_extractor + "/" + self.set_type + "/"

		for c in range(len(self.cp_folderDescription)):
			cat = self.clustering_problem[c][0]
            #add the class to labels
			self.labels.append(cat)

            #extracting the name of the file corresponding to our problem
			c2 = self.cp_folderDescription[c]
			fname = base_path + c2[0] + "/" + str(c2[1]) + "/" + str(c2[2]) + "/" + c2[3] + ".p"
            #add file to features
			file = open(fname, "rb")
			self.features.append(pickle.load(file))
			file.close()

		self.features = np.array(self.features)
