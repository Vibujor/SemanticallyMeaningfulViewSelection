import pickle

def get_classes_list(set_type = "train"):
    file = open("Annotations/classes_list_" + set_type + ".p", "rb")
    class_list = pickle.load(file)
    file.close()
    
    return class_list

def get_description(set_type = "train"):
    file = open("Annotations/description_" + set_type + ".p", "rb")
    description = pickle.load(file)
    file.close()
    
    return description