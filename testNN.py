#imports
from data_manip import *
from clustering_problem_generator import *
from keras.models import model_from_json
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import fowlkes_mallows_score as fms
from sklearn.metrics import normalized_mutual_info_score as nmi
from purity import *

import pickle

import numpy as np


def testing(architecture, layer):

    class_list = get_classes_list("test")
    data_description = get_description("test")
    feature_extractor = "xception"

    cpg = CPG(class_list, data_description, feature_extractor, "test")

    #json_file = open('model2.json', 'r')
    name_json = "model_%s_%s.json" % (architecture, layer)
    json_file = open(name_json, 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    name_w = "my_weights_%s_%s.h5" % (architecture, layer)
    loaded_model.load_weights(name_w)

    nmi_rand, nmi_opt, nmi_top = [], [], []
    fm_rand, fm_opt, fm_top = [], [], []
    pur_rand, pur_opt, pur_top = [], [], []

    base_path = "Features/" + cpg.feature_extractor + "/" + cpg.set_type + "/"
    for i in range(100):
        if i % 100 == 0:
            print(i)
        features_opt, features_top = [], []

        cpg.generate_cp()
        n_classes = len(list(set(cpg.labels)))
        labels = np.array(cpg.labels)

        for c in cpg.clustering_problem:
            # po_list does not contain the view
            po_list = c[:-1]

            # rajouter test ?
            #top_image_file = "./Features/vgg19block5_pool/%s/%d/%d/90_90.p" % (class_list[po_list[0]], 1 + po_list[1], 1 + po_list[2])
            top_image_file = "./Features/%s_%s/test/%s/%d/%d/90_90.p" % (architecture, layer, class_list[po_list[0]], 1 + po_list[1], 1 + po_list[2])
            features = pickle.load(open(top_image_file, "rb"))

            # initialising optimum view
            opt_view = [[0, 0], 0]
            for vi in data_description[po_list[0]][po_list[1]][po_list[2]]:
                ph_th = -np.pi + (np.array(vi) * np.pi / 180)
                th_ph = np.array([ph_th[-1], ph_th[0]])

                # searching for optimum view
                pred = loaded_model.predict([np.array([features[:]]), np.array([th_ph[:]])])
                print(pred[0][0])
                if pred[0][0] > opt_view[1]:
                    opt_view = [(ph_th + np.pi) * 180 / np.pi, pred[0][0]]

            opt_view[0] = [int(round(opt_view[0][0])), int(round(opt_view[0][1]))]

            view_str = str(opt_view[0]).strip("]").strip("[").replace(", ", "_")

            # creating features for optimum view
            fname = base_path + class_list[po_list[0]] + "/" + str(po_list[1] + 1) + "/" + str(
                po_list[2] + 1) + "/" + view_str + ".p"
            file = open(fname, "rb")
            features_opt.append(pickle.load(file))
            file.close()

            # creating features for top view
            fname = base_path + class_list[po_list[0]] + "/" + str(po_list[1] + 1) + "/" + str(po_list[2]+1)+"/90_90.p"
            file = open(fname, "rb")
            features_top.append(pickle.load(file))
            file.close()

        features_opt = np.array(features_opt)
        features_top = np.array(features_top)

        # testing on optimum view
        agg = AgglomerativeClustering(n_classes)
        clusters = agg.fit_predict(features_opt)
        score_glob_opt = fms(clusters, labels)  # fms score
        fm_opt.append(score_glob_opt)
        score_glob_opt = nmi(clusters, labels)  # nmi score
        nmi_opt.append(score_glob_opt)
        score_glob_opt = purity(clusters, labels)  # purity
        pur_opt.append(score_glob_opt)

        # testing on random view
        clusters = agg.fit_predict(cpg.features)
        score_glob_rand = fms(clusters, labels)  # fms score
        fm_rand.append(score_glob_rand)
        score_glob_rand = nmi(clusters, labels)  # nmi score
        nmi_rand.append(score_glob_rand)
        score_glob_rand = purity(clusters, labels)  # purity
        pur_rand.append(score_glob_rand)

        # testing on top view
        clusters = agg.fit_predict(features_top)
        score_glob_top = fms(clusters, labels)  # fms score
        fm_top.append(score_glob_top)
        score_glob_top = nmi(clusters, labels)  # nmi score
        nmi_top.append(score_glob_top)
        score_glob_top = purity(clusters, labels)  # purity
        pur_top.append(score_glob_top)

    print(np.mean(fm_opt), np.mean(nmi_opt), np.mean(pur_opt))
    print(np.mean(fm_top), np.mean(nmi_top), np.mean(pur_top))
    print(np.mean(fm_rand), np.mean(nmi_rand), np.mean(pur_rand))

    #testing matrix
    mat_res = np.array([[np.mean(fm_opt), np.mean(nmi_opt), np.mean(pur_opt)],
                        [np.mean(fm_top), np.mean(nmi_top), np.mean(pur_top)],
                        [np.mean(fm_rand), np.mean(nmi_rand), np.mean(pur_rand)]])

    name_res = "results_%s_%s_skip_100.txt" % (architecture, layer)
    f = open(name_res, "a")
    #f.write("Results for %s layer %s" % (architecture, layer))
    str1 = "\nOptimum (fm, nmi, pur) :\t" + str(np.mean(fm_opt)) + " " + str(np.mean(nmi_opt)) + " " + str(np.mean(pur_opt))
    str2 = "\nTop (fm, nmi, pur) :\t" + str(np.mean(fm_top)) + " " + str(np.mean(nmi_top)) + " " + str(np.mean(pur_top))
    str3 = "\nRandom (fm, nmi, pur)\t: " + str(np.mean(fm_rand)) + " " + str(np.mean(nmi_rand)) + " " + str(np.mean(pur_rand)) + " \n \n"
    f.write(str1)
    f.write(str2)
    f.write(str3)
    f.close()

    return mat_res