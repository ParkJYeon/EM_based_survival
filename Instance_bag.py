import os
from numpy import array


def get_list_WSI(directory, train_folder):
    list_slide = []
    for index in train_folder:
        path = os.path.join(directory, index)
        img_list = os.listdir(path)
        list_wsi_dup = []
        for img in img_list:
            list_wsi_dup.append(img[0:7])
        list_slide.append(list(set(list_wsi_dup)))
    return list_slide


def get_wsi_count(directory, train_folder):
    list_slide = get_list_WSI(directory, train_folder)
    list_wsi = []
    for index in range(len(train_folder)):
        path = os.path.join(directory, train_folder[index])
        img_list = os.listdir(path)
        for slide in list_slide[index]:
            count = 0
            for img in img_list:
                if slide in img:
                    count += 1
            list_wsi.append((slide, count))
    return list_wsi


def get_instance_index_label(directory, train_folder):
    instance_index_label = []
    for index in train_folder:
        path = os.path.join(directory, index)
        img_list = os.listdir(path)
        for img in img_list:
            instance_index_label.append((path+'\\'+img, index))

    return instance_index_label


def bag_label_from_instance_labels(instance_labels):
    return int(instance_labels[0])


def data_generation_MIL(directory):
    train_folder = array(os.listdir(directory))
    list_wsi = get_wsi_count(directory, train_folder)
    instance_index_label = get_instance_index_label(directory, train_folder)
    bags, bags_per_instance_labels, bags_label, bags_discriminative = {}, {}, {}, {}

    for (wsi, _) in list_wsi:
        bags[wsi] = []
        bags_per_instance_labels[wsi] = []
        bags_discriminative[wsi] = []

    for _ in range(len(instance_index_label)):
        instance_idx, label = instance_index_label.pop()
        bags[instance_idx[46:53]].append(instance_idx)
        bags_per_instance_labels[instance_idx[46:53]].append(label)
        bags_discriminative[instance_idx[46:53]].append([instance_idx, 1])


    for (wsi, _) in list_wsi:
        bags_label[wsi] = bag_label_from_instance_labels(bags_per_instance_labels[wsi])

    return bags, bags_label, bags_discriminative

