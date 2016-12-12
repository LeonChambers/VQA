#!/usr/bin/env python

import os.path
import subprocess

annotations_urls = [
    "http://visualqa.org/data/abstract_v002/vqa/Annotations_Train_abstract_v002.zip",
    "http://visualqa.org/data/abstract_v002/vqa/Annotations_Val_abstract_v002.zip"
]
questions_urls = [
    "http://visualqa.org/data/abstract_v002/vqa/Questions_Train_abstract_v002.zip",
    "http://visualqa.org/data/abstract_v002/vqa/Questions_Val_abstract_v002.zip"
]
images_urls = {
    "train2015": "http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip",
    "val2015": "http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip"
}

if __name__ == '__main__':
    for url in annotations_urls:
        filename = url.split("/")[-1]
        if not os.path.isfile("Annotations/{}".format(filename)):
            subprocess.call([
                "wget", url, "-P", "Annotations"
            ])
        subprocess.call([
            "unzip", "-n", filename
        ], cwd="Annotations")
    for url in questions_urls:
        filename = url.split("/")[-1]
        if not os.path.isfile("Questions/{}".format(filename)):
            subprocess.call([
                "wget", url, "-P", "Questions"
            ])
        subprocess.call([
            "unzip", "-n", filename
        ], cwd="Questions")
    for data_subtype, url in images_urls.items():
        images_folder = "Images/abstract_v002/{}".format(data_subtype)
        filename = url.split("/")[-1]
        if not os.path.isfile(images_folder+"/"+filename):
            subprocess.call([
               "wget", url, "-P", images_folder
            ])
        subprocess.call([
            "unzip", "-n", filename
        ], cwd=images_folder)
