import os, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageOps

class Preprocess:
        def __init__(self):
                self.target = ""
                self.images = ""
                self.flat_data = ""

        def resize_image_files(self, container_path):
                size = 200, 200
                image_dir = Path(container_path)
                folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]

                for i, direc in enumerate(folders):
                        for file in direc.iterdir():
                                im = Image.open(file)
                                im = ImageOps.fit(im, size, Image.ANTIALIAS)
                                im.save(str(file), "JPEG")

        def loadfile(self, container_path, dimension = (200, 200, 3)):
                image_dir = Path(container_path)
                folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
                categories = [fo.name for fo in folders]

                descr = "A image classification dataset"
                images = []
                flat_data = []
                target = []
                for i, direc in enumerate(folders):
                        for file in direc.iterdir():
                                img = imread(file, plugin='matplotlib')
                                img_resized = resize(img, dimension, anti_aliasing = True, mode = 'reflect')
                                flat_data.append(img_resized.flatten())
                                images.append(img_resized)
                                target.append(i)

                self.flat_data = np.array(flat_data)
                self.target = np.array(target)
                self.images = np.array(images)

                return Bunch(data = self.flat_data,
                                target = self.target,
                                target_names = categories,
                                images = self.images,
                                DESCR = descr)