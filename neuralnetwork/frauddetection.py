#####################################################
# IMPORTS
#####################################################
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pathlib

#####################################################
# DATA SETUP
#####################################################
data_dir = pathlib.Path("./dataset/")
image_paths = list(data_dir.glob('*/*.*'))

#####################################################
# IMPORT MODEL
#####################################################
model = VGG16(weights='imagenet', include_top=False)

#####################################################
# GET FEATURES FROM MODEL
#####################################################
features = []

for path in image_paths:
    print(path)
    img = image.load_img(path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features.append(model.predict(x))

print(len(features))

#####################################################
# PREDICTION
#####################################################

#
# ICI CHOISISSEZ VOTRE MODEL POUR FAIRE LA PREDICTION
#

#####################################################
# RÉSULTATS
#####################################################
