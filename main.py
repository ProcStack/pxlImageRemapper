#############################################################
#   pxlImageRemapper :: 0.0.1                               #
#     Kevin Edzenga (ProcStack); Oct. 2024                  #
#       (Copilot sucks for citing sources......)            #
#  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --       #
#    Primary window & control for the pxlImageRemapper      #
#      The window interacts with ./source/VAE.py            #
#  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --       #
#                                                           #
#   Diffusion models exist... Why is this a thing?          #
#     More of a learning experience than anything else.     #
#                                                           #
#   Visualize how ai's learn latent space encodings and     #
#     can be used to generate your own ai images.           #
#   With custom levels of training and generation,          #
#     you can create your own unique images based on        #
#     the specfic labels your model is trained on.          #
#                                                           #
#   Should you have a collection of images you want         #
#     to create a custom stand-alone model for,             #
#       this is a good place to start for that!             #
#   If not, I provided a sample set of hand written         #
#     letters to get you started.                           #
#                                                           #
#############################################################



import os
if __name__ == "__main__":
  # Lets not need to do this in command line, shall we?
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'


import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import json
from PyQt6.QtWidgets import QApplication

from source.pxlImageRemapper import pxlImageRemapper

options={}
options["outputFolderBase"] = 'pirSession'
options["showTrainingOnly"]=False
options["generateOnly"]=False
options["writePredictedImages"]=False
options["continueTrainingIfExist"]=True

# -- -- -- -- -- -- -- --

options["encoder_decoder_sizes"]=[ 64 ]
options["relu_layers"]=[ 32, 64 ]
options["diffusion_layers"]=[ 32, 64 ]

options["latent_dim"] = 10
options["epochs"] = 10
options["batch_size"] = 32

options["generation_epochs"] = 1
options["generation_batch_size"] = 4


# -- -- -- -- -- -- -- --

settingsFileKeys = list(options.keys())

# -- -- -- -- -- -- -- --

step=0
print(f"Initializing pxlImageRemapper...")

# Define the path to the input images
script_path = os.path.abspath(__file__)
script_baseName = os.path.basename(__file__).split(".")[0]
root_folder = os.path.dirname(script_path)
input_folder = os.path.join(root_folder, "input" )

output_folder = os.path.join(root_folder, "output" )
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

options["outputFileType"] = "keras"
options["outputFolder"] = output_folder
options["outputLabels"] = os.path.join(options["outputFolder"], "labels.json")
options["outputEncoder"] = os.path.join(options["outputFolder"], "vae_encoder")
options["outputDecoder"] = os.path.join(options["outputFolder"], "vae_decoder")
options["outputDiffusion"] = os.path.join(options["outputFolder"], "diffusion_model." + options["outputFileType"])

options["settingsPath"] = os.path.join(options["outputFolder"], "settings.json")

# Find the latest session folder
#   Think of sessions as the current state of the vae (en|de)codings and diffusion model
#     Checkpoints for results and models
output_folder_sessions = []
ofsNumberList = []
outSessionValue = 0
output_session_folder = os.path.join(output_folder, options["outputFolderBase"] + "_0")
for filename in os.listdir(output_folder):
  if options["outputFolderBase"] in filename:
    output_folder_sessions.append(filename)
    ofsNumberList.append(int(filename.split("_")[-1]))
if len(output_folder_sessions) > 0:
  output_folder_sessions = [x for _, x in sorted(zip(ofsNumberList, output_folder_sessions))]
  outSessionValue = int(output_folder_sessions[-1].split("_")[-1]) + 1
  output_session_folder = os.path.join(output_folder, options["outputFolderBase"] + "_" + str(outSessionValue))

options["sessionId"] = outSessionValue
options["outputSessionFolder"] = output_session_folder



# Load settings from file
#   Mostly doing this here so new initial settings can be added to the saved settings
#     Then figure out whats what on pxlImageRemapper

if os.path.exists(options["settingsPath"]):
  with open(options["settingsPath"], 'r') as f:
    settings = json.load(f)
    for key in settings.keys():
      if key in settingsFileKeys:
        options[key] = settings[key]



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- Parse Training Data  -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# Initialize lists to hold images and labels
images = {}
labels = []
labelLinks = []
inputTrainSize = 256

step+=1
print(f"{step} - Finding Source Data...")

# Load images and labels
edSizesReverse = options["encoder_decoder_sizes"].copy()
edSizesReverse.sort()
edSizesReverse.reverse()
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Assuming the images are in PNG format
        # Load the image
        img = Image.open(os.path.join(input_folder, filename)).convert("L")  # Convert to grayscale
        inputTrainSize = max( inputTrainSize, img.size[0] )

        for size in edSizesReverse:
          curImg = img.resize((size, size))
          imgArray = np.array(curImg) / 255.0
          if size not in images.keys():
            images[size] = []
          label = filename.split("_")
          if len(label) == 3:
            label = label[1]
          elif len(label) > 3:
            label = label[1:2]
            label = "_".join(label)
          images[size].append((imgArray, label))  # Store image and label as a tuple
          labels.append(label)

        
# Display the first image
if options["showTrainingOnly"]:
  firstKey = list(images.keys())[0]
  for x in range(10):
    plt.imshow(images[firstKey][x][0], cmap="gray")
    plt.show()
  exit()

# Convert lists to numpy arrays
"""for size in encoder_decoder_sizes:
  images[size] = np.array(images[size])
  images[size] = images[size].reshape(-1, size, size, 1)
labels = np.array(labels)"""
for size in images.keys():
  images[size] = np.array([img[0] for img in images[size]])  # Extract images
  labelLinks = np.array([img[1] for img in images[size]])  # Extract labels
  images[size] = images[size].reshape(-1, size, size, 1)

options["inputTrainSize"] = inputTrainSize

labelLinks = labelLinks.flatten()  # Ensure labels is a 1D array


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

if __name__ == "__main__":
  app = QApplication(sys.argv)

  trainingData = {}
  trainingData["images"] = images
  trainingData["labelLinks"] = labelLinks
  trainingData["labels"] = labels

  window = pxlImageRemapper(
              app,
              options,
              trainingData=trainingData
            )
  window.show()
  sys.exit(app.exec())












