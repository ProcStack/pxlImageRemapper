import os
if __name__ == "__main__":
  # Lets not need to do this in command line, shall we?
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QListWidget, QHBoxLayout, QSizePolicy, QFileDialog

from source.VAE import VAE

from source.uiWidgets import HoverButtonWidget, SliderLabelWidget, ArrayEditWidget, StatusDisplay


class pxlImageRemapper(QMainWindow):
  def __init__(self, app, options={}, trainingData=None,vae=None,encoder=None, decoder=None, diffusionModel=None):
    super().__init__()
    self.name = "pxlImageRemapper"
    self.version = "0.0.1"
    self.sessionId = options['sessionId'] if 'sessionId' in options else "Default"

    self.app = app
    self.hasBreak = False
    self.settingsPath = options["settingsPath"] if "settingsPath" in options else os.path.join(options["outputFolder"], "settings.json")

    # Model Data
    self.trainingData = trainingData
    self.images = self.trainingData['images']
    self.labelLinks = self.trainingData['labelLinks']
    self.labelOptions = []
    self.labelOneHot = to_categorical(self.labelLinks)

    self.vae = vae
    self.encoder = encoder
    self.decoder = decoder
    self.diffusionModel = diffusionModel
    
    # Training Options
    self.inputTrainSize = options["inputTrainSize"] if "inputTrainSize" in options else 256
    self.epochs = options["epochs"] if "epochs" in options else 100
    self.batchSize = options["batch_size"] if "batch_size" in options else 32
    self.latentDim = options["latent_dim"] if "latent_dim" in options else 2
    self.encoderDecoderSizes = options["encoder_decoder_sizes"] if "encoder_decoder_sizes" in options else [64]
    self.reluLayers = options["relu_layers"] if "relu_layers" in options else [64]

    # Diffusion Options
    self.diffusionLayers = options["diffusion_layers"] if "diffusion_layers" in options else [64]
    self.generationEpochs = options["generation_epochs"] if "generation_epochs" in options else 10
    self.generationBatchSize = options["generation_batch_size"] if "generation_batch_size" in options else 4

    self.generations = {}

    # Output Paths
    defaultOutputFolder = "pirSession"
    if "outputFolder" not in options:
      scriptPath = os.path.abspath(__file__)
      rootFolder = os.path.dirname(scriptPath)
      outputFolder = os.path.join(rootFolder, "output" )
      if not os.path.exists(outputFolder):
          os.makedirs(outputFolder)
      options["outputFolder"] = defaultOutputFolder
      defaultOutputFolder = outputFolder
    self.outputFolder = options["outputFolder"] if "outputFolder" in options else defaultOutputFolder
    self.outLabels = options["outputLabels"] if "outputLabels" in options else os.path.join(defaultOutputFolder, "labels.json") 
    self.outputEncoder = options["outputEncoder"] if "outputEncoder" in options else os.path.join(defaultOutputFolder, "vae_encoder")
    self.outputDecoder = options["outputDecoder"] if "outputDecoder" in options else os.path.join(defaultOutputFolder, "vae_decoder")
    self.outputFileType = options["outputFileType"] if "outputFileType" in options else "keras"
    self.outputDiffusion = options["outputDiffusion"] if "outputDiffusion" in options else os.path.join(defaultOutputFolder, "diffusion_model." + self.outputFileType)
    self.outputSessionFolder = options["outputSessionFolder"] if "outputSessionFolder" in options else defaultOutputFolder

    # -- -- -- -- -- -- -- -- -- -- -- --

    # GUI variables --
    #   Load & Format Labels after setting output paths
    self.labelOptions = self.formatLabels(self.trainingData['labels'])
    self.markToClean = []
    self.statusText = None
    self.statusTimer = QTimer()

    self.initMenu()
    self.initUI()

  # -- -- -- -- -- -- -- -- -- -- -- -- --

  def initMenu(self):
    menubar = self.menuBar()
    fileMenu = menubar.addMenu('File')
    fileMenu.addAction('Load Session', self.loadSession)
    fileMenu.addAction('Save Session', self.saveSession)
    fileMenu.addAction('Save All', self.saveAll)
    fileMenu.addSeparator()
    fileMenu.addAction('Exit', self.close)
    fileMenu.addSeparator()

    settingsMenu = menubar.addMenu('Settings')
    settingsMenu.addAction('Save Settings', self.saveSettings)
    settingsMenu.addAction('Load Settings', self.loadSettings)
    settingsMenu.addSeparator()


  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def loadSession(self):
    folder = QFileDialog.getExistingDirectory(self, "Select Session Folder")
    if folder:
      # Load session data from the selected folder
      print("Loading Session VAE & Diffusion Model from : ")
      print("  --  ",folder)

  # TODO : Install my Settings Manager from pxlDataManager (ProcStack Private Repo)
  def saveSettings(self):
    settings = {}
    settings["epochs"] = self.epochs
    settings["batch_size"] = self.batchSize
    settings["latent_dim"] = self.latentDim
    settings["encoder_decoder_sizes"] = self.encoderDecoderSizes
    settings["relu_layers"] = self.reluLayers
    settings["diffusion_layers"] = self.diffusionLayers
    settings["generation_epochs"] = self.generationEpochs
    settings["generation_batch_size"] = self.generationBatchSize

    with open(self.settingsPath, 'w') as f:
      json.dump(settings, f)

  def loadSettings(self):
    if os.path.exists(self.settingsPath):
      with open(self.settingsPath, 'r') as f:
        settings = json.load(f)
        self.epochs = settings["epochs"]
        self.batchSize = settings["batch_size"]
        self.latentDim = settings["latent_dim"]
        self.encoderDecoderSizes = settings["encoder_decoder_sizes"]
        self.reluLayers = settings["relu_layers"]
        self.diffusionLayers = settings["diffusion_layers"]
        self.generationEpochs = settings["generation_epochs"]
        self.generationBatchSize = settings["generation_batch_size"]
    else:
      self.saveSettings()

  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def formatLabels(self, labels=None):
    ret = []
    if labels is not None:
      ret += labels
    
    if self.outLabels is not None and os.path.exists(self.outLabels):
      jsonLabels = self.loadJson( self.outLabels )
      if "labels" in jsonLabels:
        ret = jsonLabels["labels"]

    ret = sorted(list(set(ret)), key=lambda x: (len(x), x.lower(), x.islower(), x))
    ret = list(map(lambda x: x+"_"+("cap" if x==x.upper() else "low") if len(x)==1 else x, ret))
    return ret

  def windowTitle(self):
    return f"{self.name} :: v{self.version}"

  def initUI(self):
    self.setWindowTitle( self.windowTitle() )
    self.setGeometry(100, 100, 800, 600)

    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    layout = QVBoxLayout()
    layout.setSpacing(4)
    layout.setContentsMargins(3, 3, 3, 3)

    self.loadGeneratorData = HoverButtonWidget("Load Generator Data")
    self.loadGeneratorData.clicked.connect(self.loadVAEModelData)
    layout.addWidget(self.loadGeneratorData)

    # -- -- --
    
    sliderLayoutBlock = QHBoxLayout()
    sliderLayoutBlock.setSpacing(6)
    sliderLayoutBlock.setContentsMargins(2, 2, 2, 2)

    # Epoch Int Slider
    self.epoch_slider = SliderLabelWidget("Epochs", (1, 100), self.epochs)
    self.epoch_slider.subscribe(self.updateTrainingOptions)
    sliderLayoutBlock.addWidget(self.epoch_slider)

    # Batch Size Int Slider
    self.batch_slider = SliderLabelWidget("Batch Size", (4, 64), self.batchSize)
    self.batch_slider.subscribe(self.updateTrainingOptions)
    sliderLayoutBlock.addWidget(self.batch_slider)
    
    # Latent Dim Int Slider
    self.latent_slider = SliderLabelWidget("Latent Dim", (2, 24), self.latentDim)
    self.latent_slider.subscribe(self.updateTrainingOptions)
    sliderLayoutBlock.addWidget(self.latent_slider)

    layout.addLayout(sliderLayoutBlock)

    # -- -- --

    layerHLayout = QHBoxLayout()

    # Encoder Decoder Sizes String Int Array
    edSizesLayoutBlock = QVBoxLayout()
    curHeader = "Encoder Decoder Sizes"
    # -- -- --
    curHeaderLabel = QLabel(curHeader)
    curHeaderLabel.setStyleSheet("font-size: 16px;")
    curHeaderLabel.setFixedHeight(40)
    edSizesLayoutBlock.addWidget(curHeaderLabel)
    # -- -- --
    self.encoder_decoder_sizes = ArrayEditWidget( curHeader, self.encoderDecoderSizes )
    self.encoder_decoder_sizes.subscribe( self.updateTrainingOptions )
    edSizesLayoutBlock.addWidget(self.encoder_decoder_sizes)
    layerHLayout.addLayout(edSizesLayoutBlock)

    # ReLU Layers String Int Array
    reluLayoutBlock = QVBoxLayout()
    curHeader = "ReLU Layers"
    # -- -- --
    curHeaderLabel = QLabel(curHeader)
    curHeaderLabel.setStyleSheet("font-size: 16px;")
    curHeaderLabel.setFixedHeight(40)
    reluLayoutBlock.addWidget(curHeaderLabel)
    # -- -- --
    self.relu_layers = ArrayEditWidget( curHeader, self.reluLayers )
    self.relu_layers.subscribe( self.updateTrainingOptions )
    reluLayoutBlock.addWidget(self.relu_layers)
    layerHLayout.addLayout(reluLayoutBlock)

    # Diffusion Layers String Int Array
    diffusionLayoutBlock = QVBoxLayout()
    curHeader = "Diffusion Layers"
    # -- -- --
    curHeaderLabel = QLabel(curHeader)
    curHeaderLabel.setStyleSheet("font-size: 16px;")
    curHeaderLabel.setFixedHeight(40)
    diffusionLayoutBlock.addWidget(curHeaderLabel)
    # -- -- --
    self.diffusion_layers = ArrayEditWidget( curHeader, self.diffusionLayers )
    self.diffusion_layers.subscribe( self.updateTrainingOptions )
    diffusionLayoutBlock.addWidget(self.diffusion_layers)
    layerHLayout.addLayout(diffusionLayoutBlock)

    layout.addLayout(layerHLayout)

    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(15)
    layout.addWidget(spacer)

    # == == ==

    # Button Frame
    insetButtonBlock = QWidget()
    insetButtonLayout = QVBoxLayout()
    insetButtonLayout.setSpacing(0)
    insetButtonLayout.setContentsMargins(5, 5, 5, 5)
    insetButtonBlock.setLayout(insetButtonLayout)
    insetButtonBlock.setStyleSheet("background-color: #656565; border: 1px solid #555555; border-radius: 5px;")
    
    # -- -- --

    self.train_button = HoverButtonWidget("Train ...")
    self.train_button.clicked.connect(self.trainVAE)
    insetButtonLayout.addWidget(self.train_button)
    
    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(10)
    spacer.setStyleSheet("border: 0px;")
    insetButtonLayout.addWidget(spacer)
    
    # -- -- --

    self.saveSession_button = HoverButtonWidget("Save VAE & Diffusion Session")
    self.saveSession_button.clicked.connect(self.saveSession)
    insetButtonLayout.addWidget(self.saveSession_button)

    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(10)
    spacer.setStyleSheet("border: 0px;")
    insetButtonLayout.addWidget(spacer)

    # -- -- --

    self.saveVAE_button = HoverButtonWidget("Save VAE Encodings & Decodings")
    self.saveVAE_button.clicked.connect(self.saveVAEClicked)
    insetButtonLayout.addWidget(self.saveVAE_button)

    # -- -- --

    self.saveDiffusion_button = HoverButtonWidget("Save Diffusion Model")
    self.saveDiffusion_button.clicked.connect(self.saveDiffusionClicked)
    insetButtonLayout.addWidget(self.saveDiffusion_button)

    # -- -- --

    layout.addWidget(insetButtonBlock)

    # == == ==

    spacer = QLabel("")
    spacer.setFixedHeight(15)
    layout.addWidget(spacer)


    # -- -- --

    self.labelCombo = QComboBox()
    self.labelCombo.setStyleSheet("font-size: 20px; font-weight: bold; background-color: #353535; color: white; border: 1px solid #808080; border-radius: 5px;")
    if self.labelOptions is not None and len(self.labelOptions) > 0:
      self.labelCombo.addItems(self.labelOptions)
    layout.addWidget(self.labelCombo)

    # -- -- --
    
    spacer = QLabel("")
    spacer.setFixedHeight(10)
    layout.addWidget(spacer)

    # -- -- --

    # Generation batch size
    self.genEpochsSlider = SliderLabelWidget("Gen Epochs", (0, 10), self.generationEpochs, 150)
    self.genEpochsSlider.subscribe(self.updateGenerationOptions)
    layout.addWidget(self.genEpochsSlider)

    # Generation batch size
    self.getBatchSizeSlider = SliderLabelWidget("Gen Batch Size", (0, 10), self.generationBatchSize, 150)
    self.getBatchSizeSlider.subscribe(self.updateGenerationOptions)
    layout.addWidget(self.getBatchSizeSlider)


    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(4)
    layout.addWidget(spacer)

    # -- -- --

    self.generateButton = HoverButtonWidget("Generate")
    self.generateButton.clicked.connect(self.generateImages)
    layout.addWidget(self.generateButton)

    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(4)
    layout.addWidget(spacer)
    
    # -- -- --
    self.imageListWidget = QListWidget()
    self.imageListWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    layout.addWidget(self.imageListWidget)

    # -- -- --

    self.statusText = QLabel(self)
    layout.addWidget(self.statusText)
    self.setStatusText("Init...")


    central_widget.setLayout(layout)

  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  # Break out of the current training or generation loop
  def checkBreak(self):
    if self.app is not None:
      # TODO : Move traing to threading for cpu and what ever for gpu
      self.app.processEvents()
    return self.hasBreak
  
  def keyPressEvent(self, event):
    if event.key() == Qt.Key.Key_Escape:
      self.hasBreak = True

  def keyReleaseEvent(self, event):
    if event.key() == Qt.Key.Key_Escape:
      self.hasBreak = False

  # -- -- --

  def UpdateDisplays(self):
    print("Updating Displays...")

  # -- -- --

  def initHelpers(self):
      self.statusTimer.timeout.connect(self.clearStatusText)

  def setNoTimerStatusText(self, text):
    self.setStatusText(text, 0)

  def setStatusText(self, text, clearDelay=5000):
      if self.statusText:
          self.statusText.setText(text)
          baseLine = "font-size:20px;"
          if "Error" in text:
              self.statusText.setStyleSheet(baseLine+"background-color: #551515;font-weight: bold;")
          elif "Warning" in text:
              self.statusText.setStyleSheet(baseLine+"background-color: #454515;font-weight: bold;")
          elif "Tip" in text:
              self.statusText.setStyleSheet(baseLine+"background-color: #153555;font-weight: bold;")
          else:
              self.statusText.setStyleSheet(baseLine+"color: #cccccc;")
      
      if text != "" and clearDelay > 0:
          self.statusText.setVisible(True)
          self.statusTimer.timeout.connect(lambda: self.setStatusText(""))
          self.statusTimer.start(clearDelay)
      elif text != "" and clearDelay == 0 :
          self.statusText.setVisible(True)
          self.statusTimer.stop()
      else:
          self.statusText.setVisible(False)
          self.statusTimer.stop()

      if self.app is not None and self.app.instance():
          self.app.processEvents()

  def clearStatusText(self):
      self.setStatusText("")

  def setDelayStatusText(self, text, delay=5000, duration=8000):
      self.statusTimer.timeout.connect(lambda: self.setStatusText(text, duration))
      self.statusTimer.start(delay)
  
  def stopDelayStatusText(self):
      self.statusTimer.stop()

  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def stringToArray(self, stringVal):
    ret = list(filter(lambda x: x.isnumeric(), str(stringVal).split(" ")))
    return list(map(lambda x: int(x), ret))

  def updateTrainingOptions(self, label, value):
    if label == "Epochs":
      self.epochs = value
      return;
    elif label == "Batch Size":
      self.batchSize = value
      return;
    elif label == "Latent Dim":
      self.latentDim = value
      return;
    elif label == "Encoder Decoder Sizes":
      self.encoderDecoderSizes = value
      return;
    elif label == "ReLU Layers":
      print("Updating ReLU Layers : ", value)
      self.reluLayers = value
      return;
    elif label == "Diffusion Layers":
      self.diffusionLayers = value
      return;
    else:
      print("Error: Unknown label ; ", label)
      return;


  def updateGenerationOptions(self, label, value):
    if label == "Gen Batch Size":
      self.batchSize = value

  def trainVAE(self):
    if self.vae is None:
      self.loadVAE()
    print("Training VAE...")
    self.runTrainGenerationStack()

  def saveVAE(self, isSession=False):
    if self.vae is None:
      self.loadVAE()
    if isSession:
      self.vae.saveSession()
    else:
      self.vae.save()

  def saveDiffusion(self, isSession=False):
    if self.diffusionModel is None:
      self.loadModel()
    if isSession:
      self.diffusionModel.save( os.path.join(self.outputSessionFolder, "diffusion_model"+str(self.sessionId)+"." + self.outputFileType) )
    else:
      self.diffusionModel.save( self.outputDiffusion )

  def saveAll(self):
    self.saveVAE()
    self.saveVAE( True )
    self.saveDiffusion()
    self.saveDiffusion( True )
    self.setStatusText("-- All Models Saved --")

  def saveVAEClicked(self):
    self.saveVAE()
    self.saveVAE( True )
    self.setStatusText("-- VAE Encodings & Decodings Saved --")

  def saveDiffusionClicked(self):
    self.saveDiffusion()
    self.saveDiffusion( True )
    self.setStatusText("-- Diffusion Model Saved --")

  def saveSession(self):
    if self.vae is None:
      self.loadVAE()
    self.vae.saveSession()
    if not os.path.exists(self.outputSessionFolder):
      os.makedirs(self.outputSessionFolder)
    if self.diffusionModel is None:
      self.loadModel()
    self.diffusionModel.save( os.path.join(self.outputSessionFolder, "diffusion_model"+str(self.sessionId)+"." + self.outputFileType) )
    self.setStatusText("-- VAE & Diffusion Session Saved --")


  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def loadVAEModelData(self):
    if self.trainingData is not None:
      if self.labelLinks is None:
        self.labelLinks = self.trainingData["labelLinks"]

    if self.labelOneHot is None:
      self.labelOneHot = to_categorical(self.labelLinks)
    if self.vae is None:
      self.loadVAE()
    if self.diffusionModel is None:
      self.loadModel()

  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def loadJson(self, jsonPath):
    with open(jsonPath, 'r') as f:
      return json.load(f)
    
  def loadVAE(self):
    #def __init__(self, parent, sessionId, labels, epochs, batch_size, outputEncoderPath, outputDecoderPath, outputSessionPath, outputType, latent_dim=2, encoderDecoderSizes=[], reluLayers=[], **kwargs):
   
    self.vae = VAE( 
                    parent=self,
                    sessionId=self.sessionId, 
                    labels=self.labelLinks,
                    epochs=self.epochs,
                    batch_size=self.batchSize,
                    outputFolder=self.outputFolder,
                    outputEncoderPath=self.outputEncoder,
                    outputDecoderPath=self.outputDecoder,
                    outputSessionPath=self.outputSessionFolder,
                    outputType=self.outputFileType,
                    latent_dim=self.latentDim,
                    encoderDecoderSizes=self.encoderDecoderSizes,
                    reluLayers=self.reluLayers
                  )

    self.vae.load()

    self.setStatusText("-- VAE Encodings & Decodings Loaded --")

  def loadModel(self):
    if os.path.exists(self.outputDiffusion):
      self.diffusionModel = models.load_model( self.outputDiffusion, custom_objects={"mse": MeanSquaredError()} )
    else:
      self.diffusionModel = self.buildDiffusionModel(self.latentDim)
      self.diffusionModel.save( self.outputDiffusion )

    # Train Compiling of the Model and fitting the data occurs only when the training occurs
    #   This is to allow easier instant use of the existing model

    self.diffusionModel.summary()
    self.setStatusText("-- Diffusion Model Loaded --")

  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def findStartingCount(self, folder, basename=None, ext=None):
    curImageCount = os.listdir(folder)
    if ext is not None:
      curImageCount = len( list(filter(lambda x: x.endswith(ext), curImageCount)) )
    if basename is not None:
      curImageCount += len( list(filter(lambda x: x.startswith(basename), curImageCount)) )
    if ext is not None and basename is not None:
      curImageCount = sorted( list(set( lambda x: int(x.split("_")[-1].split(".")[0]), curImageCount )) )
      curImageCount = curImageCount[-1]+1
    else:
      curImageCount = int(curImageCount.split("_")[-1].split(".")[0])+1
    return curImageCount

  def generateImages(self):
    selected_label = self.labelCombo.currentText()

    # Generate images using the selected label and slider values
    new_images = self.GenerateSpecificCharacter(selected_label, self.vae, self.diffusion_model, output_folder=self.outputSessionFolder)

    # Display the generated images in the list
    self.imageListWidget.clear()
    curImageCount = self.findStartingCount(self.outputSessionFolder, ext=".png")

    for img in new_images:
      img = (img * 255).astype(np.uint8)
      img = Image.fromarray(img.squeeze(), "L")
      padCount = str(curImageCount).zfill(3)
      img_path = os.path.join(self.outputSessionFolder, f"{selected_label}_{padCount}.png")
      img.save(img_path)
      self.imageListWidget.addItem(img_path)

      curImageCount += 1


  def fileExists(self, filePath):
    return os.path.exists(filePath)

  def calcZValues(self, encoder):
    # Encode the images using the VAE encoder

    concatZValues = []
    for x in range(len(self.encoderDecoderSizes)):
      size = self.encoderDecoderSizes[x]
      encoder,decoder = self.vae.getEncoder(size)
      z_mean, z_log_var = encoder.predict(self.images[size])
      z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=tf.shape(z_mean))
      concatZValues.append(z)

    # Combine latent vectors from different scales
    z_combined = np.concatenate(concatZValues, axis=0)

    return z_combined

  def GenerateRemapper(self,  outputSize, diffusion_model, output_folder="", runDiffusionFit=False ):
    # Load the VAE encoder and decoder
    encoder,decoder = self.vae.getEncoder(outputSize)
    if type(diffusion_model) == str:
      diffusion_model = models.load_model(diffusion_model, custom_objects={"mse": MeanSquaredError()})

    # Generate new latent vectors
    z_combined = self.calcZValues(encoder)
    # Compile and train the diffusion model
    if runDiffusionFit:
      diffusion_model.compile(optimizer="adam", loss="mse")
      diffusion_model.fit(z_combined, z_combined, epochs=self.epochs, batch_size=self.batchSize)

    new_latent_vectors = diffusion_model.predict(z_combined)

    outputSize = self.vae.checkOutputSize(outputSize)
    print("Output Size : ", outputSize)
    if decoder is None:
      encoder,decoder = self.vae.getEncoder(outputSize)
    
    if decoder is None:
      print(self.vae.decoders.keys())
      print("Error: No decoder found")
      return;

    # Decode the new latent vectors to generate new images
    print("Generating new images...")
    new_images = decoder.predict(new_latent_vectors)
    print("New images generated - ", len(new_images))
    # Display the generated images
    for x in range(10):  # Display 10 generated images
        dispNum = len(new_images)-1-x
        plt.imshow(new_images[dispNum], cmap='gray')
        plt.show()

    # Save the generated images
    if output_folder != "":
      for x, img in enumerate(new_images):
        img = (img * 255).astype(np.uint8)
        if not os.path.exists(output_folder):
          os.makedirs(output_folder)
        img = Image.fromarray(img)
        outputPath = os.path.join(output_folder, f'pxlImageRemapper_{x}.png')
        img.save(outputPath)

    # Calculate reconstruction loss
    mse = MeanSquaredError()
    imagesCount = self.images[outputSize].shape[0]
    displayImages = new_images[:imagesCount]
    original_size = tf.shape(self.images[outputSize])[1:3]
    displayImages = tf.image.resize(displayImages, original_size)
    reconstruction_loss = mse(self.images[outputSize], displayImages).numpy()
    print(f'Reconstruction Loss: {reconstruction_loss}')

    return new_images

  # -- -- -- -- -- -- -- --

  def GenerateSpecificCharacter(self, character, diffusion_model, output_folder=""):
    # Filter images and labels for the requested character
    filtered_images = {}
    for size in self.images.keys():
      filtered_images[size] = [img for img, label in zip(self.images[size], self.labelLinks) if label == character]
      if len(filtered_images[size]) == 0:
        print(f"No images found for character: {character}")
        return

    # Convert lists to numpy arrays
    for size in filtered_images.keys():
      filtered_images[size] = np.array(filtered_images[size])
      filtered_images[size] = filtered_images[size].reshape(-1, size, size, 1)

    # Generate new images using the filtered images
    new_images = self.GenerateRemapper(self.inputTrainSize, diffusion_model, output_folder, runDiffusionFit=False)

    # Display the generated images
    for x in range(10):  # Display 10 generated images
      if self.checkBreak():
        print("Exiting Generation...")
        break
      dispNum = len(new_images) - 1 - x
      plt.imshow(new_images[dispNum], cmap="gray")
      plt.show()

    # Save the generated images
    if output_folder != "":
      startingCount = self.findStartingCount(output_folder, character, ".png")
      for x, img in enumerate(new_images):
        if self.checkBreak():
          print("Exiting Generation...")
          break
        img = (img * 255).astype(np.uint8)
        if not os.path.exists(output_folder):
          os.makedirs(output_folder)
        img = Image.fromarray(img)
        curCount = str(startingCount+x).zfill(3)
        outputPath = os.path.join(output_folder, f"pxlImageRemapper_{character}_{curCount}.png")
        img.save(outputPath)

    return new_images

  def buildDiffusionModel(self, latent_dim):
      maxSize = self.vae.checkOutputSize(self.inputTrainSize)
      layerStack = []
      layerStack.append( layers.Input(shape=(latent_dim,)) )
      for i in range(len(self.diffusionLayers)):
        layerStack.append( layers.Dense(self.diffusionLayers[i], activation="relu") )
      layerStack.append( layers.Dense(latent_dim) )
      
      model = models.Sequential( layerStack )
      model.compile(optimizer="adam", loss="mse")
      model.summary()
      return model
  
  # -- -- -- -- -- -- -- --

  def generateImages(self, res=128):
    print(f"Entering Prediction Mode...")

    #if not fileExists(output_encoder_128) or not fileExists(output_decoder_128) or not fileExists(self.outputDiffusion):
    if not self.fileExists( self.outputDiffusion ):
      print("Error: Missing models")
      print("   Exiting ...")
      return;

    # Load the diffusion model with custom objects
    self.vae.hasLoaded()

    diffusion_model = models.load_model( self.outputDiffusion, custom_objects={"mse": MeanSquaredError()})
    #GenerateRemapper( self.inputTrainSize, diffusion_model, output_folder="" )
    
    self.GenerateSpecificCharacter("A_cap", diffusion_model, output_folder="")

    self.UpdateDisplays()


  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def runTrainGenerationStack(self):

    step=0

    step+=1
    self.setStatusText(f"{step} - Building Encoder & Decoder...")

    if self.fileExists( self.outputEncoder ) and self.fileExists( self.outputDecoder ):
      self.setStatusText("Loading existing models...")
      self.vae.load()
    else:  # Instantiate the shared encoder, decoder, and VAE
      self.setStatusText("Building new models...")
      self.vae.prepShapes()

    self.vae.train( self.images, self.labelLinks, self.epochs, self.batchSize )

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    if self.checkBreak():
      self.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    step+=1
    self.setStatusText(f"{step} - Building Diffusion model...")

    # Define the diffusion model

    diffusion_model = self.buildDiffusionModel(self.latentDim)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    if self.checkBreak():
      self.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    step+=1

    concatZValues = []
    for size in self.encoderDecoderSizes:
      encoder,decoder = self.vae.getEncoder(self.images[size][0])
      z_mean, z_log_var = encoder.predict(self.images[size])
      z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=tf.shape(z_mean))
      concatZValues.append(z)

    # Combine latent vectors from different scales
    z_combined = np.concatenate(concatZValues, axis=0)

    # Compile and train the diffusion model
    diffusion_model.compile(optimizer="adam", loss="mse")
    diffusion_model.fit(z_combined, z_combined, epochs=self.epochs, batch_size=32)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    if self.checkBreak():
      self.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    step+=1
    self.setStatusText(f"{step} - Saving VAE Encoder/Decoder & Model...")

    # Save the VAE encoder and decoder
    self.vae.save()


    # Save the diffusion model
    print(self.outputDiffusion)
    diffusion_model.save( self.outputDiffusion )
    diffusion_model.save( self.outputSessionFolder + "/diffusion_model." + self.outputFileType )

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    if self.checkBreak():
      self.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    step+=1
    print(f"{step} - Generating new images...")

    # Load the VAE encoder and decoder
    #GenerateRemapper( self.inputTrainSize, diffusion_model, output_folder )

    self.GenerateSpecificCharacter("A_cap", diffusion_model, self.outputSessionFolder)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    if self.checkBreak():
      self.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

