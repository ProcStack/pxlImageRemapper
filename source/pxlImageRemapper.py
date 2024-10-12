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
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QListWidget, QHBoxLayout, QSizePolicy, QFileDialog, QListWidgetItem

from source.VAE import VAE

from source.uiWidgetsSource.HoverButton import HoverButtonWidget
from source.uiWidgetsSource.SliderLabel import SliderLabelWidget
from source.uiWidgetsSource.ArrayEdit import ArrayEditWidget
from source.uiWidgetsSource.StatusDisplay import StatusDisplay
from source.uiWidgetsSource.ImageDataDisplay import ImageDataDisplayWidget
from source.uiWidgetsSource.GridArrayDisplay import GridArrayDisplayWidget



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
    self.images = self.trainingData['images'] if 'images' in self.trainingData else None
    self.labels = self.trainingData['labels'] if 'labels' in self.trainingData else None
    self.labelLinks = self.trainingData['labelLinks'] if 'labelLinks' in self.trainingData else None
    self.labelOptions = []
    self.labelOneHot = to_categorical(self.labelLinks)

    self.vae = vae
    self.encoder = encoder
    self.decoder = decoder
    self.diffusionModel = diffusionModel
    self.needsFit = False
    
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

    self.autoSave = options["autoSave"] if "autoSave" in options else False
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
    self.displayGridRes = (3,5)

    # Used for setting the Status Text's Cancel button callback mode
    #   When the status bar is visible, the cancel button will trigger all callback[mode] functions
    self.trainingMode = {
        "VAE":"Training VAE",
        "Diffusion":"Training Diffusion Model"
      }

    self.initMenu()
    self.initUI()
    self.initHelpers()

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

  def initHelpers(self):
    if self.statusText is not None:
      self.statusText.subscribeToCancel( self.triggerBreak, self.trainingMode["VAE"] )

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
    self.setGeometry(100, 100, 1000, 700)
    self.setStyleSheet("background-color: #252525;")

    central_widget = QWidget()
    self.setCentralWidget(central_widget)



    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # Main Layout & Staging Layouts

    layout_main = QVBoxLayout()
    layout_main.setSpacing(2)
    layout_main.setContentsMargins(3, 3, 3, 3)
    
    layout_sideGenBar = QHBoxLayout()

    widget_sideBar = QWidget()
    widget_sideBar.setFixedWidth(600)
    layout_sideBar = QVBoxLayout()
    layout_sideBar.setSpacing(2)
    layout_sideBar.setContentsMargins(3, 3, 3, 3)
    widget_sideBar.setLayout(layout_sideBar)
    layout_sideGenBar.addWidget(widget_sideBar)

    layout_generationStage = QVBoxLayout()
    layout_generationStage.setSpacing(0)
    layout_generationStage.setContentsMargins(2, 2, 2, 2)
    layout_sideGenBar.addLayout(layout_generationStage)

    layout_main.addLayout(layout_sideGenBar)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # Side Bar

    self.loadGeneratorData = HoverButtonWidget("Load Generator Data", color="INFO")
    self.loadGeneratorData.clicked.connect(self.loadVAEModelData)
    layout_sideBar.addWidget(self.loadGeneratorData)

    # -- -- --
    
    trainingOptionsBlock = QWidget()
    trainingOptionsBlock.setStyleSheet("background-color: #353535; border: 1px solid #555555; border-radius: 5px;")
    trainingOptionsBlockLayout = QVBoxLayout()
    trainingOptionsBlockLayout.setSpacing(0)
    trainingOptionsBlockLayout.setContentsMargins(3, 3, 3, 3)
    trainingOptionsBlock.setLayout(trainingOptionsBlockLayout)
    trainingOptionsStyleCancel = QWidget()
    trainingOptionsStyleCancel.setStyleSheet("background-color: #353535; border: 0px;")
    trainingOptionsLayout = QVBoxLayout()
    trainingOptionsLayout.setSpacing(0)
    trainingOptionsLayout.setContentsMargins(0, 0, 0, 0)
    trainingOptionsStyleCancel.setLayout(trainingOptionsLayout)
    trainingOptionsBlockLayout.addWidget(trainingOptionsStyleCancel)

    trainingHeader = QLabel(":: Training Options ::")
    trainingHeader.setStyleSheet("font-size: 20px; font-weight: bold;")
    trainingHeader.setAlignment(Qt.AlignmentFlag.AlignCenter)
    trainingOptionsLayout.addWidget(trainingHeader)

    # -- -- --

    sliderLayoutBlock = QVBoxLayout()
    sliderLayoutBlock.setSpacing(3)
    sliderLayoutBlock.setContentsMargins(3, 3, 3, 3)


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

    trainingOptionsLayout.addLayout(sliderLayoutBlock)

    # -- -- --

    layerHLayout = QHBoxLayout()

    # Encoder Decoder Sizes String Int Array
    edSizesLayoutBlock = QVBoxLayout()
    curHeader = "VAE Sizes"
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

    trainingOptionsLayout.addLayout(layerHLayout)

    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(15)
    trainingOptionsLayout.addWidget(spacer)

    # == == ==

    # Button Frame
    insetButtonBlock = QWidget()
    insetButtonLayout = QVBoxLayout()
    insetButtonLayout.setSpacing(0)
    insetButtonLayout.setContentsMargins(5, 5, 5, 5)
    insetButtonBlock.setLayout(insetButtonLayout)
    insetButtonBlock.setStyleSheet("background-color: #656565; border: 1px solid #555555; border-radius: 5px;")
    
    # -- -- --

    trainButtonLayout = QHBoxLayout()

    self.train_button = HoverButtonWidget("Train VAE ...", color="ACCEPT")
    self.train_button.clicked.connect(self.trainVAE)
    trainButtonLayout.addWidget(self.train_button)

    self.trainDiffusion_button = HoverButtonWidget("Train Diffusion ...", color="ACCEPT")
    self.trainDiffusion_button.clicked.connect(self.trainDiffusion)
    trainButtonLayout.addWidget(self.trainDiffusion_button)

    insetButtonLayout.addLayout(trainButtonLayout)
    
    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(10)
    spacer.setStyleSheet("border: 0px;")
    insetButtonLayout.addWidget(spacer)
    
    # -- -- --

    self.saveSession_button = HoverButtonWidget("Save Session Data", color="INFO")
    self.saveSession_button.clicked.connect(self.saveSession)
    insetButtonLayout.addWidget(self.saveSession_button)

    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(10)
    spacer.setStyleSheet("border: 0px;")
    insetButtonLayout.addWidget(spacer)

    # -- -- --

    saveButtonLayout = QHBoxLayout()

    self.saveVAE_button = HoverButtonWidget("Save VAE", color="INFO")
    self.saveVAE_button.clicked.connect(self.saveVAEClicked)
    saveButtonLayout.addWidget(self.saveVAE_button)

    self.saveDiffusion_button = HoverButtonWidget("Save Diffusion Model", color="INFO")
    self.saveDiffusion_button.clicked.connect(self.saveDiffusionClicked)
    saveButtonLayout.addWidget(self.saveDiffusion_button)
    insetButtonLayout.addLayout(saveButtonLayout)

    # -- -- --

    trainingOptionsLayout.addWidget(insetButtonBlock)
    layout_sideBar.addWidget(trainingOptionsBlock)

    # == == ==

    spacer = QLabel("")
    spacer.setFixedWidth(10)
    spacer.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    layout_sideBar.addWidget(spacer)


    # -- -- --
    
    generationOptionsBlock = QWidget()
    generationOptionsBlock.setStyleSheet("background-color: #353535; border: 1px solid #555555; border-radius: 5px;")
    generationOptionsBlockLayout = QVBoxLayout()
    generationOptionsBlockLayout.setSpacing(0)
    generationOptionsBlockLayout.setContentsMargins(3, 3, 3, 3)
    generationOptionsBlock.setLayout(generationOptionsBlockLayout)
    generationOptionsStyleCancel = QWidget()
    generationOptionsStyleCancel.setStyleSheet("background-color: #353535; border: 0px;")
    generationOptionsLayout = QVBoxLayout()
    generationOptionsLayout.setSpacing(0)
    generationOptionsLayout.setContentsMargins(0, 0, 0, 0)
    generationOptionsStyleCancel.setLayout(generationOptionsLayout)
    generationOptionsBlockLayout.addWidget(generationOptionsStyleCancel)
    layout_sideBar.addWidget(generationOptionsBlock)

    
    # -- -- --

    generationHeader = QLabel(":: Generation Options ::")
    generationHeader.setStyleSheet("font-size: 20px; font-weight: bold;")
    generationHeader.setAlignment(Qt.AlignmentFlag.AlignCenter)
    generationOptionsLayout.addWidget(generationHeader)

    # -- -- --
    
    genLabel = QLabel("Generate Label : ")
    genLabel.setStyleSheet("font-size: 16px;")
    generationOptionsLayout.addWidget(genLabel)

    self.labelCombo = QComboBox()
    self.labelCombo.setStyleSheet("font-size: 20px; font-weight: bold; background-color: #353535; color: white; border: 1px solid #808080; border-radius: 5px;")
    if self.labelOptions is not None and len(self.labelOptions) > 0:
      self.labelCombo.addItems(self.labelOptions)
    generationOptionsLayout.addWidget(self.labelCombo)

    # -- -- --
    
    spacer = QLabel("")
    spacer.setFixedHeight(10)
    generationOptionsLayout.addWidget(spacer)

    # -- -- --

    # Generation batch size
    self.genEpochsSlider = SliderLabelWidget("Gen Epochs", (0, 10), self.generationEpochs, 150)
    self.genEpochsSlider.subscribe(self.updateGenerationOptions)
    generationOptionsLayout.addWidget(self.genEpochsSlider)

    # Generation batch size
    self.getBatchSizeSlider = SliderLabelWidget("Gen Batch Size", (0, 10), self.generationBatchSize, 150)
    self.getBatchSizeSlider.subscribe(self.updateGenerationOptions)
    generationOptionsLayout.addWidget(self.getBatchSizeSlider)


    # -- -- --

    spacer = QLabel("")
    spacer.setFixedHeight(4)
    generationOptionsLayout.addWidget(spacer)

    # -- -- --

    self.generateButton = HoverButtonWidget("Generate Images", color="ACCEPT")
    self.generateButton.clicked.connect(self.generateImages)
    generationOptionsLayout.addWidget(self.generateButton)

    
    # -- -- --

    stageParentBlock = QWidget()
    stageParentBlock.setStyleSheet("background-color: #454545; border: 1px solid #555555; border-radius: 5px;")
    stageParentBlockLayout = QVBoxLayout()
    stageParentBlockLayout.setSpacing(0)
    stageParentBlockLayout.setContentsMargins(3, 3, 3, 3)
    stageParentBlock.setLayout(stageParentBlockLayout)

    self.imageListWidget = GridArrayDisplayWidget( gridRes=self.displayGridRes )
    self.imageListWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    self.imageListWidget.setStyleSheet("background-color: #454545; border: 0px;")
    stageParentBlockLayout.addWidget(self.imageListWidget)
    layout_generationStage.addWidget(stageParentBlock)

    # -- -- --

    self.statusText = StatusDisplay( self.app )
    layout_main.addWidget(self.statusText)
    self.statusText.setStatusText("Init...")


    central_widget.setLayout(layout_main)
    central_widget.setStyleSheet("background-color: #353535;")
    central_widget.setContentsMargins(0, 0, 0, 0)

  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  # Break out of the current training or generation loop
  def checkBreak(self):
    if self.app is not None:
      # TODO : Move traing to threading for cpu and what ever for gpu
      self.app.processEvents()
    return self.hasBreak
  
  def resetBreak(self):
    self.hasBreak = False

  def triggerBreak(self):
    self.hasBreak = True

  def keyPressEvent(self, event):
    if event.key() == Qt.Key.Key_Escape:
      self.hasBreak = True

  def keyReleaseEvent(self, event):
    if event.key() == Qt.Key.Key_Escape:
      self.hasBreak = False

  # -- -- --

  def UpdateDisplays(self, imgPaths=[], selectedLabel=None):
    print("Updating Displays...")

    buttonOptions = {
      "Info": {
          "color":"INFO",
          "callback":self.displayInfo
        }
    }

    for x in range(len(imgPaths)):
      newImgDisp = None
      if type(imgPaths[x]) == str and os.path.exists(imgPaths[x]):
        curLabel = selectedLabel
        pathNumber = imgPaths[x].split("_")[-1].split(".")[0]
        curLabel = f"{curLabel} :: {pathNumber}"
        newImgDisp = ImageDataDisplayWidget( imgPaths[x], curLabel, buttonOptions )
      else:
        curLabel = selectedLabel + " :: " + str(len(self.imageListWidget.gridItems))
        newImgDisp = ImageDataDisplayWidget( imgPaths[x], curLabel, buttonOptions )

      newImgDisp.subscribeToDelete( self.displayDelete )

      self.imageListWidget.addItems(newImgDisp)
  
  def displayInfo(self, curDisplay ):
    curImgPath = curDisplay.imagePath
    print("Displaying Info for : ", curImgPath)
  
  def displayDelete(self, curLabel ):
    self.statusText.setStatusText("Delisting : ", curLabel.imagePath)
    self.imageListWidget.removeItem(curLabel)


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
    if label == "Gen Epochs":
      self.generationEpochs = value
    elif label == "Gen Batch Size":
      self.generationBatchSize = value

  def trainVAE(self):
    if self.vae is None:
      self.loadVAE()
    self.statusText.setStatusText("Training VAE...")
    if self.statusText is not None:
      self.statusText.setCallbackMode( self.trainingMode["VAE"] )
    self.runTrainVAE()
    #self.runTrainGenerationStack()

  def saveVAE(self, isSession=False):
    if self.vae is None:
      self.loadVAE()
    if isSession:
      self.vae.saveSession()
    else:
      self.vae.save()

  def trainDiffusion(self):
    if self.diffusionModel is None:
      self.loadModel()
    self.statusText.setStatusText("Training Diffusion Model...")
    if self.statusText is not None:
      self.statusText.setCallbackMode( self.trainingMode["Diffusion"] )
    self.runTrainDiffusion()
    #self.runTrainGenerationStack()

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
    self.statusText.setStatusText("-- All Models Saved --")

  def saveVAEClicked(self):
    self.saveVAE()
    self.saveVAE( True )
    self.statusText.setStatusText("-- VAE Encodings & Decodings Saved --")

  def saveDiffusionClicked(self):
    self.saveDiffusion()
    self.saveDiffusion( True )
    self.statusText.setStatusText("-- Diffusion Model Saved --")

  def saveSession(self):
    if self.vae is None:
      self.loadVAE()
    self.vae.saveSession()
    if not os.path.exists(self.outputSessionFolder):
      os.makedirs(self.outputSessionFolder)
    if self.diffusionModel is None:
      self.loadModel()
    self.diffusionModel.save( os.path.join(self.outputSessionFolder, "diffusion_model"+str(self.sessionId)+"." + self.outputFileType) )
    self.statusText.setStatusText("-- VAE & Diffusion Session Saved --")


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
                    parent = self,
                    sessionId = self.sessionId, 
                    statusBar = self.statusText,
                    labels = self.labelLinks,
                    epochs = self.epochs,
                    batch_size = self.batchSize,
                    outputFolder = self.outputFolder,
                    outputEncoderPath = self.outputEncoder,
                    outputDecoderPath = self.outputDecoder,
                    outputSessionPath = self.outputSessionFolder,
                    outputType = self.outputFileType,
                    latent_dim = self.latentDim,
                    encoderDecoderSizes = self.encoderDecoderSizes,
                    reluLayers = self.reluLayers
                  )

    self.vae.load()

    self.statusText.setStatusText("-- VAE Encodings & Decodings Loaded --")

  def loadModel(self):
    if os.path.exists(self.outputDiffusion):
      self.diffusionModel = models.load_model( self.outputDiffusion, custom_objects={"mse": MeanSquaredError()} )
    else:
      self.diffusionModel = self.buildDiffusionModel(self.latentDim)
      self.diffusionModel.save( self.outputDiffusion )

    self.needsFit = True

    # Train Compiling of the Model and fitting the data occurs only when the training occurs
    #   This is to allow easier instant use of the existing model

    self.diffusionModel.summary()
    self.statusText.setStatusText("-- Diffusion Model Loaded --")

  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def findStartingCount(self, folder, basename=None, ext=None):
    if not os.path.exists(folder):
      os.makedirs(folder)
    curFolderList = os.listdir(folder)
    if len(curFolderList) == 0:
      return 0
    
    outCount = 0
    filteredList = []
    if ext is not None:
      filteredList =  list(filter(lambda x: x.endswith(ext), curFolderList)) 
    if basename is not None:
      filteredList +=  list(filter(lambda x: x.startswith(basename), curFolderList)) 
    if ext is not None and basename is not None:
      filteredList = sorted( list(set(map( lambda x: int(x.split("_")[-1].split(".")[0]), filteredList ))) )
      outCount = filteredList[-1]+1
    else:
      outCount = len(filteredList)
      
    return outCount

  def generateImages(self):
    selected_label = self.labelCombo.currentText()

    if self.diffusionModel is None:
      self.loadModel()

    # Generate images using the selected label and slider values
    new_images, qimageData = self.GenerateSpecificCharacter(selected_label, self.diffusionModel, self.outputSessionFolder)

    if new_images is None:
      print("Error: No images generated")
      return;
  
    self.UpdateDisplays(qimageData, selected_label)


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

  def GenerateRemapper(self, diffusion_model, imageList, outputSize, predictionCount=4, output_folder="", runDiffusionFit=False ):
    
    curImageList = imageList
    #curImageList = imageList if imageList is not None else self.images[outputSize]
    
    if self.vae is None:
      self.loadVAE()

    # Load the VAE encoder and decoder
    encoder,decoder = self.vae.getEncoder(outputSize)
    if type(diffusion_model) == str:
      diffusion_model = models.load_model(diffusion_model, custom_objects={"mse": MeanSquaredError()})

    # Generate new latent vectors
    z_combined = self.calcZValues(encoder)

    # Limit predictions
    if predictionCount > z_combined.shape[0]:
      predictionCount = z_combined.shape[0] - 1
    z_combined = z_combined[:predictionCount]


    # Compile and train the diffusion model
    if runDiffusionFit:
      print("-- fitting diffusion model --")
      diffusion_model.compile(optimizer="adam", loss="mse")
      diffusion_model.fit(z_combined, z_combined, epochs=self.epochs, batch_size=self.batchSize)

    self.statusText.setStatusText("Generating latent vectors...")
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
    self.statusText.setStatusText("Generating new images...")
    new_images = decoder.predict(new_latent_vectors)

    # Verify the generated images using the one-hot encoded channels
    verified_images = []
    curLabel = self.labelCombo.currentText().split("_")[0]
    for img in new_images:
      # Decode the one-hot encoded channels to get the predicted label
      predicted_label = np.argmax(img, axis=-1)
      
      """curImg = img[:, :, -1]
      width, height = curImg.shape
      curImg = (curImg * 255).astype(np.uint8)
      curImg = QImage(curImg, width, height, 3*width, QImage.Format.Format_RGB888)
      curPixmap = QPixmap.fromImage(curImg)"""


      """newImgDisplay = QLabel()
      newImgDisplay.setPixmap(curPixmap)"""

      #self.imageListWidget.addItem(newImgDisplay)

      # Check if the predicted label matches the selected label
      #if predicted_label == self.labelLinks[curLabel]:
      #  verified_images.append(img)
    
    print(f"Verified {len(verified_images)} images for label: {curLabel}")



    print("New images generated - ", len(new_images))
    self.statusText.setStatusText("-- "+str(len(new_images))+" New Images Generated --")

    # Save the generated images
    if output_folder != "":
      for x, img in enumerate(new_images):
        print(img.shape)
        curImg = img[:, :, -1]
        curImg = (curImg * 255).astype(np.uint8)
        if not os.path.exists(output_folder):
          os.makedirs(output_folder)
        curImg = Image.fromarray(curImg)
        outputPath = os.path.join(output_folder, f'pxlImageRemapper_{x}.png')
        curImg.save(outputPath)

    # Calculate reconstruction loss
    mse = MeanSquaredError()
    #imagesCount = self.images[outputSize].shape[0]
    #displayImages = new_images[:imagesCount]
    imagesCount = curImageList.shape[0]
    displayImages = new_images[:imagesCount]
    original_size = tf.shape(curImageList)[1:3]
    displayImages = tf.image.resize(displayImages, original_size)
    reconstruction_loss = mse(curImageList, displayImages).numpy()
    print(f'Reconstruction Loss: {reconstruction_loss}')

    return new_images

  # -- -- -- -- -- -- -- --

  def GenerateSpecificCharacter(self, character, diffusion_model, output_folder=""):

    character = character.split("_")[0]

    # Filter images and labels for the requested character
    filteredImages = {}
    sizes = list(self.images.keys())
    imgSize = sizes[0]

    filteredImages[imgSize] = list(filter(lambda x: x[1] == character, zip(self.images[imgSize], self.labels)))
    if len(filteredImages[imgSize]) == 0:
      print(f"No images found for character: {character}")
      return
    print(f"Found {len(filteredImages[imgSize])} images for character: {character}")
    # Convert lists to numpy arrays
    filteredImages[imgSize] = np.array([img for img, lbl in filteredImages[imgSize]])
    filteredImages[imgSize] = filteredImages[imgSize].reshape(-1, imgSize, imgSize, 1)

    # Generate new images using the filtered images
    genCount = self.generationBatchSize
    generatedImages = self.GenerateRemapper( diffusion_model, filteredImages[imgSize], imgSize, genCount, output_folder, runDiffusionFit=False )

    if self.checkBreak():
      print("Exiting Generation...")
      return;
  
    # Save the generated images
    outputImages = []
    if output_folder != "":
      startingCount = self.findStartingCount(output_folder, character, ".png")
      for x, img in enumerate(generatedImages):
        # Gather label data
        # Aggregate data from all layers into a final gray channel
        curImg = np.max( img, axis=-1 )

        curImg = (curImg * 255).astype(np.uint8)
        if not os.path.exists(output_folder):
          os.makedirs(output_folder)
        curImg = Image.fromarray(curImg)
        curCount = str(startingCount+x).zfill(3)
        outputPath = os.path.join(output_folder, f"pxlImageRemapper_{character}_{curCount}.png")
        curImg.save(outputPath)
        toqImg = QImage(curImg.tobytes("raw", "L"), curImg.width, curImg.height, QImage.Format.Format_Grayscale8)
        outputImages.append(toqImg)
        print(f"Saved image: {outputPath}")
    return generatedImages, outputImages

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

  def runGenerateImages(self, res=128):
    print(f"Entering Prediction Mode...")

    #if not fileExists(output_encoder_128) or not fileExists(output_decoder_128) or not fileExists(self.outputDiffusion):
    if not self.fileExists( self.outputDiffusion ):
      print("Error: Missing models")
      print("   Exiting ...")
      return;

    #self.vae.hasLoaded()

    if self.diffusionModel is None:
      self.loadModel()
    
    #genCount = self.generationBatchSize
    #GenerateRemapper( diffusion_model, self.images[self.inputTrainSize], self.inputTrainSize, genCount, output_folder="" )
    
    curGenLabel = self.labelCombo.currentText()
    print(f"Generating Images for : {curGenLabel}")
    self.GenerateSpecificCharacter( curGenLabel, self.diffusionModel, self.outputSessionFolder)

    self.UpdateDisplays()


  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  def runTrainVAE(self, step=0):
    statusText = "Building Encoder & Decoder..."
    if step > 0:
      statusText = f"{step} - {statusText}"
    self.statusText.setStatusText( statusText )

    if self.fileExists( self.outputEncoder ) and self.fileExists( self.outputDecoder ):
      self.statusText.setStatusText( statusText + "Loading existing models..." )
      self.vae.load()
    else:  # Instantiate the shared encoder, decoder, and VAE
      self.statusText.setStatusText( statusText + "Building new models..." )
      self.vae.prepShapes()

    self.vae.train( self.images, self.labelLinks, self.epochs, self.batchSize )
    if self.autoSave:
      self.vae.save()
    self.vae.saveSession()

  def runTrainDiffusion(self, step=0):
    statusText = "Building Diffusion model..."
    if step > 0:
      statusText = f"{step} - {statusText}"
    self.statusText.setStatusText( statusText )

    if self.vae is None:
      self.loadVAE()

    # Define the diffusion model
    if self.diffusionModel is None:
      self.loadModel()

    outputSize = self.vae.checkOutputSize(self.inputTrainSize)
    encoder,decoder = self.vae.getEncoder(outputSize)

    # Combine latent vectors from different scales
    z_combined = self.calcZValues( encoder )

    # Compile and train the diffusion model
    if self.needsFit:
      self.needsFit = False
      self.statusText.setStatusText("Fitting Diffusion Model...")
      self.diffusionModel.compile(optimizer="adam", loss="mse")
    self.diffusionModel.fit(z_combined, z_combined, epochs=self.epochs, batch_size=self.batchSize)

    # Save the diffusion model
    if self.autoSave:
      if not os.path.exists(self.outputDiffusion):
        os.makedirs(self.outputDiffusion)
      self.diffusionModel.save( self.outputDiffusion )
    if not os.path.exists(self.outputSessionFolder):
      os.makedirs(self.outputSessionFolder)
    self.diffusionModel.save( self.outputSessionFolder + "/diffusion_model." + self.outputFileType )



  def runTrainGenerationStack(self):

    step=0

    step+=1
    self.statusText.setStatusText(f"{step} - Building Encoder & Decoder...")

    if self.fileExists( self.outputEncoder ) and self.fileExists( self.outputDecoder ):
      self.statusText.setStatusText("Loading existing models...")
      self.vae.load()
    else:  # Instantiate the shared encoder, decoder, and VAE
      self.statusText.setStatusText("Building new models...")
      self.vae.prepShapes()

    self.vae.train( self.images, self.labelLinks, self.epochs, self.batchSize )

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    if self.checkBreak():
      self.statusText.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    step+=1
    self.statusText.setStatusText(f"{step} - Building Diffusion model...")

    # Define the diffusion model

    diffusion_model = self.buildDiffusionModel(self.latentDim)


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    if self.checkBreak():
      self.statusText.setStatusText("Exiting Training...")
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
      self.statusText.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    step+=1
    self.statusText.setStatusText(f"{step} - Saving VAE Encoder/Decoder & Model...")

    # Save the VAE encoder and decoder
    self.vae.save()


    # Save the diffusion model
    print(self.outputDiffusion)
    diffusion_model.save( self.outputDiffusion )
    diffusion_model.save( self.outputSessionFolder + "/diffusion_model." + self.outputFileType )

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    if self.checkBreak():
      self.statusText.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    step+=1
    print(f"{step} - Generating new images...")

    # Load the VAE encoder and decoder
    #genCount = self.generationBatchSize
    #GenerateRemapper( diffusion_model, self.images[self.inputTrainSize], self.inputTrainSize, genCount, output_folder )

    curLabel = self.labelCombo.currentText()
    self.GenerateSpecificCharacter(curLabel, diffusion_model, self.outputSessionFolder)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    
    if self.checkBreak():
      self.statusText.setStatusText("Exiting Training...")
      return

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

