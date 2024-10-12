
import os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout

if __name__ == "__main__":
  from HoverButton import HoverButtonWidget
else:
  from source.uiWidgetsSource.HoverButton import HoverButtonWidget


class ImageDataDisplayWidget(QWidget):
  def __init__(self, imagePath=None, labelText="Default", buttonOptions={}):
    super().__init__()
    self.image = imagePath if type(imagePath) == QImage else None
    self.imagePixmap = None
    self.imagePath = imagePath if type(imagePath) == str else None
    self.labelText = labelText
    self.textLabel = None
    self.imageLabel = None

    self.buttonSettings = {
      "fontSize": 14,
      "color": "DEFAULT"
    }

    self.buttonOptions = buttonOptions
    self.callbackList = []
    self.extraData = {}

    self.initUI()

  def initUI(self):
    if not self.imagePath is None and not os.path.exists(self.imagePath):
      self.imagePath = None
    
    mainLayout = QVBoxLayout()
    mainLayout.setSpacing(2)
    mainLayout.setContentsMargins(3, 1, 3, 1)

    if self.imagePath is not None:
      self.image = self.loadImage(self.imagePath)

    self.imageLabel = QLabel(self)
    if self.image is not None:

      self.imagePixmap = QPixmap.fromImage(self.image)
      self.imageLabel.setPixmap(self.imagePixmap)
    else:
      self.imageLabel.setText("[: Missing :]")
    self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    mainLayout.addWidget(self.imageLabel)

    lableLayout = QHBoxLayout()
    self.textLabel = QLabel(self.labelText)
    self.textLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lableLayout.addWidget(self.textLabel)

    
    buttonLayout = QHBoxLayout()

    buttonKeys = list(self.buttonOptions.keys())
    for x in range(len(buttonKeys)):
      curText = buttonKeys[x]
      curFontSize = self.buttonOptions[buttonKeys[x]]["fontSize"] if "fontSize" in self.buttonOptions[buttonKeys[x]] else self.buttonSettings["fontSize"]
      curColor = self.buttonOptions[buttonKeys[x]]["color"] if "color" in self.buttonOptions[buttonKeys[x]] else "DEFAULT"
      curCallback = self.buttonOptions[buttonKeys[x]]["callback"] if "callback" in self.buttonOptions[buttonKeys[x]] else None
      curWidth = self.buttonOptions[buttonKeys[x]]["width"] if "width" in self.buttonOptions[buttonKeys[x]] else 40
      curHeight = self.buttonOptions[buttonKeys[x]]["height"] if "height" in self.buttonOptions[buttonKeys[x]] else 20
      button = HoverButtonWidget( curText, curFontSize, curColor) 
      button.setFixedWidth( curWidth )
      button.setFixedHeight( curHeight )
      button.clicked.connect( curCallback )
      buttonLayout.addWidget( button) 

    # Default Buttons
    self.deleteButton = HoverButtonWidget( "Del", 14, "WARNING" )
    self.deleteButton.setFixedWidth( 40 )
    self.deleteButton.setFixedHeight( 20 )
    self.deleteButton.clicked.connect(lambda: self.delete())
    buttonLayout.addWidget(self.deleteButton)
    lableLayout.addLayout(buttonLayout)

    mainLayout.addLayout(lableLayout)

    self.setLayout(mainLayout)

  def loadImage(self, imagePath):
    if not os.path.exists(imagePath):
      return None
    img = QImage(imagePath)
    return img

  def subscribeToDelete(self, callback):
    self.callbackList.append(callback)

  def delete(self):
    for x in range(len(self.callbackList)):
      self.callbackList[x](self)
    self.deleteLater()



# Unit test
if __name__ == "__main__":
  import sys
  from PyQt6.QtWidgets import QApplication

  app = QApplication(sys.argv)
  windowWidget = QWidget()
  windowWidget.resize(400, 300)
  windowWidget.setWindowTitle("Image Data Display Test")
  mainLayout = QVBoxLayout()
  mainLayout.setSpacing(0)
  mainLayout.setContentsMargins(0, 0, 0, 0)

  def infoButtonClicked():
    print("Info Button Clicked")

  def deleteButtonClicked(obj):
    print("Removing Item...")
  
  buttonOptions = {
    "Info": {
        "color":"info",
        "callback":infoButtonClicked
      }
  }

  def addImage():
    widget = ImageDataDisplayWidget("source/uiWidgetsSource/testImage.png", "Test Image", buttonOptions)
    widget.subscribeToDelete(deleteButtonClicked)
    mainLayout.addWidget(widget)
    windowWidget.setLayout(mainLayout)

  addButton = HoverButtonWidget("Add Image")
  addButton.clicked.connect(addImage)
  mainLayout.addWidget(addButton)

  # -- --

  widget = ImageDataDisplayWidget("source/uiWidgetsSource/testImage.png", "Test Image", buttonOptions)
  widget.subscribeToDelete(deleteButtonClicked)
  mainLayout.addWidget(widget)

  windowWidget.setLayout(mainLayout)
  windowWidget.show()

  sys.exit(app.exec())