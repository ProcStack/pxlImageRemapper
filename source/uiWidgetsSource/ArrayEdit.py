#
#    Text field widget for array types
#      Handles input of integer arrays
#      Maintains `values` list object
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#
#   Standalone Unit Test -
#     Display a window with an ArrayEditWidget
#       The ArrayEditWidget will display a list of integers 1-10
#     You can edit the list in the text field
#       Click in the area below to stop editing the field
#
#   The ArrayEditWidget will only accept integers currently
#     Maybe in the future any ascii, but only integers were needed for the widgets initial purpose
#

from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtGui import QTextCursor


class ArrayEditWidget(QTextEdit):
  def __init__(self, label, values):
    super().__init__()

    self.label = label
    self.hasUpdate = False
    self.callbacks = []
    self.latchFocus = False

    self.fieldNoHover = "background-color: #353535; color: white; font-size: 20px; font-weight: bold; border: 1px solid #808080; border-radius: 5px;"
    self.fieldHover = "background-color: #404550; color: #eff6ff; font-size: 20px; font-weight: bold; border: 2px solid #5599cc; border-radius: 5px;"

    self.setFixedHeight(45)
    self.setStyleSheet(self.fieldNoHover)
    if type(values) == list:
      values = list(filter(lambda x: str(x).isnumeric(), values))
      values = " ".join(list(map(lambda x: str(x), values)))
    elif type(values) == dict:
      values = (key + ":" + str(values[key]) for key in values.keys())
      values = " ".join(values)
    values = values.replace(","," ")
    self.values = list(map(lambda x: int(x), values.split(" ")))

    valueStr = ", ".join(values.split(" "))
    self.setText(valueStr)
    self.prevValue = valueStr

    self.textChanged.connect(self.onValueChanged)

  def onValueChanged(self):
    origText = self.toPlainText()
    curValues = "".join(list(filter(lambda x: x.isnumeric() or x in [" ",","], origText)))
    if curValues != origText:
      curPos = self.textCursor().position()
      self.setText(curValues)
      curPos = min(curPos, len(curValues))
      for _ in range(curPos):
        self.moveCursor(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor)

  def enterEvent(self, event):
    self.setStyleSheet(self.fieldHover)
    super().enterEvent(event)
  
  def leaveEvent(self, event):
    curFocus = self.hasFocus()
    if not curFocus:
      self.setStyleSheet(self.fieldNoHover)
    else:
      self.latchFocus = True
    super().leaveEvent(event)

  def focusOutEvent(self, event):
    self.cleanValues()

    if self.latchFocus:
      self.setStyleSheet(self.fieldNoHover)
      self.latchFocus = False

    if self.hasUpdate:
      self.updateValues()
      self.trigger()
    super().focusOutEvent(event)

  def cleanValues(self):
    curText = self.toPlainText()
    if curText == "":
      return;
    curText = "".join(list(filter(lambda x: x.isnumeric() or x == " ", curText)))
    curText = list(filter(lambda x: x.isnumeric() and x is not None, curText.split(" ")))
    curValues = list(map(lambda x: int(x), curText))

    toText = ", ".join(list(map(lambda x: str(x), curValues)))
    if toText != self.prevValue:
      self.prevValue = toText
      self.hasUpdate = True
      self.setText( toText )

  def updateValues(self):
    curText = self.toPlainText()
    if curText == "":
      return;
    curText = "".join(list(filter(lambda x: x.isnumeric() or x == " ", curText)))
    curValues = list(map(lambda x: int(x), curText.split(" ")))
    
    self.values = curValues
  
  def subscribe(self, callback):
    self.callbacks.append(callback)
  
  def trigger(self):
    for x in range(len(self.callbacks)):
      self.callbacks[x](self.label, self.values)
    self.hasUpdate = False


# Unit test
if __name__ == "__main__":
  from PyQt6.QtCore import Qt
  from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
  import sys

  app = QApplication(sys.argv)
  widget = QWidget()
  widget.setFixedWidth(400)
  widget.setFixedHeight(200)
  widget.setWindowTitle("ArrayEditWidget Test")
  mainLayout = QVBoxLayout()
  arrayEdit = ArrayEditWidget("test", [1,2,3,4,5,6,7,8,9,10])
  mainLayout.addWidget(arrayEdit)

  def defocus():
    arrayEdit.clearFocus()
  
  nullButton = QLabel("Click to defocus")
  nullButton.setAlignment(Qt.AlignmentFlag.AlignCenter)
  nullButton.mousePressEvent = lambda event: defocus()
  mainLayout.addWidget(nullButton)

  widget.setLayout(mainLayout)
  widget.show()
  sys.exit(app.exec())