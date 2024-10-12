#
#    Slider widget with label & editable value
#      Handles updating editable text field
#      With hover & editing effects
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QTextEdit, QSlider

class SliderLabelWidget(QWidget):
  def __init__(self, label, range, value, labelWidth=100):
    super().__init__()

    self.label = label
    self.value = value
    self.range = range
    self.isValueEditing = False
    self.callbackList = []
    self.latchFocus = False

    self.valueEditNoHover = "background-color: #353535; border: 1px solid #808080; border-radius: 5px; font-size: 20px; font-weight: bold;"
    self.valueEditHover = "background-color: #404550; border: 2px solid #5599cc; border-radius: 5px; font-size: 20px; font-weight: bold;"

    self.valueSliderNoHover = """
        QSlider::groove:horizontal { border: 1px solid #808080; height: 4px; margin: 0px; border-radius: 7px; }
        QSlider::handle:horizontal { background-color: #404550; border: 1px solid #808080; width: 20px; margin: -7px 0px -7px 0px; border-radius: 7px; }
      """
    self.valueSliderHover = """
        QSlider::groove:horizontal { border: 1px solid #808080; height: 4px; margin: 0px; }
        QSlider::handle:horizontal { background-color: #404550; border: 2px solid #2565aa; width: 20px; margin: -7px 0px -7px 0px; border-radius: 7px; }
        QSlider::handle:horizontal:hover { background-color: #404550; border: 2px solid #5599cc; width: 20px; margin: -7px 0px -7px 0px; border-radius: 7px; }
      """
    self.initUI(labelWidth)
  
  def initUI(self,labelWidth):
    layout = QHBoxLayout()
    layout.setSpacing(2)
    layout.setContentsMargins(0, 0, 0, 0)
    headerLabel = QLabel(self.label + " : ")
    headerLabel.setStyleSheet("font-size: 18px;")
    headerLabel.setFixedWidth(labelWidth)
    headerLabel.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
    layout.addWidget( headerLabel )

    self.valueEdit = QTextEdit()
    self.valueEdit.setFixedHeight(45)
    self.valueEdit.setFixedWidth(55)
    self.valueEdit.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.valueEdit.setText(str(self.value))
    self.valueEdit.textChanged.connect(self.onValueChanged)
    self.valueEdit.focusOutEvent = self.delatchFocus
    self.valueEdit.setStyleSheet(self.valueEditNoHover)

    layout.addWidget(self.valueEdit)

    self.slider = QSlider(Qt.Orientation.Horizontal)
    self.slider.setRange(self.range[0], self.range[1])
    self.slider.setValue(self.value)
    self.slider.setStyleSheet( self.valueSliderNoHover )

    self.slider.valueChanged.connect(lambda : self.updateValue(False))
    layout.addWidget(self.slider)
    self.setLayout(layout)

    self.setFixedHeight(45)

  def onValueChanged(self):
    origText = self.valueEdit.toPlainText()
    curValues = "".join(list(filter(lambda x: x.isnumeric(), origText)))
    curValues = "0" if curValues == "" else curValues

    if curValues != origText:
      curPos = self.valueEdit.textCursor().position()
      self.valueEdit.setText(curValues)
      curPos = min(curPos, len(curValues))

      for _ in range(curPos):
        self.valueEdit.moveCursor(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor)
    else:
      self.latchFocus = True

  def enterEvent(self, event):
    self.valueEdit.setStyleSheet(self.valueEditHover)
    self.slider.setStyleSheet(self.valueSliderHover)
    self.valueEdit.setReadOnly(False)
    super().enterEvent(event)
  
  def leaveEvent(self, event):
    curFocus = self.valueEdit.hasFocus()
    if not curFocus:
      self.valueEdit.setStyleSheet(self.valueEditNoHover)
      self.slider.setStyleSheet(self.valueSliderNoHover)
      self.valueEdit.setReadOnly(True)
    else:
      self.latchFocus = True
    super().leaveEvent(event)

  def delatchFocus(self, event):
    if self.latchFocus:
      self.valueEdit.setStyleSheet(self.valueEditNoHover)
      self.slider.setStyleSheet(self.valueSliderNoHover)
      self.valueEdit.setReadOnly(True)
      self.latchFocus = False

      curValues = "".join(list(filter(lambda x: x.isnumeric(), self.valueEdit.toPlainText())))
      if curValues == "":
        curValues = self.value
      else:
        curValues = int(curValues)
        if curValues < self.range[0]:
          curValues = self.range[0]
        elif curValues > self.range[1]:
          curValues = self.range[1]
      if curValues != self.value:
        self.value = curValues
        self.valueEdit.setText(str(self.value))
        self.slider.setValue(self.value)
        self.trigger()

    super().focusOutEvent(event)

  def getValue(self, isField=False):
    toVal = 0
    if isField:
      origVal = self.valueEdit.toPlainText()
      toVal = "".join(list(filter(lambda x: x.isnumeric(), origVal)))
      if toVal == "":
        toVal = self.slider.value()
      else:
        toVal = int(toVal)
        if toVal < self.range[0]:
          toVal = self.range[0]
        elif toVal > self.range[1]:
          toVal = self.range[1]
      if str(toVal) != origVal:
        self.valueEdit.setText(str(toVal))
    else:
      toVal = self.slider.value()
    return toVal
  
  def updateValue(self, isField=False, widget=None):
    toVal = self.getValue( isField )
    if toVal != self.value:
      self.value = toVal
      self.valueEdit.setText(str(self.value))
      self.trigger()
    
  def subscribe(self, callback):
    self.callbackList.append(callback)
  
  def trigger(self):
    for x in range(len(self.callbackList)):
      self.callbackList[x](self.label, self.value)




# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --




# Unit test
if __name__ == "__main__":
  import sys
  from PyQt6.QtWidgets import QApplication

  def printValue(label, value):
    print(f"{label} : {value}")

  app = QApplication(sys.argv)
  window = SliderLabelWidget("Jamtasticness", (0, 100), 50, 150)
  window.subscribe(printValue)
  window.show()
  sys.exit(app.exec())