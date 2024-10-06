import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QTextEdit, QSlider, QHBoxLayout



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
    print("Delatching Focus")
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


class PIRButton(QPushButton):
  def __init__(self, label):
    super().__init__()
    self.setText(label)
    self.setFixedHeight(45)
    #self.setFixedWidth(200)

    self.styleBase = "font-size: 20px; font-weight: bold; border-radius: 5px;"
    self.styleNoHover = "background-color: #353535; color: white; border: 1px solid #808080; padding:1px 1px 1px 1px;" + self.styleBase
    self.styleHover = "background-color: #404550; color: #eff6ff; border: 2px solid #5599cc; " + self.styleBase

    self.setStyleSheet(self.styleNoHover)

  def enterEvent(self, event):
    self.setStyleSheet(self.styleHover)
    self.setCursor(Qt.CursorShape.PointingHandCursor)
    super().enterEvent(event)
  
  def leaveEvent(self, event):
    self.setStyleSheet(self.styleNoHover)
    self.setCursor(Qt.CursorShape.ArrowCursor)
    super().leaveEvent(event)

class PIRArrayEdit(QTextEdit):
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


