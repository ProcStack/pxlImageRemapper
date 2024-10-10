#
#    Custom status display widget
#      Display colored status messages & progress bars
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#


from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar


class StatusDisplay(QWidget):
  def __init__(self, app=None):
    super().__init__()
    self.app = app
    self.statusText = None
    self.statusTimer = QTimer()
    self.statusBar = None
    self.statusPercent = 0.0


    self.initHelpers()
    self.initUI()
  
  # -- -- --

  def initUI(self):
    layout = QVBoxLayout()
    layout.setSpacing(4)
    layout.setContentsMargins(3, 3, 3, 3)

    self.statusText = QLabel(self)
    layout.addWidget(self.statusText)

    self.statusBar = QProgressBar(self)
    layout.addWidget(self.statusBar)

    self.setLayout(layout)
  
  # -- -- --
  
  # Helper functions for status bar display

  def setStatusBar(self, percent=0.0):
    if not self.statusBar.isVisible():
      self.statusBar.setVisible(True)
    self.statusBar.setValue(percent)
    self.statusPercent = percent

  def showStatusBar(self):
    self.statusBar.setVisible(True)
  def hideStatusBar(self):
    self.statusBar.setVisible(False)

  # -- -- --

  # Helper functions for time based status text display
  #  Yes default status bar has a timer, but I like colored statuses

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