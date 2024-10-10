#
#    Custom status display widget
#      Display colored status messages & progress bars
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#
#  When `autoHideStatusBar` is set to False,
#    The Status Bar stays after 100% completion.
#  Its expected for a call to the StatusDisplay to hide the status bar,
#    Mostly for confirmation or long haul processes
#  callback() -> StatusDisplay.hideStatusBar()

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton

from HoverButton import HoverButtonWidget

class StatusDisplay(QWidget):
  def __init__(self, app=None,autoHideStatusBar=False):
    super().__init__()
    self.app = app
    self.statusText = None
    self.statusTimer = QTimer()
    self.statusBar = None
    self.statusPercent = 0.0
    self.autoHideStatusBar = autoHideStatusBar
    self.isDone = False

    self.progressHeight = 20
    self.cancelHeight = 22
    self.cancelFont = 18

    self.styleStatusBar = """
        QProgressBar {border: 2px solid #5599cc; border-radius: 5px; text-align: center; font-size: 16px; font-weight: bold; color: #eff6ff; background-color: #404550;}
        QProgressBar::chunk {background-color: #5599cc;}
      """

    self.cancelCallbacks = []

    self.initUI()
  
  # -- -- --

  def initUI(self):
    layout = QVBoxLayout()
    layout.setSpacing(4)
    layout.setContentsMargins(3, 3, 3, 3)

    self.statusText = QLabel(self)
    layout.addWidget(self.statusText)

    progressBarLayout = QHBoxLayout()

    self.statusBar = QProgressBar(self)
    self.statusBar.setVisible(False)
    self.statusBar.setFixedHeight(0)
    self.statusBar.setStyleSheet(self.styleStatusBar)
    progressBarLayout.addWidget(self.statusBar)

    self.cancelButton = HoverButtonWidget( "X", self.cancelFont )
    self.cancelButton.setFixedWidth( self.progressHeight )
    self.cancelButton.setFixedHeight(0)
    self.cancelButton.clicked.connect(self.cancel)
    self.cancelButton.setVisible(False)
    progressBarLayout.addWidget(self.cancelButton)

    layout.addLayout(progressBarLayout)

    self.setLayout(layout)
  
  # -- -- --
  
  # Helper functions for status bar display

  def setStatusBar(self, percent=0):
    percent = min(100, max(0, percent))
    if not self.statusBar.isVisible():
      self.showStatusBar()
    self.statusBar.setValue(percent)
    self.statusPercent = percent
    if percent == 100 :
      self.isDone = True
      if self.autoHideStatusBar:
        self.hideStatusBar()
      else:
         self.cancelButton.setText("O")

  def showStatusBar(self):
    self.isDone = False
    self.statusBar.setVisible(True)
    self.statusBar.setFixedHeight( self.progressHeight )
    self.cancelButton.setVisible(True)
    self.cancelButton.setFixedHeight( self.cancelHeight )
  def hideStatusBar(self):
    self.isDone = True
    self.statusBar.setVisible(False)
    self.statusBar.setFixedHeight(0)
    self.cancelButton.setVisible(False)
    self.cancelButton.setFixedHeight(0)
    self.cancelButton.setText("X")

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

  # -- -- --

  def subscribeToCancel(self, callback):
    self.cancelCallbacks.append(callback)
  def cancel(self):
    if not self.isDone:
      for callback in self.cancelCallbacks:
        callback()
    self.hideStatusBar()



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



# Unit test
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    unitTestWindow = QWidget()
    unitTestWindow.setWindowTitle("Status Display Widget Test")
    unitTestWindow.resize(300, 100)

    unitLayout = QVBoxLayout()

    unitTimer = QTimer()
    widget = StatusDisplay(app)
    widget.setStatusText("Booted!", 2000)

    percent = -1
    testRunning = False
    clearOnComplete = True
    def startStatusTest():
      global percent
      global testRunning
      global clearOnComplete
      print("Starting!")
      percent = -1
      testRunning = True
      clearOnComplete = True
      updateStatus()
    def startNoClearTest():
      global percent
      global testRunning
      global clearOnComplete
      print("Starting!")
      percent = -1
      testRunning = True
      clearOnComplete = False
      updateStatus()

    def updateStatus():
      global percent
      global testRunning
      global unitTimer
      if not testRunning:
        widget.hideStatusBar()
        return;
      if percent == -1:
        percent = 0
        widget.showStatusBar()
        unitTimer.start(50)
      else:
        percent += 2

      widget.setStatusBar(percent)
      if percent >= 100:
        widget.setStatusText("Done!", 2000)
        if clearOnComplete:
          widget.hideStatusBar()
        percent = -1
        unitTimer.stop()
        testRunning = False
      elif percent > -1 and percent < 100:
        widget.setNoTimerStatusText("Working...")
    def cancelTest():
      global percent
      global testRunning
      global unitTimer
      print("Cancelled!")
      percent = -1
      testRunning = False
      unitTimer.stop()
      widget.setStatusText("Cancelled!", 1500)

    unitTimer.timeout.connect(updateStatus)
    widget.subscribeToCancel(cancelTest)
    unitLayout.addWidget(widget)

    buttonLayout = QHBoxLayout()
    runProgressTestButton = QPushButton("Progress Test")
    runProgressTestButton.clicked.connect(startStatusTest)
    buttonLayout.addWidget(runProgressTestButton)

    runProgressTestButton = QPushButton("No Clear Test")
    runProgressTestButton.clicked.connect(startNoClearTest)
    buttonLayout.addWidget(runProgressTestButton)

    unitLayout.addLayout(buttonLayout)

    unitTestWindow.setLayout(unitLayout)
    unitTestWindow.show()


    sys.exit(app.exec())