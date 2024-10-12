#
#    Button widget with hover effects
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#
#   Standalone Unit Test -
#     Display a window with a few HoverButtonWidgets
#       Moving your mouse over the buttons will change it to their hover color
#       Clicking the buttons will print "Boop!" to the console
#       Clicking the Random button with recolor the button with a random theme
#
#   Currently there are three color themes:
#    DEFAULT - Blue Theme
#    ACCEPT - Green Theme
#    INFO - Yellow Theme
#    WARNING - Red Theme
#    
#  You can add more by adding to the COLORS dictionary on HoverButtonWidget 
#

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton


class HoverButtonWidget(QPushButton):
  COLORS = {
    "DEFAULT": {
      "text": "#ffffff",
      "textHover": "#eff6ff",
      "bg": "#353535",
      "bgHover": "#404550",
      "brd": "#808080",
      "brdHover": "#5599cc"
    },
    "ACCEPT": {
      "text": "#ffffff",
      "textHover": "#eff6ff",
      "bg": "#405040",
      "bgHover": "#557055",
      "brd": "#808080",
      "brdHover": "#70cc70"
    },
    "INFO": {
      "text": "#ffffff",
      "textHover": "#eff6ff",
      "bg": "#505040",
      "bgHover": "#707045",
      "brd": "#808080",
      "brdHover": "#cccc55"
    },
    "WARNING": {
      "text": "#ffffff",
      "textHover": "#eff6ff",
      "bg": "#553535",
      "bgHover": "#704550",
      "brd": "#808080",
      "brdHover": "#cc5065"
    }
  }

  def __init__(self, label, fontSize=20, color="DEFAULT"):
    super().__init__()
    self.isHover = False
    self.setText(label)
    self.setFixedHeight(45)
    #self.setFixedWidth(200)

    color = color.upper() if color.upper() in HoverButtonWidget.COLORS else "DEFAULT"
    self.theme = HoverButtonWidget.COLORS[color]
    self.styleBase = f"font-size: {fontSize}px; font-weight: bold; border-radius: 5px;"
    self.styleNoHover = "color: "+self.theme["text"]+"; background-color: "+self.theme["bg"]+";  border: 1px solid "+self.theme["brd"]+"; padding:1px 1px 1px 1px;" + self.styleBase
    self.styleHover = "color: "+self.theme["textHover"]+"; background-color: "+self.theme["bgHover"]+"; border: 2px solid "+self.theme["brdHover"]+"; " + self.styleBase

    self.setStyleSheet(self.styleNoHover)

  def enterEvent(self, event):
    self.isHover = True
    self.setStyleSheet(self.styleHover)
    self.setCursor(Qt.CursorShape.PointingHandCursor)
    super().enterEvent(event)
  
  def leaveEvent(self, event):
    self.isHover = False
    self.setStyleSheet(self.styleNoHover)
    self.setCursor(Qt.CursorShape.ArrowCursor)
    super().leaveEvent(event)

  def setTheme(self, color):
    color = color.upper() if color.upper() in HoverButtonWidget.COLORS else "DEFAULT"
    self.theme = HoverButtonWidget.COLORS[color]
    self.styleNoHover = "color: "+self.theme["text"]+"; background-color: "+self.theme["bg"]+";  border: 1px solid "+self.theme["brd"]+"; padding:1px 1px 1px 1px;" + self.styleBase
    self.styleHover = "color: "+self.theme["textHover"]+"; background-color: "+self.theme["bgHover"]+"; border: 2px solid "+self.theme["brdHover"]+"; " + self.styleBase
    if self.isHover:
      self.setStyleSheet(self.styleHover)
    else:
      self.setStyleSheet(self.styleNoHover)



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --




# Unit test
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

    def onButtonClicked():
        print("Boop!")
    

    app = QApplication(sys.argv)
    windowWidget = QWidget()
    windowWidget.resize(400, 300)
    windowWidget.setWindowTitle("Hover Button Test")
    mainLayout = QVBoxLayout()
    widget = HoverButtonWidget("Tester Face McGee")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Info Goo", color="info")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Tester Face McGoo", 30, "WARNING")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Done", 25, "ACCEPT")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Random", 25, "ACCEPT")

    def setRandomTheme():
        import random
        color = random.choice(list(HoverButtonWidget.COLORS.keys()))
        print(f"Setting Theme to {color}")
        widget.setTheme(color)

    widget.clicked.connect(setRandomTheme)
    mainLayout.addWidget(widget)

    windowWidget.setLayout(mainLayout)
    windowWidget.show()

    sys.exit(app.exec())