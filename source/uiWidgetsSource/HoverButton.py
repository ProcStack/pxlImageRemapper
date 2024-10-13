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
#   Currently available color themes:
#    DEFAULT - Blue Theme
#    ACCEPT - Green Theme
#    INFO - Yellow Theme
#    WARNING - Red Theme
#    
#  You can add more by adding to the THEMES dictionary in pxlColors.py
#

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton

from .pxlColors import pxlColors


class HoverButtonWidget(QPushButton):

  def __init__(self, label, fontSize=20, theme="DEFAULT"):
    super().__init__()
    self.isHover = False
    self.setText(label)
    self.setFixedHeight(45)
    #self.setFixedWidth(200)

    theme = theme.upper() if theme.upper() in pxlColors.THEMES else "DEFAULT"
    self.theme = pxlColors.THEMES[theme]
    self.styleBase = f"font-size: {fontSize}px; font-weight: bold; border-radius: 5px;"
    self.styleNoHover = "color: "+self.theme["BASE"]["text"]+"; background-color: "+self.theme["BASE"]["bg"]+";  border: 1px solid "+self.theme["BASE"]["brd"]+"; padding:1px 1px 1px 1px;" + self.styleBase
    self.styleHover = "color: "+self.theme["HOVER"]["text"]+"; background-color: "+self.theme["HOVER"]["bg"]+"; border: 2px solid "+self.theme["HOVER"]["brd"]+"; " + self.styleBase

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
    color = color.upper() if color.upper() in pxlColors.THEMES else "DEFAULT"
    self.theme = pxlColors.THEMES[color]
    self.styleNoHover = "color: "+self.theme["BASE"]["text"]+"; background-color: "+self.theme["BASE"]["bg"]+";  border: 1px solid "+self.theme["BASE"]["brd"]+"; padding:1px 1px 1px 1px;" + self.styleBase
    self.styleHover = "color: "+self.theme["HOVER"]["text"]+"; background-color: "+self.theme["HOVER"]["bg"]+"; border: 2px solid "+self.theme["HOVER"]["brd"]+"; " + self.styleBase
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

    widget = HoverButtonWidget("Bloogity", 25, "green")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Info Goo", theme="yellow")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Tester Face McGoo", 30, "red")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Grank a tank a lank a loo", 18, "BLUE")
    widget.clicked.connect(onButtonClicked)
    mainLayout.addWidget(widget)

    widget = HoverButtonWidget("Random", 25, "green")

    def setRandomTheme():
        import random
        color = random.choice(list(pxlColors.THEMES.keys()))
        print(f"Setting Theme to {color}")
        widget.setTheme(color)

    widget.clicked.connect(setRandomTheme)
    mainLayout.addWidget(widget)

    windowWidget.setLayout(mainLayout)
    windowWidget.show()

    sys.exit(app.exec())