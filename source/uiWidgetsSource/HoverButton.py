#
#    Button widget with hover effects
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton


class HoverButtonWidget(QPushButton):
  def __init__(self, label, fontSize=20):
    super().__init__()
    self.setText(label)
    self.setFixedHeight(45)
    #self.setFixedWidth(200)

    self.styleBase = f"font-size: {fontSize}px; font-weight: bold; border-radius: 5px;"
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


# Unit test
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    def onButtonClicked():
        print("Boop!")

    app = QApplication(sys.argv)
    widget = HoverButtonWidget("Tester Face McGee")
    widget.clicked.connect(onButtonClicked)
    widget.show()
    sys.exit(app.exec())