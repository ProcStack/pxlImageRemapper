#
#    Array-Based Grid Display Widget
#      Add/Remove items to a grid layout
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#
#   Standalone Unit Test -
#    Display a window with an Add Item button on top and a GridArrayDisplayWidget
#      The GridArrayDisplayWidget will display a grid of items
#    Clicking the "Add Item" button will add a new item to the grid
#    Clicking an item in the grid will remove it
#
#  I was getting annoyed with the limitations of the QGridLayout
#    So I made this to handle a grid layout with a list of QWidgets
#  Overkill, sure, but I just wanted a simple-to-use auto-adjusting grid layout
#

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QScrollArea
from PyQt6.QtCore import Qt


class GridArrayDisplayWidget(QWidget):
  def __init__(self, gridItems=[], gridRes=[2,3], expandVertically=True):
    super().__init__()
    self.gridItems = gridItems
    self.gridRes = gridRes
    self.expandVertically = expandVertically
    self.layoutRowItems = []
    self.layoutItemMap = {}
    self.itemLayoutMap = {}

    self.gridLayout = None

    self.initUI()
  
  def initUI(self):
    
    mainLayout = QVBoxLayout()
    mainLayout.setSpacing(0)
    mainLayout.setContentsMargins(0, 0, 0, 0)

    # Create a scroll area
    scrollarea = QScrollArea(self)
    scrollarea.setWidgetResizable(True)
    scrollarea.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    mainLayout.addWidget(scrollarea)

    # Create a widget to hold the grid layout
    gridWidget = QWidget()
    gridWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    self.gridLayout = QVBoxLayout()
    self.gridLayout.setSpacing(0)
    self.gridLayout.setContentsMargins(0, 0, 0, 0)
    self.gridLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
    gridWidget.setLayout(self.gridLayout)
    
    scrollarea.setWidget(gridWidget)

    # -- -- --

    self.addItems( self.gridItems )
    
    # -- -- --

    self.setLayout( mainLayout)

  # -- -- --

  def setSpacing(self, spacing):
    self.gridLayout.setSpacing(spacing)
  
  def setContentsMargins(self, left, top, right, bottom):
    self.gridLayout.setContentsMargins(left, top, right, bottom)

  def setAlignment(self, alignment):
    self.gridLayout.setAlignment(alignment)

  # -- -- --

  def newEmpty(self):
    emptyLabel = QLabel(" ")
    if self.expandVertically:
      emptyLabel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
      emptyLabel.setFixedWidth(1)
    else:
      emptyLabel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
      emptyLabel.setFixedHeight(1)
    emptyLabel.setStyleSheet("border: 0px;background-color: transparent;")
    return emptyLabel

  def removeEmpties(self):
    if len(self.layoutRowItems) > 0:
      endLayout = self.layoutRowItems[-1]
      childCount = endLayout.count()
      for x in range(childCount):
        idx = childCount - x - 1
        child = endLayout.itemAt(idx)
        childWidget = child.widget()
        if type(childWidget) == QLabel:
          print(childWidget.text())
        if type(childWidget) == QLabel and childWidget.text() == " ":
          endLayout.removeWidget(childWidget)
          childWidget.deleteLater()

  def addEmpties(self):
    endRow = len(self.layoutRowItems) - 1
    if endRow >= 0:
      endLayout = self.layoutRowItems[endRow]
      endLayoutCount = endLayout.count()
      if endLayoutCount == 0:
        self.layoutRowItems.pop(endRow)
        endLayout.deleteLater()
      elif endLayoutCount < self.gridRes[0]:
        for x in range(endLayoutCount, self.gridRes[0]):
          endLayout.addWidget( self.newEmpty() )

  # -- -- --

  def checkLastRow(self):
    firstWidget = self.layoutRowItems[-1].itemAt(0)
    if firstWidget is not None:
      firstWidget = firstWidget.widget()
    if type(firstWidget) == QLabel and firstWidget.text() == " ":
      curLayout = self.layoutRowItems.pop()
      childCount = curLayout.count()
      for x in range(childCount):
        idx = childCount - x - 1
        child = curLayout.itemAt(idx)
        childWidget = child.widget()
        if type(childWidget) == QLabel and childWidget.text() == " ":
          curLayout.removeWidget(childWidget)
          childWidget.deleteLater()
      curLayout.deleteLater()

  # -- -- --

  def currentLocation(self):
    ret = (0,0)
    icount = len(self.gridItems)
    ret[0] = icount % self.gridRes[0]
    ret[1] = icount // self.gridRes[0]
    return ret
  

  def addItems(self, items):
    if type(items) != list:
      items = [items]

    startIndex = len(self.gridItems)
    self.gridItems += items

    # Wipe empties
    #   I know its not efficient, but I'm gettin some idx update issues
    self.removeEmpties()

    for x, item in enumerate(items):
      # XY location
      idx = startIndex + x
      curLoc = (idx % self.gridRes[0], idx // self.gridRes[0])

      # Check if we need to expand the layout
      if curLoc[0] not in self.layoutItemMap:
        self.layoutItemMap[curLoc[0]] = {}

      if curLoc[1] not in self.layoutItemMap[curLoc[0]]:
        self.layoutItemMap[curLoc[0]][curLoc[1]] = None

      if self.layoutItemMap[curLoc[0]][curLoc[1]] is not None:
        checkItem = self.layoutItemMap[curLoc[0]][curLoc[1]]
        if type(checkItem) == QLabel and checkItem.text() == " ":
          self.layoutItemMap[curLoc[0]][curLoc[1]] = None
          checkItem.deleteLater()

      curHLayout = None
      if curLoc[1] >= len(self.layoutRowItems):
        curHLayout = QHBoxLayout()
        curHLayout.setSpacing(0)
        curHLayout.setContentsMargins(0, 0, 0, 0)
        curHLayout.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        self.gridLayout.addLayout(curHLayout)
        self.layoutRowItems.append(curHLayout)
      else:
        curHLayout = self.layoutRowItems[curLoc[1]]

      # -- -- --

      # Add the item to the layout
      self.layoutItemMap[curLoc[0]][curLoc[1]] = item
      curHLayout.addWidget(item)

      self.itemLayoutMap[item] = (curLoc[0], curLoc[1])
    
    self.addEmpties()


  def removeItem(self, input):
    itemPos = (0,0)
    itemObj = None
    
    # -- -- --

    # Remove item at index
    if type(input) == int:
      input = self.gridItems[input]

    # Remove item at (X,Y) position
    if type(input) == tuple:
      itemPos = input
      itemObj = self.layoutItemMap[itemPos[0]][itemPos[1]]

    # Remove item by object
    else:
      itemObj = input
      itemPos = self.itemLayoutMap[itemObj]
    
    # -- -- --

    # Remove the item from the layout
    if itemObj is None or itemPos is None:
      return
    
    self.layoutItemMap[itemPos[0]][itemPos[1]] = None
    itemIndex = self.gridItems.index(itemObj)
    self.gridItems.pop(itemIndex)
    
    self.layoutRowItems[itemPos[1]].removeWidget(itemObj)
    itemObj.deleteLater()

    # Shift first row items back to fill the gap
    if itemPos[1]+1 < len(self.layoutRowItems):
      for row in range(itemPos[1]+1, len(self.layoutRowItems)):
        if row not in self.layoutItemMap[0]:
          continue;
        firstItem = self.layoutItemMap[0][row]
        toRow = row - 1
        if toRow < 0:
          continue;
        self.layoutRowItems[toRow].addWidget(firstItem)

        
    # Clean up empty rows
    self.checkLastRow()
    
    
    # Rebuild self.layoutItemMap to shifted grid items
    self.layoutItemMap = {}
    for x in range(len(self.gridItems)):
      item = self.gridItems[x]
      curLoc = (x % self.gridRes[0], x // self.gridRes[0])
      #print(f"Item {x} at {curLoc}")
      if curLoc[0] not in self.layoutItemMap:
        self.layoutItemMap[curLoc[0]] = {}
      self.layoutItemMap[curLoc[0]][curLoc[1]] = item
      self.itemLayoutMap[item] = curLoc


    self.addEmpties()
    

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --




# Unit test
if __name__ == "__main__":
  import sys
  from PyQt6.QtWidgets import QApplication, QLabel

  app = QApplication(sys.argv)

  colCount = 3
  rowCount = 4
  

  gridLayoutWidget = QWidget()
  gridLayoutWidget.setWindowTitle("Grid Array Display Widget Test")
  gridLayoutWidget.resize(400, 300)
  gridLayout = QVBoxLayout()


  gridLayoutDisplay = GridArrayDisplayWidget(
              gridItems=[],
              gridRes=[colCount,rowCount],
              expandVertically=False
            )

  gridItems = []

  def clicked(button):
    text = button.text()
    print(f"Delete item: {text}")
    gridLayoutDisplay.removeItem(button)
    idx = gridItems.index(button)
    gridItems.pop(idx)


  def makeButton(label):
    button = QLabel(label)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    button.setFixedHeight(35)
    button.setStyleSheet("border: 1px solid #000000; padding: 3px;")
    button.mousePressEvent = lambda event: clicked(button)
    return button

  gridItems += [ makeButton("Test "+str(x+1)) for x in range(colCount*rowCount + 1) ]
  gridLayoutDisplay.addItems(gridItems)

  # -- -- --

  def addItem():
    newItem = makeButton("Test "+str(len(gridItems)+1))
    gridItems.append(newItem)
    gridLayoutDisplay.addItems(newItem)

  addItemButton = QLabel("Add Item")
  addItemButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
  addItemButton.setFixedHeight(35)
  addItemButton.setStyleSheet("border: 1px solid #000000; padding: 3px;font-weight: bold;font-size: 20px;color: #ffffff;background-color: #656565;")
  addItemButton.mousePressEvent = lambda event: addItem()
  gridLayout.addWidget(addItemButton)

  gridLayout.addWidget(gridLayoutDisplay)
  gridLayoutWidget.setLayout(gridLayout)
  gridLayoutWidget.show()

  sys.exit(app.exec())