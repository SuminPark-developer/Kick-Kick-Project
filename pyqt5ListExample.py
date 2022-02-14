import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTreeView, QFileSystemModel, QVBoxLayout
from PyQt5.QtCore import QModelIndex

# https://www.youtube.com/watch?v=1i-7vlxeiwI

class FileSystemView(QWidget):
    def __init__(self, dir_path):
        super().__init__()
        appWidth = 800
        appHeight = 300
        self.setWindowTitle('File System Viewer')
        self.setGeometry(300, 300, appWidth, appHeight)

        self.model = QFileSystemModel()
        self.model.setRootPath(dir_path)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(dirPath))
        self.tree.setColumnWidth(0, 250)
        self.tree.setAlternatingRowColors(True)

        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dirPath = r'C:\Users\user\PycharmProjects\kick-kick-front'
    demo = FileSystemView(dirPath)
    demo.show()
    sys.exit(app.exec_())