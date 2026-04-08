import zmq
import pickle
import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph
from pyqtgraph import QtWidgets
import numpy as np
import pyqtgraph as pg
import os
# Referenced: https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())
        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)
        self.label = QtGui.QLabel("<font color='red'>Live Reward Updates</font>")
        self.mainbox.layout().addWidget(self.label)
        #  line plot
        self.otherplot = self.canvas.addPlot()
        self.h2 = self.otherplot.plot(pen='y')

        self.ydata = []
        self.x, self.y, self.z = [], [], []
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:8080")
        self.socket.subscribe("PolicyReward")
        self.ydata = []
        self.window_size = 100
        self._update()


    def _update(self):
        top = self.socket.recv_string()
        message = self.socket.recv()
        self.ydata.append(float(pickle.loads(message)))
        if len(self.ydata) > self.window_size:
            length = len(self.ydata)
            data = self.ydata[length-self.window_size:length]
        else:
            data = self.ydata
        self.h2.setData(data)
        QtCore.QTimer.singleShot(1, self._update)

if __name__ == "__main__":
    os.system('clear')
    app = QtWidgets.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
