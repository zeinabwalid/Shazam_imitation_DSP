from PyQt5 import QtWidgets
from gui import Ui_MainWindow
from PIL import Image
import cv2
from PyQt5.QtWidgets import QFileDialog,QMessageBox
import sys
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout,QTableWidgetItem
from PyQt5 import QtCore, QtWidgets, QtMultimedia
import logging
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd
import librosa
import librosa.display
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from imagededup.methods import PHash
phasher = PHash()
import os
import csv
from math import floor 
from pydub import AudioSegment	
count=0
temp2=[]
for filename in os.listdir("./"):
    count+=1
    temp2=temp2+[filename]
    
if(("tempDir" in temp2)==False):
    os.mkdir('tempDir')
if(("back" in temp2)==False):
    os.mkdir('back')



class ApplicationWindow(QtWidgets.QMainWindow):
    data = pd.read_csv('hash.csv',encoding='latin-1')
    song=[0,0,0]
    same=[0,0,0]
    similarity_arr=np.zeros(len(data))
    songs_out=["" for x in range(11)]
    similarity_out=np.zeros(11)
    j=0
    path1=""
    path2=""
    w=0
    
    
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.pushButton.clicked.connect(self.control)
        self.ui.actionOpenSongOne.triggered.connect(self.loaddata)
        self.ui.actionOpenSongTwo.triggered.connect(self.loaddata2)
        self.ui.horizontalSlider.valueChanged[int].connect(self.read_slider)

        self.ui.widget1.hideAxis('bottom')
        self.ui.widget1.hideAxis('left')
        self.translate = QtCore.QCoreApplication.translate

        self.user=np.empty(11)  
        self.data2 = genfromtxt('data.csv',delimiter=',')
        self.data2=self.data2[1:,1:]
        self.corr=np.empty(90) 
        self.names = pd.read_csv('data.csv')
        self.names = self.names[["filename"]]
        
        


    def read_slider(self,value):
        self.w=value/100
        

    def Conn_Table(self):
        self.loadTable(self.song,self.same)

  
    def loadTable(self, arr1 , arr2):
        for n , value in enumerate(arr1): #load data in first column
            self.ui.tableWidget.setItem(n,0,QTableWidgetItem(str(value)))
            #print(n) index of first col.
            #print(value) value of each index
        for n, value in enumerate(arr2): #load data in second column
            self.ui.tableWidget.setItem(n,1,QTableWidgetItem(str(value)))
        index_MavVal = np.argmax(arr2 , axis =0)    
        self.ui.tableWidget.selectRow(index_MavVal)
    
    def jaro_distance(self,s1, s2) : 
        if (s1 == s2) : 
            return 1.0 
        len1 = len(s1) 
        len2 = len(s2) 
        if (len1 == 0 or len2 == 0) : 
            return 0.0
        max_dist = (max(len(s1), len(s2)) // 2 ) - 1 
        match = 0 
        hash_s1 = [0] * len(s1) 
        hash_s2 = [0] * len(s2)  
        for i in range(len1) : 
            for j in range( max(0, i - max_dist), 
                        min(len2, i + max_dist + 1)) : 
                if (s1[i] == s2[j] and hash_s2[j] == 0) : 
                    hash_s1[i] = 1 
                    hash_s2[j] = 1 
                    match += 1
                    break
        if (match == 0) : 
            return 0.0
        t = 0 
        point = 0 
        for i in range(len1) : 
            if (hash_s1[i]) : 
                while (hash_s2[point] == 0) : 
                    point += 1
                if (s1[i] != s2[point]) : 
                    point += 1
                    t += 1 
                else : 
                    point += 1
            t /= 2 
        return ((match / len1 + match / len2 +
                (match - t) / match ) / 3.0) 

    def similarity_check(self,s1, s2) : 
        jaro_dist = self.jaro_distance(s1, s2)
        if (jaro_dist > 0.7) : 
            prefix = 0
            for i in range(min(len(s1), len(s2))) : 
                if (s1[i] == s2[i]) : 
                    prefix += 1 
                else : 
                    break 
            prefix = min(4, prefix) 
            jaro_dist += 0.1 * prefix * (1 - jaro_dist)

        return jaro_dist

    def load_arr(self,path):
        if path.endswith(".mp3"):
            sound = AudioSegment.from_mp3(path)
            sound.export("back/new.wav", format="wav")
            path = "back/new.wav"
        ###############################
        t1 = 0 #Works in milliseconds
        t2 = 60*1000
        newAudio = AudioSegment.from_wav(path)
        newAudio = newAudio[t1:t2]
        newAudio.export("back/new.wav", format="wav")
        audio_name = "back/new.wav"
        y, sr = librosa.load(audio_name)
        return(y)

    def loaddata(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path1 = filename[0]
        
            
            
    def loaddata2(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path2 = filename[0]
        

    def fn(self):
        
        self.user[0] = np.mean(librosa.feature.chroma_stft(y=self.out, sr=self.sr))
        self.user[1] = np.mean(librosa.feature.spectral_centroid(y=self.out, sr=self.sr))
        self.user[2] = np.mean(librosa.feature.spectral_bandwidth(y=self.out, sr=self.sr))
        self.user[3] = np.mean(librosa.feature.spectral_rolloff(y=self.out, sr=self.sr))
        self.user[4] = np.mean(librosa.feature.zero_crossing_rate(self.out))
    ########################################################
        hop_length = 512
        window_size = 1024
        window = np.hanning(window_size)
        self.out  = librosa.core.spectrum.stft(self.out, n_fft = window_size, hop_length = hop_length, window=window)
        self.out = 2 * np.abs(self.out) / np.sum(window)
        self.user[5] = np.mean(librosa.feature.poly_features(S=self.out, order=0))
        self.user[6] = np.mean(librosa.feature.poly_features(S=self.out, order=1))
        self.user[7] = np.mean(librosa.feature.poly_features(S=self.out, order=2))
        self.user[8] = np.mean(librosa.feature.spectral_flatness(S=self.out))
        self.user[9] = np.mean(librosa.feature.spectral_contrast(S=self.out, sr=self.sr))
        self.user[10] = np.mean(librosa.feature.rms(S=self.out,frame_length=1025))

    def load_arr(self,path):  
        y, sr = librosa.load(path,duration=60)
        self.sr=sr
        return(y)
    def song_name(self):
        
        if (self.path1==""):
            data2 =self.load_arr(self.path2)
            new_song = self.w*data2
        elif(self.path2==""):
            data1 = self.load_arr(self.path1)
            new_song = self.w*data1
        else:
            data1 = self.load_arr(self.path1)
            data2 =self.load_arr(self.path2)
            new_song = (self.w*data1)+((1-self.w)*data2)
        
        hop_length = 512
        window_size = 1024
        window = np.hanning(window_size)
        new_song = np.abs(librosa.core.spectrum.stft(new_song, n_fft = window_size, hop_length = hop_length, 
        window=window))
     
        fig = plt.Figure()
        #canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p=librosa.display.specshow(librosa.amplitude_to_db(new_song, ref=np.max), ax=ax,  y_axis='log', x_axis='time')
        fig.savefig('tempDir/temp.png')
        img = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap('tempDir/temp.png'))
        self.ui.widget1.addItem(img)
        self.ui.widget1.invertY(True)
        Hash = phasher.encode_image(image_array=new_song)
        for i in range(len(self.data)):
            s1= self.data.iloc[i,0]
            s2=Hash
            self.similarity_arr[i]= self.similarity_check(s1,s2)
        index_max = np.sort(np.argpartition(self.similarity_arr, -11)[-11:])
        for i in index_max:
            self.similarity_out[self.j]=self.similarity_arr[i]
            self.songs_out[self.j]=self.data.iloc[i,1]
            self.j+=1
        self.j=0
        self.loadTable(self.songs_out,self.similarity_out)
        self.ui.label7.setText("{}%".format(round(np.max(self.similarity_out)*100,2)))
      
    
    def feature(self):
        if (self.path1==""):
            data2 =self.load_arr(self.path2)
            new_song = self.w*data2
        elif(self.path2==""):
            data1 = self.load_arr(self.path1)
            new_song = self.w*data1
        else:
            data1 = self.load_arr(self.path1)
            data2 =self.load_arr(self.path2)
            new_song = (self.w*data2)+((1-self.w)*data1)

        self.out=new_song
        hop_length = 512
        window_size = 1024
        window = np.hanning(window_size)
        new_song = np.abs(librosa.core.spectrum.stft(new_song, n_fft = window_size, hop_length = hop_length, 
        window=window))
        fig = plt.Figure()
        #canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p=librosa.display.specshow(librosa.amplitude_to_db(new_song, ref=np.max), ax=ax,  y_axis='log', x_axis='time')
        fig.savefig('tempDir/x.png')
        img = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap('tempDir/temp.png'))
        self.ui.widget1.addItem(img)
        self.ui.widget1.invertY(True)

        self.fn()
        for i in range(np.shape(self.data2)[0]):
            self.corr[i]=(1 - (np.linalg.norm(self.data2[i] - self.user) / 1500)) 


        index_max = np.sort(np.argpartition(self.corr, -11)[-11:])
        for i in index_max:
            self.similarity_out[self.j]=self.corr[i]
            self.songs_out[self.j]=self.names.iloc[i,0]
            self.j+=1
        self.j=0

        self.loadTable(self.songs_out,self.similarity_out)
        self.ui.label7.setText("{}%".format(round(np.max(self.similarity_out)*100,2)))

    def control(self):
        comp = self.ui.comboBox.currentText()
        if (comp=="Hash Method"):
            if((self.path1=="") and (self.path2=="")):
                self.error("Please Select song first")
            else:    
                self.song_name()
        if(comp=="Spectrogram Main Features Method"):
            if((self.path1=="") and (self.path2=="")):
                self.error("Please Select song first")
            else:
                self.feature()
                
        if(comp=="Similarity Check Methods :"):
            self.error("Please Select Method")

    def error(self,value):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText(value)  
        msg.setIcon(QMessageBox.Warning)
        x=msg.exec_()   


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
	main()







