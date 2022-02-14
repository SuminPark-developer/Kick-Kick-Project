from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, cv2, numpy, time

from kick_kick_detect import kick_kick_detect
import datetime
import threading

class Worker(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name            # thread 이름 지정

    def run(self, fn, input, output):
        print("sub thread start ", threading.currentThread().getName())
        fn(input, output)
        print("sub thread end ", threading.currentThread().getName())


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScooterHelmetDetector")
        self.setGeometry(0, 0, 1800, 800)
        self.initUI()

        self.detectionObj = kick_kick_detect(confthres=0.3, nmsthres=0.1)
        self.detectionObj.load()

    def initUI(self):
        # self.cpt = cv2.VideoCapture(0) # Cam 사용일 경우
        self.cpt = cv2.VideoCapture('./주행영상.mp4') # 다운로드 된 영상일 경우

        self.fps = 24
        self.sens = 300
        _,self.img_o = self.cpt.read()
        self.img_o = cv2.cvtColor(self.img_o, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('img_o.jpg', self.img_o)

        self.cnt = 0

        self.frame = QLabel(self)
        self.frame.resize(850, 650)
        self.frame.setScaledContents(True) # cam 사이즈가 화면에 맞게 됨.
        self.frame.move(50, 50)

        self.btn_on = QPushButton("켜기", self)
        self.btn_on.resize(100, 25)
        self.btn_on.move(50, 750)
        self.btn_on.clicked.connect(self.start)

        self.btn_off = QPushButton("끄기", self)
        self.btn_off.resize(100, 25)
        self.btn_off.move(5+100+5+50, 750)
        self.btn_off.clicked.connect(self.stop)

        # 테이블 추가
        self.table = QTableWidget(self)
        self.table.resize(580, 400)
        self.table.move(1000, 50)
        self.table.setRowCount(0)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['날짜', '시각', '킥보드 검출수', '헬멧 검출수', '최종 검출'])
        self.table.setColumnWidth(0, round(self.table.width() / 5 - 1))
        self.table.setColumnWidth(1, round(self.table.width() / 5 - 1))
        self.table.setColumnWidth(2, round(self.table.width() / 5 - 1))
        self.table.setColumnWidth(3, round(self.table.width() / 5 - 1))
        self.table.setColumnWidth(4, round(self.table.width() / 5))

        self.imageLabel = QLabel(self)
        self.imageLabel_width = 580
        self.imageLabel_height = 300
        self.imageLabel.resize(580, 300)
        self.imageLabel.move(1000, 470)

        self.prt = QLabel(self)
        self.prt.resize(400, 25)
        self.prt.move(5+105+105+50, 750)

        self.sldr = QSlider(Qt.Horizontal,self) # Qt.Vertical 하면 세로 조절 축 생김.
        self.sldr.resize(100,25)
        self.sldr.move(5+105+105+200+200+50, 750)
        self.sldr.setMinimum(1)
        self.sldr.setMaximum(30)
        self.sldr.setValue(24)
        self.sldr.valueChanged.connect(self.setFps)

        self.sldr1 = QSlider(Qt.Horizontal,self)
        self.sldr1.resize(100,25)
        self.sldr1.move(5+105+105+200+50+200+105, 750)
        self.sldr1.setMinimum(50)
        self.sldr1.setMaximum(500)
        self.sldr1.setValue(300)
        self.sldr1.valueChanged.connect(self.setSens)
        self.show()

    # 왼쪽 슬라이더(프레임 조절)
    def setFps(self):
        self.fps = self.sldr.value()
        self.prt.setText("FPS "+str(self.fps)+"로 조정!")
        self.timer.stop()
        self.timer.start(int(1000/self.fps))

    # 오른쪽 슬라이더(감도 조절) - 감도를 높일수록 크게 움직여야 검출되는 듯.
    def setSens(self):
        self.sens=self.sldr1.value()
        self.prt.setText("감도 "+str(self.sens)+"로 조정!")

    # 켜기 버튼 클릭 시
    def start(self):
        # self.codec = cv2.VideoWriter_fourcc(*'XVID')
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot) # nextFrameSlot메소드 실행.
        self.timer.start(int(1000/self.fps)) # 1000 = 1초. => 즉 초당 24프레임으로 전달하겠다.


    def nextFrameSlot(self):
        _, cam = self.cpt.read() #cam=
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB) #bgr rgb로 변환
        # cam = cv2.flip(cam, 0) # 캠 반전
        self.img_p = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY) # rgb gray로 변환
        cv2.imwrite('img_p.jpg', self.img_p) # 이미지 쓰기

        ######################################## 여기에 모듈 추가하면 될 듯 + pyqt5 리스트뷰

        self.compare(self.img_o, self.img_p) # 전 이미지와 비교
        self.img_o = self.img_p.copy()
        # 캠을 보여줌.
        img = QImage(cam, cam.shape[1], cam.shape[0], QImage.Format_RGB888) #input, width, heigth, format
        pix = QPixmap.fromImage(img)
        self.frame.setPixmap(pix)

    # 끄기 버튼 클릭 시
    def stop(self):
        self.frame.setPixmap(QPixmap.fromImage(QImage()))
        self.timer.stop()


    def compare(self,img_o, img_p):
        err = numpy.sum((img_o.astype("float")-img_p.astype("float"))**2)
        err /= float(img_o.shape[0]*img_p.shape[1])
        if(err>=self.sens):
            t = time.localtime()
            self.prt.setText("{}-{}-{} {}:{}:{} 움직임 감지!".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour,t.tm_min,t.tm_sec))

            input = './img_p.jpg'
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            filename = "_".join(["predicted", suffix])
            # print(filename)
            output = "./" + filename + ".jpg"
            # output = './test_image_predicted.jpg'

            thread = Worker(filename)

            def thread_predict(input, output):
                result = self.detectionObj.predict(input, output)

                # if result == 1:
                #     print("HAHA")
                # elif result == 0:
                #     print("zzz")

                # print('####################### result: %s #######################' %(result))
                self.addRow(result, output)

            thread.run(thread_predict, input, output)


    def addRow(self, result, output):
        if result['detected'] >= 1:
            self.imageLabel.setPixmap(QPixmap(output).scaled(self.imageLabel_width, self.imageLabel_height))
            rowPosition = self.table.rowCount()
            self.table.insertRow(rowPosition)
            t = time.localtime()
            self.table.setItem(rowPosition, 0, QTableWidgetItem("{}-{}-{}".format(t.tm_year, t.tm_mon, t.tm_mday)))
            self.table.setItem(rowPosition, 1, QTableWidgetItem("{}:{}:{}".format(t.tm_hour, t.tm_min, t.tm_sec)))
            kick_scooter_len = result['kick_scooter_len']
            helmet_len = result['helmet_len']

            print(result['kick_scooter_len'], result['helmet_len'])
            self.table.setItem(rowPosition, 2, QTableWidgetItem(str(kick_scooter_len))) # kick_scooter_len
            self.table.setItem(rowPosition, 3, QTableWidgetItem(str(helmet_len))) # helmet_len

            if kick_scooter_len > helmet_len:
                self.table.setItem(rowPosition, 4, QTableWidgetItem('최종 검출 됨')) # helmet_len
            
            # res += "착용 : {}건      ".format(str(result))
            # detail += "helmet {}".format(str(label['helmet']))
        #elif result == 0:
            #res += "착용 : {}건  ".format(str(result))
            # detail += "nonhelmet {}".format(str(label['nonhelmet']))
        # self.table.setItem(rowPosition, 2, QTableWidgetItem(res))
        # self.table.setItem(rowPosition, 4, QTableWidgetItem(detail))
        self.table.scrollToBottom()

        # global recent_prt_sec
        # recent_prt_sec = t.tm_sec
        # global recent_prt_min
        # recent_prt_min = t.tm_min
        # print(recent_prt_sec, recent_prt_min)


app = QApplication(sys.argv)
w = Example()
sys.exit(app.exec_())