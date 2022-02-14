# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from flask import Flask, request, Response, jsonify
from numpy.core.arrayprint import repr_format
import jsonpickle
#import binascii
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image


class kick_kick_detect:
  def __init__(self, 
                  confthres=0.3, 
                  nmsthres=0.1, 
                  labelsPath="yolo_v3/obj.names", 
                  cfgpath="yolo_v3/yolov3_kick.cfg",
                  wpath="yolo_v3/yolov3_kick_last.weights"):
    self.yolo_path = './'
    self.confthres = confthres
    self.nmsthres = nmsthres
    self.labelsPath=labelsPath
    self.cfgpath=cfgpath
    self.wpath=wpath

  def get_labels(self, labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath = os.path.sep.join([self.yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")

    return LABELS

  def get_colors(self, LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    return COLORS

  def get_weights(self, weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([self.yolo_path, weights_path])
    return weightsPath

  def get_config(self, config_path):
      configPath = os.path.sep.join([self.yolo_path, config_path])
      return configPath

  def load_model(self, configpath, weightspath):
      # load our YOLO object detector trained on COCO dataset (80 classes)
      print("[INFO] loading YOLO from disk...")
      net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
      return net

  def image_to_byte_array(self, image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

  def get_predection(self, image, net, LABELS, COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    
    end = time.time()

    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
      # loop over each of the detections
      for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        # print(scores)
        classID = np.argmax(scores)
        # print(classID)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > self.confthres:
          # scale the bounding box coordinates back relative to the
          # size of the image, keeping in mind that YOLO actually
          # returns the center (x, y)-coordinates of the bounding
          # box followed by the boxes' width and height
          box = detection[0:4] * np.array([W, H, W, H])
          (centerX, centerY, width, height) = box.astype("int")

          # use the center (x, y)-coordinates to derive the top and
          # and left corner of the bounding box
          x = int(centerX - (width / 2))
          y = int(centerY - (height / 2))

          # update our list of bounding box coordinates, confidences,
          # and class IDs
          boxes.append([x, y, int(width), int(height)])
          confidences.append(float(confidence))
          classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confthres, self.nmsthres)

    kick_scooter_len = 0
    helmet_len = 0
    # ensure at least one detection exists
    if len(idxs) > 0:
      # loop over the indexes we are keeping
      for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

        if LABELS[classIDs[i]] == 'kick scooter':
          kick_scooter_len += 1
        else:
          helmet_len += 1

        # print(boxes)
        # print(classIDs)
        cv2.putText(image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return {
      'image': image,
      'detected': len(idxs),
      'kick_scooter_len': kick_scooter_len,
      'helmet_len': helmet_len
    }

  def load(self):
    self.Lables   = self.get_labels(self.labelsPath)
    self.CFG      = self.get_config(self.cfgpath)
    self.Weights  = self.get_weights(self.wpath)
    self.nets     = self.load_model(self.CFG, self.Weights)
    self.Colors   = self.get_colors(self.Lables)

  def predict(self, input_img, output_img):
    img = Image.open(input_img)
    npimg = np.array(img)
    image = npimg.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predicted_result = self.get_predection(image, self.nets, self.Lables, self.Colors)
    predicted_image = predicted_result['image']

    image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
    np_img = Image.fromarray(image)
    np_img.save(output_img)

    return predicted_result