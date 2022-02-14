from kick_kick_detect import kick_kick_detect

detectionObj = kick_kick_detect(confthres=0.3, nmsthres=0.1)

detectionObj.load()

input = './img_o.jpg'
output = './test_image_predicted.jpg'
result = detectionObj.predict(input, output)

print('result: ', result)