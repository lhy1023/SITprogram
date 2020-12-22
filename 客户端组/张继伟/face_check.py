# encoding:utf-8
import urllib.request
import cv2
import ast
import base64
import urllib.parse
from urllib.parse import urlencode
 
'''
人脸检测与属性分析
'''
#openCV导入图片
filename = '3.jpg'
img = cv2.imread(filename)
#对图片进行base64编码
f = open(filename,'rb')
img_test = base64.b64encode(f.read())
#请求地址
request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
#请求参数
params={'image':''+str(img_test,'utf-8')+'','image_type':'BASE64','face_field':'age,beauty,faceshape,gender,glasses,landmark'}
#对base64数据进行urlencode处理
params=urlencode(params)
#生成的access_token
access_token = '24.6b3139249f9574755b80c4faf75bb3a2.2592000.1600775236.282335-22189693'
request_url = request_url + "?access_token=" + access_token
request = urllib.request.Request(url=request_url,data=params.encode("utf-8"))
request.add_header('Content-Type', 'application/json')
response = urllib.request.urlopen(request)
content = response.read()
if content:
    print(content)
content = content.decode("utf-8")
content = ast.literal_eval(content)
#描绘72个特征点部分
test_pic=content['result']['face_list'][0]['landmark72']
for x in test_pic:
    cv2.circle(img, (int(x['x']),int(x['y'])), 1, (255, 255, 0), 0,8)
#人脸框选部分
left_top = (int(content['result']['face_list'][0]['location']['left']), int(content['result']['face_list'][0]['location']['top']))
right_bottom = (int(left_top[0] + content['result']['face_list'][0]['location']['width']),int(left_top[1] + content['result']['face_list'][0]['location']['height']))
cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
