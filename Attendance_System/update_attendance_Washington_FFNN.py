#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import time
tic = time.time()
import os
import numpy as np
import pandas as pd
import torch
import imageio
import torch.nn as nn
import cv2
from PIL import Image
import math
from sklearn.preprocessing import OneHotEncoder
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from datetime import date
import re
# # Functions Definitions

# In[30]:


# Attendance

def label_test_images(cascade,scaleFactor,minNeighbors,dir_list,pred,names,data_dir,save_dir):
  for i in range(np.array(dir_list).size):
    im = np.array(Image.open(data_dir+'/'+dir_list[i]))
    faces_rect = cascade.detectMultiScale(im, scaleFactor=scaleFactor, minNeighbors=minNeighbors)    
    ctr = 0
    for (x, y, _w, _h) in faces_rect:
      cv2.rectangle(im, (x, y), (x+_w, y+_h), (0, 255, 0), 2)
      name = names[pred[3*i + ctr]]
      ctr += 1
      cv2.putText(im, name, (x + 6, y+_h - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
      
    imageio.imwrite(save_dir+'/'+dir_list[i], im)

def update_attendance(pred,names,save_dir,date):
  data = np.unique(pred)
  enc = OneHotEncoder()
  data = np.sum(enc.fit_transform(np.expand_dims(np.append(data,np.arange(n_persons)),1)).toarray()[0:data.size],axis=0)
  #date=str(sys.argv[1]) #'25.05.2020'
  if(os.path.exists(save_dir+'/Attendance_Washinton.xlsx')):
    df = pd.read_excel(save_dir+'/Attendance_Washinton.xlsx', sheet_name='Students')
  else:
    df = pd.DataFrame(names,columns=['Name'])

  df[date] = data  
  df.to_excel(save_dir+"/Attendance_Washinton.xlsx", sheet_name='Students',index=False)  

def hist_equalizer(im):
  img_yuv = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2YUV)
  # equalize the histogram of the Y channel
  img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
  # convert the YUV image back to RGB format
  im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
  return im

def get_aligned_img(img, sF=1.05, mN=1): # input PIL Image
  img = np.array(img)
  img_raw = img.copy() 
  eyes = eye_cascade.detectMultiScale(img,sF,mN)
  gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  eye_1 = (0,0,0,0)
  eye_2 = (0,0,0,0)
  eyes = np.array(eyes)
  possible_eyes = []
  for i in range(len(eyes)):
    if(quadrant_checker(eyes[i][1])):
      possible_eyes.append(i)
  eyes = eyes[possible_eyes]
  if(len(eyes) == 0):
    return cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)

  left_eye = eyes[np.argmin(eyes.T[0])]
  right_eye = eyes[np.argmax(eyes.T[0])]
  cv2.rectangle(img,(left_eye[0], left_eye[1]),(left_eye[0]+left_eye[2], left_eye[1]+left_eye[3]), 1)
  cv2.rectangle(img,(right_eye[0], right_eye[1]),(right_eye[0]+right_eye[2], right_eye[1]+right_eye[3]), 1)
  left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
  left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]

  right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
  right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]

  if(len(left_eye) == 0 or len(right_eye) == 0):
    return cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)

  cv2.circle(img, left_eye_center, 1, (255, 0, 0) , 1)
  cv2.circle(img, right_eye_center, 1, (255, 0, 0) , 1)
  cv2.line(img,right_eye_center, left_eye_center,(67,67,67),1)
  if(left_eye_y > right_eye_y):
    point_3rd = (right_eye_x, left_eye_y)
    direction = -1 #rotate same direction to clock

  else:
    point_3rd = (left_eye_x, right_eye_y)
    direction = 1 #rotate inverse direction of clock

  cv2.line(img,right_eye_center, left_eye_center,(67,67,67),1)
  cv2.line(img,left_eye_center, point_3rd,(67,67,67),1)
  cv2.line(img,right_eye_center, point_3rd,(67,67,67),1)
  
  a = euclidean_distance(left_eye_center, point_3rd)
  b = euclidean_distance(right_eye_center, left_eye_center)
  c = euclidean_distance(right_eye_center, point_3rd)

  if(b == 0 or c == 0):
    #new_img = Image.fromarray(img_raw)
    gray_new_img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    #return gray_new_img

  else:
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = np.arccos(cos_a)
    angle = (angle * 180) / math.pi

    if direction == -1:
      angle = 90 - angle  

    new_img = Image.fromarray(img_raw)
    new_img = np.array(new_img.rotate(direction * angle))
    gray_new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)

    for i in range(gray_new_img.shape[0]):
      if(gray_new_img[i][0] == 0):
        ctr = 0
        while(np.allclose(gray_new_img[i][ctr],0)):
          ctr += 1
        gray_new_img[i][0:ctr] = np.ones(gray_new_img[i][0:ctr].size)*gray_new_img[i][ctr]
      if(gray_new_img[i][-1] == 0):
        ctr = 0
        while(np.allclose(gray_new_img[i][-1-ctr],0)):
          ctr += 1
        gray_new_img[i][gray_new_img.shape[1]-ctr:] = np.ones(gray_new_img[i][gray_new_img.shape[1]-ctr:].size)*gray_new_img[i][-1-ctr]

  return gray_new_img

def quadrant_checker(y):
  if(y<64 and y>0):
    return True
  else:
    return False

def euclideanDistance(x,y):
  distance = 0
  for i in range(x.size):
    distance += (x[i] - y[i])**2
  return np.sqrt(distance)

def euclidean_distance(a, b):
  x1 = a[0]; y1 = a[1]
  x2 = b[0]; y2 = b[1]
  return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def first_comes_first(arr,x):
    arr[1:] = arr[0:-1]
    arr[0] = x
    return arr

def kNN(test_data,train_data,train_class,k):
  test_pred = np.zeros(test_data.shape[0])
  for i in range(test_data.shape[0]):
    dist_arr = np.asarray(np.zeros(k))
    nearest = np.asarray(np.zeros(k))
    ret_arr = np.array([])
    for j in range(k):
      dist_arr[j] = euclideanDistance(train_data[j],test_data[i])
      nearest[j]=j

    for j in range(train_data.shape[0]):
      temp_dist = euclideanDistance(test_data[i],train_data[j])
      if(temp_dist <= np.amin(dist_arr)):
        dist_arr = first_comes_first(dist_arr,temp_dist)
        nearest = first_comes_first(nearest,j)

    for j in (nearest):
      ret_arr = np.append(ret_arr,train_class[int(j)])
    ret_arr = np.array([int(q) for q in ret_arr])
    test_pred[i] = np.argmax(np.bincount(ret_arr))

  return test_pred

def run_SVM(X_train_scaled,X_test_scaled,Y_train,cv=5,kernel='rbf',gamma=0.001,C=10):
#  params_grid = [{'kernel': [kernel], 'gamma': [gamma],'C': [C]}]
  svm_model = SVC(kernel=kernel,C=C,gamma=gamma) #GridSearchCV(SVC(), params_grid, cv=cv)
  svm_model.fit(X_train_scaled, Y_train)
  Y_pred = svm_model.predict(X_test_scaled)
  
  return Y_pred, svm_model


def accuracy(yhat,Y_test):
  ctr = 0.0
  for i in range(Y_test.size):  
    if(yhat[i] == Y_test[i]):
      ctr += 1.0

  return ((ctr/Y_test.size)*100)
  

def get_cropped_faces(cascade, eye_cascade, dir_list, path, crop_dims=[28,116,20,108], w=128, h=128, scaleFactor = 1.1, minNeighbors=1, sF = 1.05, mN = 1):
  test_images = np.zeros(h*w)
  for i in range(len(dir_list)):
    im = np.array(Image.open(path+'/'+dir_list[i]))
    faces_rect = cascade.detectMultiScale(im, scaleFactor=scaleFactor, minNeighbors=minNeighbors)  
    for (x, y, _w, _h) in faces_rect:
      crop_im = im[y:y+_h, x:x+_w]
      crop_im = cv2.resize(crop_im, (w,h), interpolation = cv2.INTER_AREA)
      crop_im = crop_im[crop_dims[0]:crop_dims[1],crop_dims[2]:crop_dims[3]]
      crop_im = cv2.resize(crop_im, (w,h), interpolation = cv2.INTER_AREA)
      crop_im = hist_equalizer(crop_im)
      crop_im = get_aligned_img(crop_im,sF,mN)
#      crop_im = cv2.cvtColor(crop_im, cv2.COLOR_RGB2GRAY)
      temp = np.array(crop_im)
      
      test_images = np.row_stack((test_images,temp.ravel()))
    
  return test_images[1:]


# # Class Definition

# In[10]:


# simply define a silu function
def hakuna_matata(x,x0 = 0,a=1):
  return 2/(1+torch.exp(-a*(x-x0))) - 1 # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class Hakuna_Matata(nn.Module):
  def __init__(self,x0=0,a=1):
    super().__init__() # init the base class
    self.a = a
    self.x0 = x0

  def forward(self, ip):
    return hakuna_matata(ip,self.x0,self.a) # simply apply already implemented SiLU

class RegularizedLinear(nn.Linear):
    def __init__(self, *args, ar_weight=1e-3, l1_weight=1e-3, l2_weight=2, **kwargs):
        super(RegularizedLinear, self).__init__(*args, **kwargs)
        #self.ar_weight = ar_weight
        #self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self._losses = {}

    def forward(self, input):
        output = super(RegularizedLinear, self).forward(input)
        #self._losses['activity_regularization'] = (output * output).sum() * self.ar_weight
        #self._losses['l1_weight_regularization'] = torch.abs(self.weight).sum() * self.l1_weight
        self._losses['l2_weight_regularization'] = torch.abs(torch.mul(self.weight,self.weight)).sum() * self.l2_weight
        return output

class FFNNetwork_Regularized(nn.Module):
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.scale_factor = 1
    self.shift_param = -1
    self.net = nn.Sequential(
        #nn.Dropout(0.2),
        RegularizedLinear(n_components,250), 
        nn.BatchNorm1d(250),
        Hakuna_Matata(self.shift_param,self.scale_factor),
        RegularizedLinear(250,150), 
        nn.BatchNorm1d(150),
        nn.Dropout(0.2),
        Hakuna_Matata(self.shift_param,self.scale_factor),
        RegularizedLinear(150,48), 
        nn.BatchNorm1d(48),
        Hakuna_Matata(self.shift_param,self.scale_factor),
        RegularizedLinear(48, n_persons), 
        nn.BatchNorm1d(n_persons),
        Hakuna_Matata(self.shift_param,self.scale_factor),
        nn.Softmax()
    )

  def forward(self, X):
    return self.net(X)

  def softmax(self,x):
    return torch.exp(x)/torch.sum(torch.exp(x))

  def cross_entropy(self,pred,label):
    yl=torch.mul(pred,label)
    yl=yl[yl!=0]
    yl=-torch.log(yl)
    yl=torch.mean(yl)
    return yl

  def accuracy(self,y_hat, y):
    pred = torch.argmax(y_hat, dim=1)
    return (pred == y).float().mean()

  def predict(self, X):
    Y_pred = self.forward(X)
    return np.array(Y_pred).squeeze()
  
  def accuracy_n(self,y_hat, y,topk=(1,)):
    maxk = max(topk)
    batch_size = y.size(0)

    _, pred = y_hat.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
  
    return res


# # Variables Definition

# In[7]:


path = "/media/kiran/New Volume/SEM_6/Machine Learning for Signal Processing/Course Project/FINAL/SC17B106_SC17B150"
folder = "/Washington_Dataset"
data_folder = "/Test_images"
save_folder = "/Labelled_images"
M = np.loadtxt(path+folder+'/M_Washington.txt')
C = np.loadtxt(path+folder+'/C_Washington.txt')
X_train = np.loadtxt(path+folder+'/X_train_Washington.txt')
Y_train = np.loadtxt(path+folder+'/Y_train_Washington.txt')
scaler = StandardScaler()#load(open(path+'/new/scaler.pkl', 'rb'))
weightage = np.array([0.3,0.3,0.4])#np.loadtxt(path+'/new/weightage.txt')
n_components = C.shape[0]

n_persons = 24
names = np.array([str(i) for i in np.arange(25)[1:]])


# # Reading Testing Samples

# In[15]:


eye_cascade = cv2.CascadeClassifier(path+folder+'/haarcascade_eye.xml')
haar_cascade_face = cv2.CascadeClassifier(path+folder+'/haarcascade_frontalface_alt.xml')
scaleFactor = 1.1
minNeighbors = 1
form = '.tga'
crop_dims=[28,116,20,108]
dir_list = os.listdir(path+folder+data_folder) 
w=128
h=128

#Y_test = np.array([])
X_test = np.array([])
sF = 1.07
mN = 1
test_images = get_cropped_faces(haar_cascade_face,eye_cascade, dir_list, path=path+folder+data_folder, crop_dims=[28,116,20,108],scaleFactor=1.2, minNeighbors=5,sF=1.07,mN=1)
X_test = np.matmul(test_images-M,C.T)    


# # Classification

# In[18]:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

FFNN_model = FFNNetwork_Regularized()
FFNN_model.load_state_dict(torch.load(path+folder+"/CLASS_FFNNetwork_Washington_91.txt",  map_location=lambda storage, loc: storage)) 
FFNN_model.eval()
X_test_scaled = torch.tensor(X_test_scaled).float()
with torch.no_grad():
  pred = FFNN_model.predict(X_test_scaled)

Y_pred_FFNN = np.argmax(pred,axis=1)


# # Marking Attendance
if(not os.path.exists(path+folder+save_folder)):
  os.mkdir(path+folder+save_folder)

# In[32]:
if(len(sys.argv) == 3):
  arr = np.array([ str(sys.argv[1]), str(sys.argv[2]) ])
  today = arr[np.argmax( np.array([ len(str(sys.argv[1])), len(str(sys.argv[2])) ]) )]
  if(str(arr[np.argmin( np.array([ len(str(sys.argv[1])), len(str(sys.argv[2])) ]) )] ) == '1'):
    save_folder = '/'+today
    if(not os.path.exists(path+folder+save_folder)):
      os.mkdir(path+folder+save_folder)

elif(len(sys.argv) == 2):
  if(str(sys.argv[1]) == '1'):
    today = date.today()
    today = today.strftime("%d.%m.%Y")
    today = str(today)
    save_folder = '/'+today
    if(not os.path.exists(path+folder+save_folder)):
      os.mkdir(path+folder+save_folder)
  else:
    today = str(sys.argv[1])
  
else:
  today = date.today()
  today = today.strftime("%d.%m.%Y")
  today = str(today)

dir_list = os.listdir(path+folder+data_folder) 

if(save_folder == "/Labelled_images"): # if default folder, then clear all previous files and replace them with the latest output files
  for root, dirs, files in os.walk(save_folder):
    for file in files:
      os.remove(os.path.join(root, file))

label_test_images(haar_cascade_face,1.2,5,dir_list,Y_pred_FFNN,names,path+folder+data_folder,path+folder+save_folder)

update_attendance(Y_pred_FFNN,names,path+folder,date=today)
toc = time.time()
print("Total Time for Attendance Updation : {}s".format(toc-tic))
