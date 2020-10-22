#google trans

import googletrans
from  googletrans import Translator



from django.shortcuts import render
import pandas as pd
import numpy as np
import time
import datetime
import pandas as pd
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.tri import Triangulation
import matplotlib

import cv2

import pytesseract
import os
import pandas as pd
import re
from PIL import Image


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import json

from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt

# from .models  import article 
from rest_framework.parsers import JSONParser

from .serializer import ArticleSerializer
clf=None
train_data=None

@csrf_exempt
def train_model(request):
    global clf
    if request.method== 'POST':
        try:
            json_data = json.loads(request.body)
            print(json_data)
            file = json_data['file']
            data= pd.read_csv(file)
            data=data.fillna(0)
            s = data["Birth year"]
            s[s!=0]
            data["Birth year"]=s[s!=0].str.replace("/","").astype(int)
            data=data.fillna(0) 

            data['Birth year'].apply(type)
            data['Uid']=data['Uid'].astype(str).str.replace(' ', '').astype(float)

            s=data['Uid']
            




            X1 = data['Birth year'].values.reshape(-1,1)
            X2 = data['Uid'].values.reshape(-1,1)

            X = np.concatenate((X1,X2),axis=1)
            outliers_fraction = 0.01
            outliers_fraction = 0.01
            xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
            clf = IForest(contamination=outliers_fraction,random_state=0)
            clf.fit(X)
            # predict raw anomaly score
            scores_pred = clf.decision_function(X) * -1

                    
            # prediction of a datapoint category outlier or inlier
            y_pred = clf.predict(X)
            n_inliers = len(y_pred) - np.count_nonzero(y_pred)
            n_outliers = np.count_nonzero(y_pred == 1)
            plt.figure(figsize=(8, 8))
            # copy ofa dataframe
            data1  = data 
            data['outlier'] = y_pred.tolist()


                
            # sales - inlier feature 1,  profit - inlier feature 2
            inliers_Uid = np.array(data['Uid'][data['outlier'] == 0]).reshape(-1,1)
            inliers_Birth_year = np.array(data['Birth year'][data['outlier'] == 0]).reshape(-1,1)
                
            # sales - outlier feature 1, profit - outlier feature 2
            outliers_Uid = data1['Uid'][data1['outlier'] == 1].values.reshape(-1,1)
            outliers_Birth_year  = data1['Birth year'][data1['outlier'] == 1].values.reshape(-1,1)
                    
            print('OUTLIERS: ',n_outliers,'INLIERS: ',n_inliers)

            output={
                'OUTLIERS ':n_outliers,
                'INLIERS ':n_inliers
            }

            return JsonResponse(output)
        except Exception:
            return JsonResponse(Exception,safe=False)
            
@csrf_exempt
def scrap(request):
    global train_data
    if request.method == 'POST':
        try:
            json_data = json.loads(request.body)
            print(json_data)
            folder=json_data['folder']
            arr = os.listdir(folder)

            details = []
            i=0
            for files in arr:
                try:    
                    i=i+1
                    # Initializing data variable
                    name = None
                    gender = None
                    ayear = None
                    uid = None
                    yearline = []
                    genline = []
                    nameline = []
                    text1 = []
                    text2 = []
                    genderStr = '(Female|Male|emale|male|ale|FEMALE|MALE|EMALE)$'
                    # image=cv2.imread('./images/2020-09-11 15_18_41/'+files)
                    # img = Image.open('./images/2020-09-11 15_18_41/'+files)
                    img = Image.open('./invalid/'+files)
                    img = img.convert('RGBA')
                    pix = img.load()

                    for y in range(img.size[1]):
                        for x in range(img.size[0]):
                            if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
                                pix[x, y] = (0, 0, 0, 255)
                            else:
                                pix[x, y] = (255, 255, 255, 255)

                    img.save('temp.png')

                    text = pytesseract.image_to_string(Image.open('temp.png'))



                    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


                    # text =pytesseract.image_to_string(image,config='')

                    lines = text
                    # print(lines)
                    for wordlist in lines.split('\n'):
                        xx = wordlist.split()
                        if [w for w in xx if re.search('(Year|Birth|irth|YoB|YOB:|DOB:|DOB)$', w)]:
                            yearline = wordlist
                            break
                        else:
                            text1.append(wordlist)
                        
                    try:
                        text2 = text.split(yearline, 1)
                    
                    except Exception:
                        pass

                    try:
                        pattern='(?:[0-9]{2}/*-*){2}[0-9]{4}'
                        ayear=re.search(pattern,yearline)
                        ayear  =  ayear.group(0)
                        
                        

                        # print(ayear)
                        if yearline:
                            ayear = dparser.parse(yearline, fuzzy=True).year
                    
                    except Exception:
                        pass

                    try:
                        for wordlist in lines.split('\n'):
                            xx = wordlist.split()
                            if [w for w in xx if re.search(genderStr, w)]:
                                genline = wordlist
                                break

                        if 'Female' in genline or 'FEMALE' in genline:
                            gender = "Female"
                        if 'Male' in genline or 'MALE' in genline:
                            gender = "Male"

                        text2 = text.split(genline, 1)[1]
                    except Exception:
                        pass
                    uid = set()
                    try:
                        newlist = []
                        for xx in text2.split('\n'):
                            newlist.append(xx)
                        newlist = list(filter(lambda x: len(x) > 12, newlist))
                        for no in newlist:
                            # print(no)
                            if re.match("^[0-9 ]+$", no):
                                uid.add(no)
                                break

                    except Exception:
                        pass

                    try:
                        para=lines.split('\n')
                        i=0
                        valid_names=[]
                        for wordlist in lines.split('\n'):

                            i=i+1 
                            prev=''
                            regex = re.compile(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+")
                            # print(regex.search(wordlist))
                            names=regex.search(wordlist)
                            
                            if names:
                                # print(names.group(0))
                                valid_names.append(names.group(0))
                            
                            #     # print(para[i])
                            #     if para[i]:
                            #          print(names.group(0))
                            prev=wordlist
                            
                        
                            
                    
                    except Exception:
                        pass
                    
                    data = {}
                    if gender:
                        data['Gender'] = gender
                        if valid_names:
                            data['Name'] = valid_names[0]
                        else:
                            data['Name'] = None
                        if ayear:
                            data['Birth year'] = ayear
                        else:
                            data['Birth year'] = None
                        if len(list(uid)) >= 1:
                            data['Uid'] = list(uid)[0]
                        else:
                            data['Uid'] = None
                    else:
                        data['Gender'] = None
                        if valid_names:
                            data['Name'] = valid_names[0]
                        else:
                            data['Name'] = None
                        if ayear:
                            data['Birth year'] = ayear
                        else:
                            data['Birth year'] = None
                        if len(list(uid)) > 1:
                            data['Uid'] = list(uid)[0]
                        else:
                            data['Uid'] = None

                
                        
                    



                    

                



                    print("+++++++++++++++++++++++++++++++")
                    print(data['Name'])
                    print("-------------------------------")
                    print(data['Gender'])
                    print("-------------------------------")
                    print(data['Birth year'])
                    print("-------------------------------")
                    print(data['Uid'])
                    print("-------------------------------")
                    print("*************"+files,i)
                    print("a************************************************")
                    details.append(data)
                    print(i+files)
                except Exception:
                    pass

            
            df = pd.DataFrame(details)
            # df.to_csv('valid.csv', index=False)
            output=open('invalid.csv','wb')
            df.to_csv('invalid.csv', index=False)
            train_data=df
            print(df)
        
            return JsonResponse("data successfully read",safe=False)
        except Exception:
            return JsonResponse("error while reading",safe=False)

@csrf_exempt
def predict(request):
    global clf
    if request.method == 'POST':
        try:
            json_data = json.loads(request.body)
            print(json_data)
            file = json_data['file']
            data=train_data
            data=data.fillna(0)

            s =data["Birth year"]
            s[s!=0]
            data["Birth year"]=s[s!=0].apply(lambda x: time.mktime(datetime.datetime.strptime(str(x), "%d/%m/%Y").timetuple()) ) 
            data=data.fillna(0)
            data['Birth year'].apply(type)
            data['Uid']=data['Uid'].astype(str).str.replace(' ', '').astype(float)

            s=data['Uid']





            X1 = data['Birth year'].values.reshape(-1,1)
            X2 = data['Uid'].values.reshape(-1,1)

            X = np.concatenate((X1,X2),axis=1)
                # prediction of a datapoint category outlier or inlier
            y_pred = clf.predict(X)
            n_inliers = len(y_pred) - np.count_nonzero(y_pred)
            n_outliers = np.count_nonzero(y_pred == 1)
            plt.figure(figsize=(8, 8))
            # copy ofa dataframe
            data1  = data 
            data['outlier'] = y_pred.tolist()


                
            # sales - inlier feature 1,  profit - inlier feature 2
            inliers_Uid = np.array(data['Uid'][data['outlier'] == 0]).reshape(-1,1)
            inliers_Birth_year = np.array(data['Birth year'][data['outlier'] == 0]).reshape(-1,1)
                
            # sales - outlier feature 1, profit - outlier feature 2
            outliers_Uid = data1['Uid'][data1['outlier'] == 1].values.reshape(-1,1)
            outliers_Birth_year  = data1['Birth year'][data1['outlier'] == 1].values.reshape(-1,1)
                    
            print('OUTLIERS: ',n_outliers,'INLIERS: ',n_inliers)

            output={
                'OUTLIERS ':n_outliers,
                'INLIERS ':n_inliers
            }

            return JsonResponse(output)
        except Exception:
            return JsonResponse(Exception,safe=False)



@csrf_exempt
def languages(request):
    if request.method == 'GET':
        print(googletrans.LANGUAGES)
        old_dict=googletrans.LANGUAGES
        new_languages = dict([(value, key) for key, value in old_dict.items()]) 
        return JsonResponse(new_languages)


@csrf_exempt
def translate(request):
    if request.method == 'POST':
        translator=Translator()
        lan_param = json.loads(request.body)
        
        trans_text=translator.translate(lan_param["text"],src=lan_param["source"] ,dest=lan_param["destination"])
        print(trans_text.text)
        return JsonResponse(trans_text.text,safe=False)


















        