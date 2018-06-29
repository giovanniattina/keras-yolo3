import json
import pprint
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage import io
#from skimage.transform import rescale, resizes
from skimage.util import random_noise
import os

def readJson(inputFile):
    jsonData  = open(inputFile).read()

    myData = json.loads(jsonData)


    return myData


def correctJson():
    jsonData = open("images.json").read()
    mydata = json.loads(jsonData)
    for item in mydata:
        item["filename"] = "images/" + str(item["filename"].split("/")[-1:][0])
    
    myFile = open("imagesAnotation.json", "w")
    json.dump(mydata, myFile, sort_keys = True, indent = 4)

def writeFile(jsonData):
    myFile = open("train.txt", "w")
    for item in jsonData:

        aux = []
        aux.append(item["filename"])
        #get items in Json annotations/class/filename
        for annotationsItem in item["annotations"]:

            for anyItem in annotationsItem:
                aux.append(str(annotationsItem[anyItem]))
        if len(aux) > 1:
            aux.append("\n")
            myFile.write(",".join(aux))

def trainFile(jsonData):
    myFile = open("train.txt", "w")

    for item in jsonData:
        print(item)
        imageName = item["filename"]
        toPrint = []
        for annotationsItem in item["annotations"]:
            aux = [ ] # save the data
            #get data
            for anyItem in annotationsItem:
                if anyItem != "class":
                    if(annotationsItem[anyItem] < 0.00 ):
                        aux.append(str(0))
                    else:
                        aux.append(str(annotationsItem[anyItem]))
                elif anyItem == "class":
                    if annotationsItem[anyItem] == "robot":
                        clas = 0
                    elif annotationsItem[anyItem] == "ball":
                        clas = 1
                    aux.append(str(clas))
            toPrint.append(aux)
        aux = ''
        for data in toPrint:
            aux += ",".join(data) + " "
        if aux != "":
            myFile.write(imageName + " " + aux + "\n")
            print(aux)

        
        

def JsonToArray(jsonData):
    myArray = []
    for item in jsonData:
        
        #get items in Json annotations/class/filename 
        for annotationsItem in item["annotations"]:
            
            aux = []
            for anyItem in annotationsItem:
                aux.append(annotationsItem[anyItem])
            aux.append(item["filename"])
            myArray.append(aux)
    
    return np.array(myArray)


def normalizeData(data, height, width):
    for i in range(data.shape[0]):
        
        data[i,1] = float(data[i][1])/height     
        data[i,2] = float(data[i][2])/height
        data[i,3] = float(data[i][3])/height
        data[i,4] = float(data[i][4])/height

    return data


def getData(file):
    myData = pd.DataFrame(normalizeData(JsonToArray(readJson(file)), 480., 640.), columns=[
                          'type', 'height', 'width', 'xM', 'yM', 'path'])
    importante = myData.drop(['type', 'path'], axis=1)
    return importante[['height', 'width', ]].apply(pd.to_numeric)


def getModelTrain(data):
    return KMeans(n_clusters=5, random_state=0).fit(data)


#pprint.pprint(trainFile(readJson("images.json")))


def flipImages(jsonFile):
    myData = readJson(jsonFile)
    allN = []
    for image in myData:

        imageName = "/".join(image["filename"].split("/")[-2:])
        #open image 

        imageArray = io.imread(imageName)
        imageFlip = np.fliplr(imageArray)
        imageName = imageName.split(".")[0]+"flip.jpg"
        annotations = {}
       
        annotations["annotations"] = []

        for info in image["annotations"]:
            annotAux = {}
            annotAux["class"] = info["class"]
            annotAux["height"] = info["height"]
            annotAux["width"] = info["width"]
            annotAux["x"] = 640 - float(info["x"]) - float(info["width"])
            annotAux["y"] = info["y"]

            annotations["annotations"].append(annotAux)
        annotations["class"] = "image"
        annotations["filename"] = imageName
        
        io.imsave(os.path.join(imageName), imageFlip)
        allN.append(annotations)


    myFile = open("imagensFlip.json", "w")
    json.dump(allN, myFile, sort_keys=True, indent=4)
        

def cropImages(jsonFile):
    myData = readJson(jsonFile)
    allN = []
    for image in myData:

        imageName = "/".join(image["filename"].split("/")[-2:])
        #open image

        imageArray = io.imread(imageName)
        print(imageName)
        scale_out = rescale(imageArray, scale=2.0, mode='constant',
                            multichannel=True, anti_aliasing=True)
        scale_in = rescale(scale_out, scale=0.5, mode='constant',
                           anti_aliasing=True, multichannel=True)
       
        imageName = imageName.split(".")[0]+"rescale.jpg"
        annotations = {}
        annotations["annotations"] = []

        for info in image["annotations"]:
            annotAux = {}
            annotAux["class"] = info["class"]
            annotAux["height"] = info["height"]
            annotAux["width"] = info["width"]
            annotAux["x"] = info["x"]
            annotAux["y"] = info["y"]

            annotations["annotations"].append(annotAux)
        annotations["class"] = "image"
        annotations["filename"] = imageName

        io.imsave(os.path.join(imageName), scale_in)
        allN.append(annotations)

    myFile = open("imagensFlip.json", "w")
    json.dump(allN, myFile, sort_keys=True, indent=4)

def disImages(jsonFile):
    myData = readJson(jsonFile)
    allN = []
    for image in myData:

        imageName = "/".join(image["filename"].split("/")[-2:])
        #open image

        imageArray = io.imread(imageName)
        print(imageName)
        randoImage = random_noise(imageArray, mode='gaussian')
        imageName = imageName.split(".")[0]+"Gaussian.jpg"
        annotations = {}
        annotations["annotations"] = []

        for info in image["annotations"]:
            annotAux = {}
            annotAux["class"] = info["class"]
            annotAux["height"] = info["height"]
            annotAux["width"] = info["width"]
            annotAux["x"] = info["x"]
            annotAux["y"] = info["y"]

            annotations["annotations"].append(annotAux)
        annotations["class"] = "image"
        annotations["filename"] = imageName

        io.imsave(os.path.join(imageName), randoImage)
        allN.append(annotations)

    myFile = open("imagensGaussian.json", "w")
    json.dump(allN, myFile, sort_keys=True, indent=4)
'''
def cropImages(jsonFile):
    myData = readJson(jsonFile)
    allN = []
    for image in myData:
        imageName = "/".join(image["filename"].split("/")[
        #open image
        imageArray = io.imread(imageName)
        print(imageName)
        scale_out = rescale(imageArray, scale=2.0, mode='c
                            multichannel=True, anti_aliasi
        scale_in = rescale(scale_out, scale=0.5, mode='con
                           anti_aliasing=True, multichanne
       
        imageName = imageName.split(".")[0]+"rescale.jpg"
        annotations = {}
        annotations["annotations"] = []
        for info in image["annotations"]:
            annotAux = {}
            annotAux["class"] = info["class"]
            annotAux["height"] = info["height"]
            annotAux["width"] = info["width"]
            annotAux["x"] = info["x"]
            annotAux["y"] = info["y"]
            annotations["annotations"].append(annotAux)
        annotations["class"] = "image"
        annotations["filename"] = imageName
        io.imsave(os.path.join(imageName), scale_in)
        allN.append(annotations)
    myFile = open("imagensFlip.json", "w")
    json.dump(allN, myFile, sort_keys=True, indent=4)
'''
def passToInt(jsonFile):
    myData = readJson(jsonFile)
    allN = []
    for image in myData:
        imageName = image["filename"]
        #open image
        annotations = {}
        annotations["annotations"] = []
        for info in image["annotations"]:
            annotAux = {}
            annotAux["class"] = info["class"]
            annotAux["height"] = int(info["height"])
            annotAux["width"] = int(info["width"])
            annotAux["x"] = int(info["x"])
            annotAux["y"] = int(info["y"])
            annotations["annotations"].append(annotAux)

        annotations["class"] = "image"
        annotations["filename"] = imageName
        allN.append(annotations)
    myFile = open("imagensAnoInt.json", "w")
    json.dump(allN, myFile, sort_keys=True, indent=4)


trainFile(readJson("imagensAnoInt.json"))
#passToInt("imagesAnotation.json")



