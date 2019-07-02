import numpy as np
import cv2
import os
def get_sample(img,imagesdir):
    num = ['041','130','050','051','140']
    #Label = np.random.randint(0,5)
    #np.random.shuffle(num)
    #print(Label)
    #print(num)
    #testdir = "D:/apeopleD/gen/"
    txtname = 'test.txt'
    testimage = 'H.jpg'
    #img = cv2.imread(testimage)
    resNew = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
    tempImage = testimage.split('.')
    print(tempImage)
    f=open(txtname, "w")

    for i in range(36):
        Label = np.random.randint(0,5)
        np.random.shuffle(num)
        if i<10:
            tempImage[0] = str(Label)+ "_0" + str(i) + "_" + num[Label] + "." + "png"
        else:
            tempImage[0] = str(Label)+ "_" + str(i) + "_" + num[Label] + "." + "png" 
        temp = tempImage[0] + " " + str(Label) + " " + num[Label] + " " + str(Label) + " " + num[Label] + '\n'
        cv2.imwrite(imagesdir + tempImage[0],resNew)
        f.write(temp)
    f.close()



