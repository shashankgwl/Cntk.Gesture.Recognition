import cv2 as cv
import os
import glob2 as g

path = "D:\Shashank\One Drive Business\OneDrive - Microsoft\SHASHANK\learning\ML\CNTK\data\signature\{0}\\train"
pathTest = "D:\Shashank\One Drive Business\OneDrive - Microsoft\SHASHANK\learning\ML\CNTK\data\signature\{0}\\test"
num_of_train_images = 1000

def get_labels(labfor):
    if labfor ==1:
        return "0 1 0 0"
    elif labfor == 2:
        return "0 0 1 0"
    elif labfor == 3:
        return "0 0 0 1"
    
def get_image_features(image):
    finalimgstr = ""
    ndarray= cv.imread(image,cv.IMREAD_GRAYSCALE)
    print(ndarray.shape)
    print(ndarray.size)
    #img = img.reshape(41,50)
    finalimgstr=' '.join(ndarray.flatten().astype(str))
    #finalimgstr = ' '.join(map(str,ndarray)).replace('\n','').replace('[','').replace(']','')
    #for row in ndarray:
    #    finalimgstr += ' '.join(row.astype(str))
    
    print (len(finalimgstr.split(' ')))
    return finalimgstr
        
 
def create_train_file(tt):
    counter = 1 
    #              "|labels {0} |features {1}\n"
    labelFeature = "|labels {0} |features {1}\n"
    txtfile=""
    txtfile = "../../train_cntktxt.txt" if tt=="train" else "../../test_cntktxt.txt"
    while counter<=3:        
        actualpath = path.format(counter) if tt=="train" else pathTest.format(counter)
        print("checking path {0}".format(actualpath))
        if not (os.path.exists(actualpath)):
            print("folder {0} doesn't exist".format(actualpath))
            break
        print("switching to {0}".format(actualpath))        
        os.chdir(actualpath)
        imagesfiles = g.glob(os.path.join(actualpath,"*.png"))
        global showfile
        with open(txtfile,"a+") as f:
            for file in imagesfiles:
                print("working on file {0}".format(file))
                lbls = get_labels(counter)
                features = get_image_features(file)            
                print("writing finalImageData")
                f.write(labelFeature.format(lbls,features)) 
                showfile=file
        counter+=1
        f.close()


def create_train_file_201B(tt):
    counter = 0 
    #              "|labels {0} |features {1}\n"
    labelFeature = "{0}\t{1}\n"
    txtfile=""
    txtfile = "../../train_201B.txt" if tt=="train" else "../../test_201B.txt"
    while counter<3:        
        actualpath = path.format(counter+1) if tt=="train" else pathTest.format(counter+1)
        print("checking path {0}".format(actualpath))
        if not (os.path.exists(actualpath)):
            print("folder {0} doesn't exist".format(actualpath))
            break
        print("switching to {0}".format(actualpath))        
        os.chdir(actualpath)
        imagesfiles = g.glob(os.path.join(actualpath,"*.png"))
        global showfile
        with open(txtfile,"a+") as f:
            for file in imagesfiles:
                print("working on file {0}".format(file))
                lbls = str(counter)
                features = os.path.abspath(file)            
                print("writing finalImageData")
                f.write(labelFeature.format(features,lbls)) 
                showfile=file
        counter+=1
        f.close()
 
print("changing dir")
#os.chdir('D:\Shashank\One Drive Business\OneDrive - Microsoft\SHASHANK\learning\ML\data\signature')       
#create_train_file("test")
create_train_file_201B("train")

