import cv2
import glob
import os

def loadImage(filename,flag = 0):
    img = cv2.imread(filename, flag)
    return img


def getMSERegions(img):
    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(img)
    return regions, bboxes


def plotRegions(img,bboxes,drawType = 'Ellipse'):
    '''
    plot MSERegions
    default draw the bbox using Ellipse figure
    '''
    N = (bboxes.shape)[0]
    if(drawType == 'Rectangle'):
        for i in range(N):
            cv2.rectangle(img, (bboxes[i][0], bboxes[i][1]),
                          (bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3]), (0, 255, 0), 3)
    if(drawType == 'Ellipse'):
        for i in range(N):
            center = (int(bboxes[i][0] + bboxes[i][2]/2.0),int(bboxes[i][1] + bboxes[i][3]/2.0))
            width = (int(bboxes[i][2]/2.0),int(bboxes[i][3]/2.0))
            cv2.ellipse(img, center, width, 0, 0, 360, (0, 255, 0), 3)
    return img


def colorRegion(img, region):
    img[region[:, 1], region[:, 0], 0] = 0
    img[region[:, 1], region[:, 0], 1] = 0
    img[region[:, 1], region[:, 0], 2] = 0
    return img
def getSundirs(dir):
    subdirs = [os.path.join(dir,val) for val in os.listdir(dir)]
    return subdirs
def getSubfiles(dir):
    return glob.glob(os.path.join(dir,'*.ppm'))

def processImages(dir,processType = 'mser'):
    '''
    process all images(with postfix .ppm) under a specific dir
    '''
    allFile = getSubfiles(dir)
    if (processType == 'mser'):
        for filename in allFile:
            imageName = filename.split('.')[0]
            storeName = imageName + '_processed_mser.jpg'
            img = loadImage(filename, flag=1)
            regions, bboxes = getMSERegions(img)
            img = plotRegions(img, bboxes, drawType='Ellipse')
            cv2.imwrite(storeName, img)
    else:
        for filename in allFile:
            imageName = filename.split('.')[0]
            storeName = imageName + '_processed_sift.jpg'
            img = loadImage(filename, flag=1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray, None)
            cv2.drawKeypoints(img, kp, img)
            cv2.imwrite(storeName, img)

def processAllImages(dir,processType = 'mser'):
    '''
    process all images(with postfix .ppm) under a specific dir's subdirs
    processType can be mser or sift
    '''
    allDirs = getSundirs(dir)
    for dir_val in allDirs:
        processImages(dir_val,processType)


if __name__ == '__main__':
    # img = loadImage('data/trees/img1.ppm', flag=1)
    # regions, bboxes = getMSERegions(img)
    #
    # img = plotRegions(img,bboxes[1].reshape(1,4), drawType='Rectangle')
    # img = plotRegions(img, bboxes[1].reshape(1,4), drawType='Ellipse')
    #
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # processAllImage('data\\trees\\')

    # subdirs = getSundirs('data\\')
    # print(subdirs)
    # print(getSubfiles(subdirs[0]))

    # img = loadImage('data/trees/img1.ppm', flag=1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None)
    #
    # img = cv2.drawKeypoints(gray, kp)
    #
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    processAllImages('data', processType='sift')