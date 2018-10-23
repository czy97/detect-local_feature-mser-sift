import cv2
import glob
import os

def loadImage(filename,flag = 0):
    img = cv2.imread(filename, flag)
    return img


def getMSERegions(img):
    mser = cv2.MSER_create(_min_area=10,_max_area=200)#30 200
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
                          (bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3]), (0, 255, 0), 2)
    if(drawType == 'Ellipse'):
        for i in range(N):
            center = (int(bboxes[i][0] + bboxes[i][2]/2.0),int(bboxes[i][1] + bboxes[i][3]/2.0))
            width = (int(bboxes[i][2]/2.0),int(bboxes[i][3]/2.0))
            cv2.ellipse(img, center, width, 0, 0, 360, (0, 255, 0), 2)
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
    return glob.glob(os.path.join(dir,'*.p[pg]m'))

def siftProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


def processImages(dir,processType = 'mser'):
    '''
    process all images(with postfix .ppm) under a specific dir
    '''
    allFile = getSubfiles(dir)

    seperator = '/'
    if('\\' in allFile[0]):
        seperator = '\\'

    if(processType == 'mser'):
        storeDir = os.path.join(dir,'mser')
        if(not os.path.exists(storeDir)):
            os.makedirs(storeDir)
    elif(processType == 'sift'):
        storeDir = os.path.join(dir, 'sift')
        if (not os.path.exists(storeDir)):
            os.makedirs(storeDir)

    if (processType == 'mser'):
        for filename in allFile:
            imageName = filename.split(seperator)[-1].split('.')[0]
            storeName = imageName + '_processed_mser.jpg'
            storeName = os.path.join(storeDir,storeName)
            img = loadImage(filename, flag=1)
            regions, bboxes = getMSERegions(img)
            img = plotRegions(img, bboxes, drawType='Ellipse')
            cv2.imwrite(storeName, img)
    elif (processType == 'sift'):
        for filename in allFile:
            imageName = filename.split(seperator)[-1].split('.')[0]
            storeName = imageName + '_processed_mser.jpg'
            storeName = os.path.join(storeDir, storeName)
            img = loadImage(filename, flag=1)
            img = siftProcess(img)
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
    # img = loadImage('data/bikes/img6.ppm', flag=1) #452 -> 256
    # regions, bboxes = getMSERegions(img)
    # print(bboxes.shape)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None) #3384 -> 373
    # print(len(kp))


    #
    # # img = plotRegions(img,bboxes, drawType='Rectangle')
    # img = plotRegions(img, bboxes, drawType='Ellipse')
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
