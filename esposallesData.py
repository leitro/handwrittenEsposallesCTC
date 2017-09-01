import cv2
import numpy as np
import string

'''
Total number of the textline-based datasets:
    training data: 2759
    validation data: 311
    test data: 757

Total number of the word-based datasets:
    training data: 28346
    validation data: 3155
    test data: 8026
'''
IMG_HEIGHT = 80
IMG_WIDTH_TEXTLINE = 1400
IMG_WIDTH_WORD = 460
TEXTLINE = True # True: textline-based    False: word-based

# Download the Esposalles datasets from http://rrc.cvc.uab.es/?ch=10&com=downloads
baseDir = '/home/lkang/datasets/OfficialEsposalles/'
train = baseDir + 'train/'
validation = baseDir + 'validation/'
test = baseDir + 'test/'

def labelDictionary():
    if TEXTLINE:
        labels = [' ']
        labels += list(string.ascii_lowercase)
    else:
        labels = list(string.ascii_lowercase)
    labels += list(string.ascii_uppercase)
    labels += list('0123456789#ç')
    labels.remove('k')
    labels.remove('K')
    labels.remove('w')
    labels.remove('W')
    labels.remove('Z')
    return len(labels), {label:n for n, label in enumerate(labels)}


def init():
    trainImage = []
    validationImage = []
    testImage = []
    trainLabel = []
    validationLabel = []
    testLabel = []
    tmp_trainLabel = []
    tmp_validationLabel = []
    tmp_testLabel = []
    for i, v in enumerate([train, validation, test]):
        # Do remember to put the groundtruth file to the datasets directory
        if TEXTLINE:
            groundtruth = 'line_groundtruth.txt'
        else:
            groundtruth = 'groundtruth.txt'
        with open(v + groundtruth, 'r') as gt:
            for line in gt:
                values = line[:-1].split(':')
                if i == 0:
                    trainImage.append(values[0])
                    tmp_trainLabel.append(values[1])
                elif i == 1:
                    validationImage.append(values[0])
                    tmp_validationLabel.append(values[1])
                elif i == 2:
                    testImage.append(values[0])
                    tmp_testLabel.append(values[1])

    labelNum, labelDict = labelDictionary()

    for i in tmp_trainLabel:
        label = [labelDict[j] for j in i]
        trainLabel.append(label)
    for i in tmp_validationLabel:
        label = [labelDict[j] for j in i]
        validationLabel.append(label)
    for i in tmp_testLabel:
        label = [labelDict[j] for j in i]
        testLabel.append(label)

    #trainLabel = sequence.pad_sequences(trainLabel, padding='post', maxlen=LABEL_LENGTH)
    #validationLabel = sequence.pad_sequences(validationLabel, padding='post', maxlen=LABEL_LENGTH)
    #testLabel = sequence.pad_sequences(testLabel, padding='post', maxlen=LABEL_LENGTH)

    return labelNum, (trainImage, trainLabel), (validationImage, validationLabel), (testImage, testLabel)


def readImage(base, imageId):
    info = imageId.split('_')
    if TEXTLINE:
        fileName = base + '_'.join(info[:-1]) + '/lines/' + imageId + '.png'
    else:
        fileName = base + '_'.join(info[:2]) + '/words/' + imageId + '.png'
    img = cv2.imread(fileName, 0)
    rate = float(IMG_HEIGHT) / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1]*rate), IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    #img = 1. - img.astype('float32') / 255.
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    binary = thresh/255.
    img_width = binary.shape[-1]
    #return binary, img_width
    if TEXTLINE:
        IMG_WIDTH = IMG_WIDTH_TEXTLINE
    else:
        IMG_WIDTH = IMG_WIDTH_WORD
    outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    if img_width > IMG_WIDTH:
        outImg = binary[:, :IMG_WIDTH]
    else:
        outImg[:, :img_width] = binary #outImg.shape (IMG_HEIGHT, IMG_WIDTH)
    #output = np.transpose(outImg, (1, 0)) #output shape (IMG_WIDTH, IMG_HEIGHT)
    return outImg, img_width
    #return output[:, :, None] #return shape (IMG_WIDTH, IMG_HEIGHT, 1)
     #img has fixed height of IMG_HEIGHT, and its value is between 0-1 0:background

#img = readImage(train, trainImage[0])

def getData(train_data_size=None, validation_data_size=None, test_data_size=None):
    labelNum, (trainImage, trainLabel), (validationImage, validationLabel), (testImage, testLabel) = init()
    trainImg = []
    seqLen_train = []
    for i in trainImage[:train_data_size]:
        img, width = readImage(train, i)
        trainImg.append(img)
        seqLen_train.append(width)
    #validationImg = [readImage(validation, i) for i in validationImage[:validation_data_size]]

    validationImg = []
    seqLen_validation = []
    for i in validationImage[:validation_data_size]:
        img, width = readImage(validation, i)
        validationImg.append(img)
        seqLen_validation.append(width)

    testImg = []
    seqLen_test = []
    for i in testImage[:test_data_size]:
        img, width = readImage(test, i)
        testImg.append(img)
        seqLen_test.append(width)
    return labelNum, (trainImg, seqLen_train, trainLabel[:train_data_size]), (validationImg, seqLen_validation, validationLabel[:validation_data_size]), (testImg, seqLen_test, testLabel[:test_data_size])


if __name__ == '__main__':
    #labelNum, (trainImg, seqLen_train, trainLabel), (validationImg, seqLen_validation, validationLabel), (testImg, seqLen_test, testLabel) = getData(50, 10, 20)
    labelNum, (trainImg, seqLen_train, trainLabel), (validationImg, seqLen_validation, validationLabel), (testImg, seqLen_test, testLabel) = getData(None, None, None)
    #print(max(seqLen_train), max(seqLen_validation), max(seqLen_test))
    #print(len(trainLabel), len(validationLabel), len(testLabel))

