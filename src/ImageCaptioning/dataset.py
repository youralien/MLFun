import os
from pycocotools.coco import COCO
import numpy as np

dataDir='/home/luke/datasets/coco'

def coco(mode="dev"):

    # train_fns
    dataType='train2014'
    train_fns = os.listdir("%s/features/%s"%(dataDir, dataType))

    # reduce it to a dev set
    if mode == "dev":
        train_fns = train_fns[:50]
    trX, trY = loadFeaturesTargets(train_fns, dataType)
    
    # val_fns
    dataType='val2014'
    test_fns = os.listdir("%s/features/%s"%(dataDir, dataType))

    # reduce it to a dev set
    if mode == "dev":
        test_fns = test_fns[:25]
    teX, teY = loadFeaturesTargets(test_fns, dataType)

    return trX, teX, trY, teY 

def loadFeaturesTargets(fns, dataType):
    """
    Note: filenames should come from the same type of dataType.

    filenames from val2014, for example, should have dataType val2014
    Parameters
    ----------
    fns: filenames, strings


    """
    annFile = '%s/annotations/captions_%s.json'%(dataDir,dataType)
    caps=COCO(annFile)
    
    X = []
    Y = []

    for fn in fns:
        # Features
        x = np.load('%s/features/%s/%s'%(dataDir, dataType, fn))
        X.append(x)
        
        # Targets
        annIds = caps.getAnnIds(imgIds=getImageId(fn));
        anns = caps.loadAnns(annIds)

        # Just get one (the first) caption for now...
        Y.append(getCaption(anns[0]))

    return X, Y

def getImageId(fn):
    """Filename to image id

    Parameters
    ----------
    fn: a string
        filename of the COCO dataset.

        example:
        COCO_val2014_000000581929.npy

    Returns
    imageId: an int
    """
    return int(fn.split("_")[-1].split('.')[0])

def getCaption(ann):
    """gets Caption from the COCO annotation object
    
    Parameters
    ----------
    ann: list of annotation objects
    """
    return str(ann["caption"])

if __name__ == '__main__':
    import ipdb

    trX, teX, trY, teY = coco(mode="dev")
    ipdb.set_trace()