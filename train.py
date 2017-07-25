import sys
import numpy as np
sys.setrecursionlimit(10000)

def getTopPrincipalDirections(data, topK=1, approxPcaItCount=1):
    dim = data.shape[1]
    result = np.zeros((topK, dim), dtype="float32")
    data -= np.mean(data, axis=0)
    covMatrix = np.dot(data.T, data)
    for k in xrange(topK):
        init = np.random.randn(dim)
        for prev in xrange(k):
            init -= result[prev,:] * np.dot(init, result[prev,:].T)
    current = init.copy()
    for it in xrange(approxPcaItCount):
        newCurrent = np.dot(covMatrix, current)
        for prev in xrange(k):
            newCurrent -= result[prev,:] * np.dot(newCurrent, result[prev,:].T)
        newNorm = np.sum(newCurrent ** 2) ** 0.5
        eigEstimate = newCurrent / newNorm
        current = newCurrent.copy()
    result[k,:] = eigEstimate.copy()
    return result

def getPcaSplit(data):
    direction = getTopPrincipalDirections(data).ravel()
    return direction

def getPcaTreeDirections(data, treeDepth, saveFilenamePrefix="./directions_"):
    pcaDirections = []
    def buildSplit(subset, currentLayerDepth):
        if currentLayerDepth == treeDepth:
            return
        direction = getPcaSplit(subset)
        projections = np.dot(subset, direction)
        threshold = np.median(projections)
        leftIds = projections < threshold
        rightIds = projections >= threshold
        pcaDirections.append(direction)
        leftSubset = subset[leftIds,:]
        rightSubset = subset[rightIds,:]
        buildSplit(leftSubset, currentLayerDepth + 1)
        buildSplit(rightSubset, currentLayerDepth + 1)
    buildSplit(data, 0)
    directions = np.asarray(pcaDirections)
    saveFilename = saveFilenamePrefix + str(directions.shape[0])
    saveFile = open(saveFilename, "wb")
    directions.tofile(saveFile)
    saveFile.close()

dataFile = "/home/arbabenko/data/sift1M.dat"
dim = 128
data = np.fromfile(open(dataFile, "rb"), dtype="float32").reshape((-1,dim))
data = data[:10000,:]

getPcaTreeDirections(data[:,:dim/2].copy(), 2, "sift_directions_half1_")
getPcaTreeDirections(data[:,dim/2:].copy(), 2, "sift_directions_half2_")
