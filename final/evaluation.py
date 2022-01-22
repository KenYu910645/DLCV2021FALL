import os
import math
import sys

import sklearn.metrics as skl_metrics
import numpy as np
import csv

# NoduleFinding begin
class NoduleFinding(object):
    '''
    Represents a nodule
    '''
  
    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
               CADprobability=None, noduleType=None, diameter=None, state=None, seriesInstanceUID=None):

        # set the variables and convert them to the correct type
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.noduleType = noduleType
        self.diameter_mm = diameter
        self.state = state
        self.candidateID = None
        self.seriesuid = seriesInstanceUID
# NoduleFinding end

# csvTools begin
def writeCSV(filename, lines):
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def tryFloat(value):
    try:
        value = float(value)
    except:
        value = value
    
    return value

def getColumn(lines, columnid, elementType=''):
    column = []
    for line in lines:
        try:
            value = line[columnid]
        except:
            continue
            
        if elementType == 'float':
            value = tryFloat(value)

        column.append(value)
    return column
# csvTools end

# Evaluation settings
seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
FROC_maxX = 8 # Maximum value of x-axis of FROC curve

def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    print(f"FROCGTList_local = {FROCGTList_local}")
    print(f"FROCProbList_local = {FROCProbList_local}")
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(FROCGTList): # Handle border case when there are no false positives and ROC analysis give nan values.
      print ("WARNING, this system has no false positives..")
      fps = np.zeros(len(fpr))
    else:
      fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, allNodules, maxNumberOfCADMarks=-1):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_filename: file with results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''

    results = readCSV(results_filename)

    allCandsCAD = {}
    
    for seriesuid in seriesUIDs:
        
        # collect candidates from result file
        nodules = {}
        header = results[0]
        
        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.items():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True) # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.items():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2
        
        # print ('adding candidates: ' + seriesuid)
        allCandsCAD[seriesuid] = nodules
    
    
    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    minProbValue = -1000000000.0 # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    excludeList = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(candidates.keys())

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:
            # increment the number of nodules
            if noduleAnnot.state == "Included":
                totalNumberOfNodules += 1

            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(noduleAnnot.diameter_mm)
            if diameter < 0.0:
              diameter = 10.0
            radiusSquared = pow((diameter / 2.0), 2.0)

            found = False
            noduleMatches = []
            for key, candidate in candidates.items():
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)
                dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
                if dist < radiusSquared:
                    if (noduleAnnot.state == "Included"):
                        found = True
                        noduleMatches.append(candidate)
                        if key not in candidates2.keys():
                            print ("This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (str(candidate.id), seriesuid, str(noduleAnnot.id)))
                        else:
                            del candidates2[key]
            if len(noduleMatches) > 1: # double detection
                doubleCandidatesIgnored += (len(noduleMatches) - 1)
            if noduleAnnot.state == "Included":
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True:
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    for idx in range(len(noduleMatches)):
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)
                    FROCGTList.append(1.0)
                    FROCProbList.append(float(maxProb))
                    excludeList.append(False)
                else:
                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue)
                    excludeList.append(True)

        # add all false positives to the vectors
        for key, candidate3 in candidates2.items():
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate3.CADprobability))
            excludeList.append(False)


    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList,FROCProbList,len(seriesUIDs),excludeList)
    

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    
    sens_itp = np.interp(fps_itp, fps, sens)
    # print(len(sens_itp))
    
    average_num_fp_per_scan = (((np.array([0.125,0.25,0.5,1,2,4,8])-0.125)*10000)/(8-0.125)).astype(int)
    sensitivity = []
    for fp_per_scan in average_num_fp_per_scan:
        # print(fp_per_scan)
        # print(sens_itp[fp_per_scan])
        sensitivity.append(sens_itp[fp_per_scan])

    results = {
              '0.125 FPs per scan': sensitivity[0],
              '0.25 FPs per scan': sensitivity[1],
              '0.5 FPs per scan': sensitivity[2],
              '1 FPs per scan': sensitivity[3],
              '2 FPs per scan': sensitivity[4],
              '4 FPs per scan': sensitivity[5],
              '8 FPs per scan': sensitivity[6],
              'FROC': np.mean(sensitivity)
              }
    return results
    
def getNodule(annotation, header, state = ""):
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]
    
    if diameter_mm_label in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]
    
    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]
    
    if not state == "":
        nodule.state = state

    return nodule
    
def collectNoduleAnnotations(annotations, seriesUIDs):
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:
        # print ('adding nodule annotations: ' + seriesuid)
        
        nodules = []
        numberOfIncludedNodules = 0
        
        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state = "Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1
        
        allNodules[seriesuid] = nodules
        noduleCount      += numberOfIncludedNodules
        noduleCountTotal += len(nodules)

    # print ('Total number of included nodule annotations: ' + str(noduleCount))
    # print ('Total number of nodule annotations: ' + str(noduleCountTotal))
    return allNodules
    
    
def collect(annotations_filename,seriesuids_filename):
    annotations    = readCSV(annotations_filename)
    seriesUIDs_csv = readCSV(seriesuids_filename)
    
    seriesUIDs = []
    num = len(seriesUIDs_csv)
    
    count = 0
    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])
        if count >= num:
            break
        count += 1

    allNodules = collectNoduleAnnotations(annotations, seriesUIDs)
    
    return (allNodules, seriesUIDs)
    
    
def evaluate(test_annotation_file,user_annotation_file,seriesuids_filename):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param test_annotation_file: list of annotations
    @param user_annotation_file: list of CAD marks with probabilities
    @param seriesuids_filename: list of CT images in seriesuids
    '''
    
    (allNodules, seriesUIDs) = collect(test_annotation_file, seriesuids_filename)
    
    results = evaluateCAD(seriesUIDs, user_annotation_file, allNodules, maxNumberOfCADMarks=100)
    # Print results
    for key, value in results.items():
        print(key+":", value)

if __name__ == '__main__':

    test_annotation_file       = sys.argv[1]
    user_annotation_file       = sys.argv[2]
    seriesuids_filename        = sys.argv[3]
    # execute only if run as a script
    evaluate(test_annotation_file,user_annotation_file,seriesuids_filename)
