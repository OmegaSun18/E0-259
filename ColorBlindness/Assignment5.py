import numpy as np
from typing import Tuple, List, Any, Union, Optional
from joblib import Parallel, delayed

def readFasta(file) -> str:
    lines = np.loadtxt(file, dtype=str)
    # Remove the first line
    lines = lines[1:]
    seq = ''.join(lines)
    return seq

def readLastcol(file) -> str:
    lines = np.loadtxt(file, dtype=str)
    lastCol = ''.join(lines)
    return lastCol

# def getFirstcol(last_col) -> str:
#     firstCol = ''.join(sorted(last_col))
#     firstCol = firstCol.replace('$', '') + '$'
#     #Give the index of the last A
#     print(firstCol.index('A'))
#     return firstCol

def getReads(file: str) -> np.ndarray[str]:
    reads = np.loadtxt(file, dtype=str)
    return reads

def getMapping(file) -> np.ndarray[int]:
    lines = np.loadtxt(file, dtype=int)
    return lines

def createMilestones(BWT, milestone) -> Tuple[np.ndarray[int], np.ndarray[int], np.ndarray[int], np.ndarray[int]]:
    AMilestones = []
    CMilestones = []
    GMilestones = []
    TMilestones = []
    controlA = controlC = controlG = controlT = 0

    for element in range(len(BWT)):
        if BWT[element] == 'A':
            controlA += 1
        elif BWT[element] == 'C':
            controlC += 1
        elif BWT[element] == 'G':
            controlG += 1
        elif BWT[element] == 'T':
            controlT += 1
        else:
            pass

        if element % milestone == 0:
            AMilestones.append(controlA)
            CMilestones.append(controlC)
            GMilestones.append(controlG)
            TMilestones.append(controlT)
        else:
            pass

    AMilestones.append(controlA)
    CMilestones.append(controlC)
    GMilestones.append(controlG)
    TMilestones.append(controlT)

    AMilestones = np.array(AMilestones)
    CMilestones = np.array(CMilestones)
    GMilestones = np.array(GMilestones)
    TMilestones = np.array(TMilestones)
    
    #firstCol is a numpy array of the last index of each character in the first column
    firstCol = np.array([AMilestones[-1] - 1, AMilestones[-1] + CMilestones[-1] - 1, AMilestones[-1] + CMilestones[-1] + GMilestones[-1] - 1, AMilestones[-1] + CMilestones[-1] + GMilestones[-1] + TMilestones[-1] - 1])
    return AMilestones, CMilestones, GMilestones, TMilestones, firstCol

def rank(lastCol, start, end, character, milestone, milestones) -> Tuple[int, int]:
    #rank takes the reference sequence, start and end index, character and the milestone as input

    startModulus = start % milestone
    startFloor = start // milestone
    if startModulus == 0:
        firstLocation = milestones[startFloor] if (lastCol[start] == character) else (milestones[startFloor] + 1 if character in lastCol[start : end + 1] else -1)
    else:
        firstLocation = milestones[startFloor] + lastCol[startFloor * milestone + 1 : start + 1].count(character) if (lastCol[start] == character) else (milestones[startFloor] + lastCol[startFloor * milestone + 1 : start + 1].count(character) + 1 if character in lastCol[start : end + 1] else -1)

    endModulus = end % milestone
    endFloor = end // milestone
    if endModulus == 0:
        lastLocation = milestones[endFloor] if (lastCol[end] == character) else (milestones[endFloor] if character in lastCol[start : end + 1] else -1)
    else:
        lastLocation = milestones[endFloor] + lastCol[endFloor * milestone + 1 : end + 1].count(character) if (lastCol[end] == character) else (milestones[endFloor] + lastCol[endFloor * milestone + 1 : end + 1].count(character) if character in lastCol[start : end + 1] else -1)
    return firstLocation, lastLocation

def select(location, character, firstCol) -> int:
    # select takes the index i and character as input
    # Given index i, it returns the ith character in the first column of the BWT
    # The firstCol is an array which has the last index of each character in the first column
    # The firstCol array is created in the createMilestones function
    if character == 'A':
        return location - 1
    elif character == 'C':
        return firstCol[0] + location
    elif character == 'G':
        return firstCol[1] + location
    elif character == 'T':
        return firstCol[2] + location
    else:
        pass
    return None

def BWTMapper(splitReads, firstCol, lastCol, mapping, milestone, AMilestones, CMilestones, GMilestones, TMilestones) -> Tuple[np.ndarray[int], int]:
    segment = -1
    locations = []
    for l in range(len(splitReads)):
        # Initialize the range of the read to be the entire last column
        start = 0
        end = len(lastCol) - 1
        # print(f'start = {start}, end = {end}, length of lastCol = {len(lastCol)}')
        for i in range(len(splitReads[l]) - 1, -1, -1):
            if splitReads[l][i] == 'A':
                firstLocation, lastLocation = rank(lastCol, start, end, 'A',  milestone, AMilestones)
                if firstLocation == -1 or lastLocation == -1:
                    break
                start, end = select(firstLocation, 'A', firstCol), select(lastLocation, 'A', firstCol)
            elif splitReads[l][i] == 'C':
                firstLocation, lastLocation = rank(lastCol, start, end, 'C',  milestone, CMilestones)
                if firstLocation == -1 or lastLocation == -1:
                    break                
                start, end = select(firstLocation, 'C', firstCol), select(lastLocation, 'C', firstCol)
            elif splitReads[l][i] == 'G':
                firstLocation, lastLocation = rank(lastCol, start, end, 'G',  milestone, GMilestones)
                if firstLocation == -1 or lastLocation == -1:
                    break
                start, end = select(firstLocation, 'G', firstCol), select(lastLocation, 'G', firstCol)
            elif splitReads[l][i] == 'T':
                firstLocation, lastLocation = rank(lastCol, start, end, 'T',  milestone, TMilestones)
                if firstLocation == -1 or lastLocation == -1:
                    break
                start, end = select(firstLocation, 'T', firstCol), select(lastLocation, 'T', firstCol)
            elif splitReads[l][i] == 'N':
                firstLocation, lastLocation = rank(lastCol, start, end, 'A',  milestone, AMilestones)
                if firstLocation == -1 or lastLocation == -1:
                    break
                start, end = select(firstLocation, 'A', firstCol), select(lastLocation, 'A', firstCol)
            else:
                pass
            if start > end:
                firstLocation = lastLocation = -1
                break
        # print(f"locations = {locations}, segment = {segment}, start = {start}, end = {end}, firstLocation = {firstLocation}, lastLocation = {lastLocation}, splitReads = {splitReads}, l = {l}")
        if firstLocation != -1 and lastLocation != -1:
            for i in range(start, end + 1):
                locations.append(mapping[i])
            segment = l
            break
    locations = np.array(locations)
    
    return locations, segment

def readSplitter(read) -> np.ndarray[str]:
    # readSplitter takes a read as input
    # It splits the read into 3 parts and returns a numpy array of three strings
    length = len(read)
    split = length // 3
    return np.array([read[:split], read[split:2*split], read[2*split:]])

def removeN(read) -> str:
    return read.replace('N', 'A')

def reverseComplement(read) -> str:
    # reverseComplement takes a read as input
    # A -> T, C -> G, G -> C, T -> A
    # It returns the reverse complement of the read
    return read.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

def bruteForce(read, readLocation, reference) -> int:
    mismatches = 0
    for i in range(len(read)):
        #take care of the case where readLocation + i is greater than the length of the reference, meaning it wraps around to the start of the reference
        if readLocation + i < len(reference):
            if read[i] != reference[readLocation + i]:
                mismatches += 1
        else:
            if read[i] != reference[readLocation + i - len(reference)]:
                mismatches += 1
    return mismatches

def locationinReference(splitReads, locations, flag, reference):
    #Returns a list of tuples
    trueLocations = []
    if flag == 0:
        for location in locations:
            read = splitReads[0] + splitReads[1] + splitReads[2]
            readLocation = location
            mismatches = bruteForce(read, readLocation, reference)
            if mismatches <= 2:
                trueLocations.append((location, location + len(splitReads[0]) + len(splitReads[1]) + len(splitReads[2])))
    elif flag == 1:
        for location in locations:
            read = splitReads[0] + splitReads[1] + splitReads[2]
            readLocation = location - len(splitReads[0])
            mismatches = bruteForce(read, readLocation, reference)
            if mismatches <= 2:
                trueLocations.append((readLocation, location + len(splitReads[1]) + len(splitReads[2])))
    elif flag == 2:
        for location in locations:
            read = splitReads[0] + splitReads[1] + splitReads[2]
            readLocation = location - len(splitReads[0]) - len(splitReads[1])
            mismatches = bruteForce(read, readLocation, reference)
            if mismatches <= 2:
                trueLocations.append((readLocation, location + len(splitReads[2])))
    else:
        return None
    trueLocations = np.array(trueLocations)
    # print(f'mismatches = {mismatches}, trueLocations = {trueLocations}')
    return trueLocations

def forwardRead(read, reference, firstCol, lastCol, mapping, milestone, AMilestones, CMilestones, GMilestones, TMilestones) -> Optional[np.ndarray[Tuple[int, int]]]:
    #Some reads may have N, need to read these as A
    # print(f'length of lastCOl = {len(lastCol)}')
    read = removeN(read)
    splitReads = readSplitter(read)
    locations, flag = BWTMapper(splitReads=splitReads, firstCol=firstCol, lastCol=lastCol, mapping=mapping, milestone=milestone, AMilestones=AMilestones, CMilestones=CMilestones, GMilestones=GMilestones, TMilestones=TMilestones)
    if flag != -1:
        trueLocations = locationinReference(splitReads, locations, flag, reference)
        if trueLocations.any():
            return trueLocations
    return np.array([])

def backwardRead(read, reference, firstCol, lastCol, mapping, milestone, AMilestones, CMilestones, GMilestones, TMilestones) -> Optional[np.ndarray[Tuple[int, int]]]:
    read = removeN(read)
    read = reverseComplement(read)
    splitReads = readSplitter(read)
    locations, flag = BWTMapper(splitReads=splitReads, firstCol=firstCol, lastCol=lastCol, mapping=mapping, milestone=milestone, AMilestones=AMilestones, CMilestones=CMilestones, GMilestones=GMilestones, TMilestones=TMilestones)
    if flag != -1:
        trueLocations = locationinReference(splitReads, locations, flag, reference)
        if trueLocations.any():
            return trueLocations
    return np.array([])


def ExonMapper(redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5, locations) -> Tuple[int]:
    #checks if read is in the exon ambiguously or unambiguously
    #The ranges for the red exon are as follows:
    # 149256127 - 149256423
    # 149258412 - 149258580
    # 149260048 - 149260213
    # 149261768 - 149262007
    # The ranges for the green exon are as follows:
    # 149293258 - 149293554
    # 149295542 - 149295710
    # 149297178 - 149297343
    # 149298898 - 149299137
    red = green = -1
    for location in locations:
        start, end = location
        secondRed = 149256127 < start and end < 149256423
        thirdRed = 149258412 < start and end < 149258580
        fourthRed = 149260048 < start and end < 149260213
        fifthRed = 149261768 < start and end < 149262007
        secondGreen = 149293258 < start and end < 149293554
        thirdGreen = 149295542 < start and end < 149295710
        fourthGreen = 149297178 < start and end < 149297343
        fifthGreen = 149298898 < start and end < 149299137
        if secondRed:
            red = 2
        elif thirdRed:
            red = 3
        elif fourthRed:
            red = 4
        elif fifthRed:
            red = 5
        else:
            pass
        if secondGreen:
            green = 2
        elif thirdGreen:
            green = 3
        elif fourthGreen:
            green = 4
        elif fifthGreen:
            green = 5
        else:
            pass
    if red != -1 and green != -1:
        if red == 2:
            redExon2 += 0.5
            greenExon2 += 0.5
        elif red == 3:
            redExon3 += 0.5
            greenExon3 += 0.5
        elif red == 4:
            redExon4 += 0.5
            greenExon4 += 0.5
        elif red == 5:
            redExon5 += 0.5
            greenExon5 += 0.5
        else:
            pass
    elif red != -1:
        if red == 2:
            redExon2 += 1
        elif red == 3:
            redExon3 += 1
        elif red == 4:
            redExon4 += 1
        elif red == 5:
            redExon5 += 1
        else:
            pass
    elif green != -1:
        if green == 2:
            greenExon2 += 1
        elif green == 3:
            greenExon3 += 1
        elif green == 4:
            greenExon4 += 1
        elif green == 5:
            greenExon5 += 1
        else:
            pass
    else:
        pass
    return redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5

def colorBlindness(redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5) -> Tuple[float]:
    redFraction2 = redExon2 / (redExon2 + greenExon2)
    redFraction3 = redExon3 / (redExon3 + greenExon3)
    redFraction4 = redExon4 / (redExon4 + greenExon4)
    redFraction5 = redExon5 / (redExon5 + greenExon5)
    return redFraction2, redFraction3, redFraction4, redFraction5


# __main__

reference = readFasta('../data/chrX.fa')
lastCol = readLastcol('../data/chrX_last_col.txt')
reads = getReads('../data/reads')
mapping = getMapping('../data/chrX_map.txt')
milestone = 1000
AMilestones, CMilestones, GMilestones, TMilestones, firstCol = createMilestones(lastCol, milestone)
redExon2 = redExon3 = redExon4 = redExon5 = 0
greenExon2 = greenExon3 = greenExon4 = greenExon5 = 0

def parallel(i):
    #forwardRead and backwardRead are run in parallel
    global redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5
    forwardLocations = forwardRead(reads[i], reference, firstCol, lastCol, mapping, milestone, AMilestones, CMilestones, GMilestones, TMilestones)
    backwardLocations = backwardRead(reads[i], reference, firstCol, lastCol, mapping, milestone, AMilestones, CMilestones, GMilestones, TMilestones)
    if forwardLocations.any():
        redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5 = ExonMapper(redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5, forwardLocations)
    if backwardLocations.any():
        redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5 = ExonMapper(redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5, backwardLocations)
    if i % 1000 == 0:
        print(i)

results = Parallel(n_jobs=8)(delayed(parallel)(i) for i in range(len(reads)))

redFraction2, redFraction3, redFraction4, redFraction5 = colorBlindness(redExon2, redExon3, redExon4, redExon5, greenExon2, greenExon3, greenExon4, greenExon5)
print(redFraction2, redFraction3, redFraction4, redFraction5)