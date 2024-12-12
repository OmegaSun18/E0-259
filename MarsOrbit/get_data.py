import numpy as np
from datetime import datetime

def get_times(data):
    '''Extracts the times from the data and returns a numpy array of the times in days, assuming the first time to be the reference time'''
    # >>> get_data([2023, 1, 1, 0, 0], [2023, 1, 2, 12, 30], [2023, 1, 3, 6, 45], [2023, 1, 4, 18, 15], [2023, 1, 5, 9, 0], [2023, 1, 6, 23, 59])
    # array([0.0, 1.5208333333333333, 2.28125, 3.7604166666666665, 4.375, 5.999305555555556])

    #Indexing to get only the times data from all the columns of the data
    #The data is a 2D numpy array, containing 12 rows and 14 columns
    #The first 5 columns contain the times data
    #The last 9 columns contain the angles data
    times = data[:, :5]

    #The times data is stored as Year, Month, Day, Hour, Minute
    #Converting the times data to days
    #Assuming the first row to be the reference time
    #The first time is taken to be "zero". That is times[0] = 0.0
    #The time difference between the first time and the other times is calculated in days

    #Converting the times data to datetime objects
    times = [datetime(*time) for time in times]

    #Calculating the time difference between the first time and the other times in days
    times = [(time - times[0]).total_seconds() / (24 * 3600) for time in times]

    return np.array(times)

def get_oppositions(data):
    """Extracts the longitudes and latitudes from the data and returns a numpy array where each element is a tuple of the longitude and latitude in degrees for each opposition"""

    #Indexing to get only the angles data from all the columns of the data
    #The data is a 2D numpy array, containing 12 rows and 14 columns
    #The first 5 columns contain the times data
    #The next 6 columns contain the angles data
    #The last 3 columns contain data that is irrelevant to us
    angles = data[:, 5:11]

    #The angles data has 6 columns where the columns are ZodiacIndex, Degree, Minute, Second, LatDegree, LatMinute
    #To calculate longitude we use the following formula: Longitude = ZodiacIndex * 30 + Degree + Minute/60 + Second/3600
    #To calculate latitude we use the following formula: Latitude = LatDegree + LatMinute/60
    #Converting the angles data to degrees

    #Calculating the longitude and latitude for each opposition
    oppositions = []
    for angle in angles:
        longitude = angle[0] * 30 + angle[1] + angle[2] / 60 + angle[3] / 3600
        latitude = angle[4] + angle[5] / 60
        oppositions.append((longitude, latitude))
    
    return np.array(oppositions)

