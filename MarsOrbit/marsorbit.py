import numpy as np
# import get_data
from scipy.optimize import minimize
import math

# data = np.genfromtxt(
#     "../data/01_data_mars_opposition_updated.csv",
#     delimiter=",",
#     skip_header=True,
#     dtype="int",
# )

# System Windows path
# data = np.genfromtxt(
#         r".\data\01_data_mars_opposition_updated.csv",
#         delimiter=",",
#         skip_header=True,
#         dtype="int",
#     )

# times = get_data.get_times(data)
# oppositions = get_data.get_oppositions(data)

def y_intersection(e_1, e_2, z, x):
    '''Returns the y-coordinate of the intersection of the line from the equant to the ellipse'''
    return e_1 * math.sin(e_2) + math.tan(z) * x - math.tan(z) * e_1 * math.cos(e_2)

def quad_form(e_1, e_2, c, z, r):
    '''Returns the x-coordinates of the intersection of the line from the equant to the ellipse'''
    a = 1 + math.tan(z)**2
    b = -2 * math.cos(c) + 2 * math.tan(z) * (e_1 * math.sin(e_2) - math.sin(c) - e_1 * math.cos(e_2) * math.tan(z))
    ci = math.cos(c)**2 + (e_1 * math.sin(e_2) - math.sin(c) - e_1 * math.cos(e_2) * math.tan(z))**2 - r**2
    #Using the quadratic formula to find the roots
    discriminant = b**2 - 4*a*ci

    if discriminant < 0:
        raise ValueError("No real roots")
    
    root1 = (-b + math.sqrt(discriminant))/ (2*a)
    root2 = (-b - math.sqrt(discriminant))/ (2*a)
    
    #Returned both roots
    return root1, root2

def Erros_and_MaxError(params):
    '''Returns the errors and the maximum error for the given parameters'''
    #Unpacking the parameters
    c, r, e1, e2, z, s = params
    #Initialising the errors array
    errors = []
    #Converting the angles to radians
    c_rad = math.radians(c)
    e2_rad = math.radians(e2)
    #Initialising the list of equant longitudes
    equant_longitudes = []

    #Calculating the equant longitudes
    for i in times:
        equant_long = z + s * (i - times[0])
        equant_long = equant_long - 360 * math.floor(equant_long / 360)
        equant_long = math.radians(equant_long)
        equant_longitudes.append(equant_long)
    
    #Initialising the list of predicted angles
    predicted_angles = []

    for equant_long in equant_longitudes:
        #Calculating the x-coordinates of the intersection of the line from the equant to the ellipse
        predicted_x1, predicted_x2 = quad_form(e1, e2_rad, c_rad, equant_long, r)
        #Calculating the y-coordinates of the intersection of the line from the equant to the ellipse
        predicted_y1, predicted_y2 = y_intersection(e1, e2_rad, equant_long, predicted_x1), y_intersection(e1, e2_rad, equant_long, predicted_x2)
        #Choosing the correct x and y coordinates based on the quadrant
        if predicted_y1 > predicted_y2:
            if 0 <= equant_long <= math.pi:
                #If the equant longitude is in the first or second quadrant
                predicted_x = predicted_x1
                predicted_y = predicted_y1
            else:
                #If the equant longitude is in the third or fourth quadrant
                predicted_x = predicted_x2
                predicted_y = predicted_y2
        else:
            if 0 <= equant_long <= math.pi:
                #If the equant longitude is in the first or second quadrant
                predicted_x = predicted_x2
                predicted_y = predicted_y2
            else:
                #If the equant longitude is in the third or fourth quadrant
                predicted_x = predicted_x1
                predicted_y = predicted_y1

        #Calculating the predicted angle
        if predicted_x == 0:
            #Can also set the predicted angle to be 0, but I chose to raise an exception as the code was converging properly without this
            raise ValueError("Divide by 0 Error")
        if predicted_x > 0 and predicted_y > 0:
            #If the predicted x and y coordinates are in the first quadrant
            predicted_angle = math.degrees(math.atan(predicted_y / predicted_x))
        elif predicted_x < 0 and predicted_y > 0:
            #If the predicted x and y coordinates are in the second quadrant
            predicted_angle = 180 + math.degrees(math.atan(predicted_y / predicted_x))
        elif predicted_x < 0 and predicted_y < 0:
            #If the predicted x and y coordinates are in the third quadrant
            predicted_angle = 180 + math.degrees(math.atan(predicted_y / predicted_x))
        else:
            #If the predicted x and y coordinates are in the fourth quadrant
            predicted_angle = 360 + math.degrees(math.atan(predicted_y / predicted_x))
        predicted_angles.append(predicted_angle)
    
    #Calculating the errors
    for j in range(len(predicted_angles)):
        errors.append(predicted_angles[j] - oppositions[j][0])

    #Calculating the maximum error with sign
    max_error = 0
    for k in errors:
        if abs(k) > abs(max_error):
            max_error = k
    return errors, max_error


def MarsEqantOrbit(params):
    '''Returns the sum of squared errors for the given parameters for the minimize function'''
    #Unpacking the parameters
    c, r, e1, e2, z, s = params
    #Initialising the errors array
    errors = []
    #Converting the angles to radians
    c_rad = math.radians(c)
    e2_rad = math.radians(e2)
    #Initialising the list of equant longitudes
    equant_longitudes = []

    #Calculating the equant longitudes
    for i in times:
        equant_long = z + s * (i - times[0])
        equant_long = equant_long - 360 * math.floor(equant_long / 360)
        equant_long = math.radians(equant_long)
        equant_longitudes.append(equant_long)
    
    #Initialising the list of predicted angles
    predicted_angles = []

    for equant_long in equant_longitudes:
        #Calculating the x-coordinates of the intersection of the line from the equant to the ellipse
        predicted_x1, predicted_x2 = quad_form(e1, e2_rad, c_rad, equant_long, r)
        #Calculating the y-coordinates of the intersection of the line from the equant to the ellipse
        predicted_y1, predicted_y2 = y_intersection(e1, e2_rad, equant_long, predicted_x1), y_intersection(e1, e2_rad, equant_long, predicted_x2)
        #Choosing the correct x and y coordinates based on the quadrant
        if predicted_y1 > predicted_y2:
            if 0 <= equant_long <= math.pi:
                #If the equant longitude is in the first or second quadrant
                predicted_x = predicted_x1
                predicted_y = predicted_y1
            else:
                #If the equant longitude is in the third or fourth quadrant
                predicted_x = predicted_x2
                predicted_y = predicted_y2
        else:
            if 0 <= equant_long <= math.pi:
                #If the equant longitude is in the first or second quadrant
                predicted_x = predicted_x2
                predicted_y = predicted_y2
            else:
                #If the equant longitude is in the third or fourth quadrant
                predicted_x = predicted_x1
                predicted_y = predicted_y1

        #Calculating the predicted angle
        if predicted_x == 0:
            #Can also set the predicted angle to be 0, but I chose to raise an exception as the code was converging properly without this
            raise ValueError("Divide by 0 Error")
        if predicted_x > 0 and predicted_y > 0:
            #If the predicted x and y coordinates are in the first quadrant
            predicted_angle = math.degrees(math.atan(predicted_y / predicted_x))
        elif predicted_x < 0 and predicted_y > 0:
            #If the predicted x and y coordinates are in the second quadrant
            predicted_angle = 180 + math.degrees(math.atan(predicted_y / predicted_x))
        elif predicted_x < 0 and predicted_y < 0:
            #If the predicted x and y coordinates are in the third quadrant
            predicted_angle = 180 + math.degrees(math.atan(predicted_y / predicted_x))
        else:
            #If the predicted x and y coordinates are in the fourth quadrant
            predicted_angle = 360 + math.degrees(math.atan(predicted_y / predicted_x))
        predicted_angles.append(predicted_angle)
    
    #Calculating the errors
    for j in range(len(predicted_angles)):
        errors.append(predicted_angles[j] - oppositions[j][0])

    #Calculating the sum of squared errors
    predicted = np.array(predicted_angles)
    actual = np.array([oppositions[k][0] for k in range(len(oppositions))])
    squared_error = np.sum((predicted - actual)**2)
    return squared_error

def bestMarsOrbitParams(timesx, oppositionsx):
    '''Returns the best parameters for the Mars orbit by doing an exhaustive search'''
    # [c, r, e1, e2, z, s]
    # initial_guess = [0, 0, 0, 0, 0, 0]
    #Global variables for the times and oppositions
    global times, oppositions
    times = timesx
    oppositions = oppositionsx
    #Initialising s with the starting value as we know approximate mars orbit time period.
    s = 0.5241
    #Initialising the minimum error and the list of errors
    min_error = 100
    min_errors_list = []
    #Initialising the tolerance
    tolerance = 0
    #Initialising the output parameters
    r_out, s_out, c_out, e1_out, e2_out, z_out = 0, 0, 0, 0, 0, 0
    #Exhaustive search
    for r in np.linspace(1, 10, 10):
        for c in np.linspace(0, 360, 18):
            for e1 in np.linspace(1, r - 0.01, 10):
                for e2 in np.linspace(0, 360, 18):
                    for z in np.linspace(0, 360, 18):
                        #Using the try and except block to handle the errors
                        try:
                            #Using the Powell method for optimization
                            result = minimize(MarsEqantOrbit, [c, r, e1, e2, z, s], method='powell')
                            #Getting the errors and the maximum error
                            errors, maxError = Erros_and_MaxError(list(result.x))
                            #Checking if the error is less than the minimum error
                            if abs(maxError) < 0.06 and abs(maxError) <= abs(min_error):
                                #Updating the minimum error and the list of errors and the output parameters
                                min_error = maxError
                                min_errors_list = errors
                                c_out, r_out, e1_out, e2_out, z_out, s_out = list(result.x)
                                #Resetting the tolerance
                                tolerance = 0
                            else:
                                #Incrementing the tolerance if the error is not less than the minimum error
                                tolerance += 1
                            if tolerance > 10000:
                                #Breaking the loop if the tolerance is greater than 10000
                                    return r_out, s_out, c_out, e1_out, e2_out, z_out, min_errors_list, min_error
                        except:
                            continue
    return r_out, s_out, c_out, e1_out, e2_out, z_out, min_errors_list, min_error

