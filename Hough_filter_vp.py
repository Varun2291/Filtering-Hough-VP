import cv2
import numpy as np
import random
import math
import operator

def determine(l1, l2):
    return l1[0] * l2[1] - l1[1] * l2[0]

# Function to check if two lines intersect
def intersectingLines(line_1, line_2):
    xVal = (line_1[0][0] - line_1[1][0], line_2[0][0] - line_2[1][0])
    yVal = (line_1[0][1] - line_1[1][1], line_2[0][1] - line_2[1][1])

    intersect = determine(xVal, yVal)

    if intersect == 0:
        return 0,0

    temp = (determine(*line_1), determine(*line_2))
    x = determine(temp, xVal) / intersect
    y = determine(temp, yVal) / intersect

    return x, y

# Function to calculation the equation [A*x - b]
def calculateError(A, x, b):
    return np.subtract(np.dot(A,x), b)

# Function to get the distance between points
def distanceBetweenPoints(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to check if line pass intersect point1 & point2 
def lineIntersection(line, point1, point2):
    length1 = distanceBetweenPoints((line[0][0],line[0][1]),(point1, point2))    
    length2 = distanceBetweenPoints((line[1][0],line[1][1]),(point1, point2))
    length3 = distanceBetweenPoints((line[0][0],line[0][1]),(line[1][0],line[1][1]))

    if int(length1) + int(length2)  == int(length3):
        return True
    else:
        return False

# Function to get the points given a line segment start and end value
def bresenham_line((x,y),(x2,y2)):
    """Brensenham line algorithm"""
    steep = 0
    coords = []
    dx = abs(x2 - x)
    if (x2 - x) > 0: sx = 1
    else: sx = -1
    dy = abs(y2 - y)
    if (y2 - y) > 0: sy = 1
    else: sy = -1
    if dy > dx:
        steep = 1
        x,y = y,x
        dx,dy = dy,dx
        sx,sy = sy,sx
    d = (2 * dy) - dx
    for i in range(0,dx):
        if steep: coords.append((y,x))
        else: coords.append((x,y))
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)
    coords.append((x2,y2))
    return coords

# Function to Get the Filtered Hough lines using Standard Hough Transform's result
def filteringHoughLines(img, ori_img, houghLinesArrary):
    filteredHoughLines = []

    # First gradients along x & y direction and then calculate magnitude
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)               # Gradient along x direction
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)               # Gradient along y direction
    magnitude = np.absolute(cv2.magnitude(dx, dy))      # Magnitude of the image

    meanValue = np.mean(magnitude)

    for line in houghLinesArrary:
        points = bresenham_line((line[0][0],line[0][1]),(line[1][0],line[1][1]))    # Call Bresenham algorithm to get the points
        tempPoint1 = [(line[0][0]-line[1][0]), (line[0][1]-line[1][1])]             # Store the point for dot product
        threshold = 0
    
        # Removing the Vertical and Horizontal lines
        if line[1][0] == 0 or line[0][0] == 0 or line[1][0] - line[0][0] == 0:
            lineSlope = 1
            lineAngle = 0
            continue
        else:
            lineSlope = (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
            lineAngle = np.absolute(np.arctan(lineSlope)*180.0 / np.pi)

        # Check for Vertical Lines
        if lineAngle > 85 and lineAngle < 95:
            continue
        # Check for Horizontal Lines
        if lineAngle >= 0 and lineAngle < 10:
            continue

        # Iterate through all the points
        for a, b in points:
            if a < 1200 and b < 1600:
                xVal = np.absolute(dx[a,b]) # Get gradient for point at X
                yVal = np.absolute(dy[a,b]) # Get gradient for point at Y

                # Save point 2 only if the magnitude is greater than the threshold
                if magnitude[a, b] > 30:
                    tempPoint2 = [xVal, yVal]
                else:
                    continue
            else:
                continue

            #Get the dot product of the two points
            resDotProduct = np.dot(tempPoint1, tempPoint2)
            point1Mag = np.dot(tempPoint1, tempPoint1)**0.5
            point2Mag = np.dot(tempPoint2, tempPoint2)**0.5

            if point1Mag == 0 or point2Mag == 0:
                continue

            # Get the cosine of the angle
            cosAngle = math.cos(resDotProduct/point2Mag/point1Mag)

            angleMod = math.degrees(cosAngle) % 360

            if angleMod-180 >= 0:
                angleMod = 360 - angleMod
            else:
                angleMod = angleMod

            # Increase thresholdValue if the angle is between 85 & 95
            if not (angleMod > 85 and angleMod < 95):
                threshold += 1

        # Plot Line on the original image and add it to the array
        if threshold > 45:
            cv2.line(ori_img,(line[0][0],line[0][1]),(line[1][0],line[1][1]),(255,0,0),2)
            filteredHoughLines.append(((line[0][0],line[0][1]),(line[1][0],line[1][1])))

    cv2.imwrite("FilteredHoughLines.jpg", ori_img)
    return filteredHoughLines

# Function to get the Vanishing point using from a pair of lines
def linePairVP(image, filteredHoughLines):
    intersectionCounter = {}
    
    for line1 in filteredHoughLines:
        for line2 in filteredHoughLines:
            # if both the lines are same then don't process
            if line1 == line2:
                continue
            
            # check if point the lines are intersecting
            pointX, pointY = intersectingLines(line1, line2)
            intersectionCounter[(pointX, pointY)] = 0

            # now check for all other lines that intersect with this pair
            for lines in filteredHoughLines:
                # if lines are same then don't process
                if lines == line1 or lines == line2:
                    continue

                # check if the line is passing through the pair of points
                flag =  lineIntersection(lines, pointX, pointY)
                if flag:
                    intersectionCounter[(pointX, pointY)] += 1

    finalIntersectionCounter = max(intersectionCounter.iteritems(), key=operator.itemgetter(1))[0]
    cv2.circle(image, finalIntersectionCounter, 55, (0, 255, 0), 2)
    cv2.imwrite("LinePairVP.jpg", image)
    print finalIntersectionCounter
            
# Function to get the Vanishing point using Least Square method for all the lines
def leastSquareVP(img, houghLinesArrary):
    A = [[0 for y in xrange(2)] for x in xrange(len(houghLinesArrary))]
    b = [[0 for y in xrange(1)] for x in xrange(len(houghLinesArrary))]
    count = 0
    for lines in houghLinesArrary:

        # Removing the Vertical and Horizontal lines
        if lines[0][0] == 0 or lines[1][0] == 0 or lines[1][0] - lines[0][0] == 0:
            lineSlope = 1
            lineAngle = 0
            continue
        else:
            lineSlope = (lines[1][1] - lines[0][1])/(lines[1][0] - lines[0][0])
            lineAngle = np.absolute(np.arctan(lineSlope)*180.0 / np.pi)

        # Check for Vertical Lines
        if lineAngle > 85 and lineAngle < 95:
            continue
        # Check for Horizontal Lines
        if lineAngle >= 0 and lineAngle < 10:
            continue
        
        # Convert the line to a*x + b*y = c form 
        A[count][0] = -(lines[1][1] - lines[0][1])                              # -(y2-y1)
        A[count][1] = (lines[1][0] - lines[0][0])                               # x2-x1
        b[count][0] = A[count][0] * lines[0][0]  +  A[count][1] * lines[0][1]   # -(y2-y1)*x1 + (x2-x1)*y1
        count += 1

    x = [[0 for y in xrange(1)] for x in xrange(2)]
    error_array = [[0 for y in xrange(1)] for x in xrange(len(houghLinesArrary))]
    solution = [[0 for y in xrange(1)] for x in xrange(2)]
    error_value = 9999999999.1
    aTemp = [[0 for y in xrange(2)] for x in xrange(2)]
    bTemp = [[0 for y in xrange(1)] for x in xrange(2)]

    # iterate through the rows
    for i in xrange(count):
        for j in xrange(count):
            
            if i >= j:
                continue

            # Store the values of A & b in temp 
            aTemp[0] = A[i]
            aTemp[1] = A[j]
            bTemp[0] = b[i]
            bTemp[1] = b[j]

            # check if the rank of A temp matrix is 2, if not skip
            if not np.linalg.matrix_rank(aTemp) == 2:
                continue

            # Calculate A*x = b, get x value by passing A & b
            x = np.linalg.solve(aTemp, bTemp)

            if len(x) == 0 or len(x[0]) == 0:
                continue

            # Calculate error assuming perfect intersection [A * x - b]
            error_array = calculateError(A, x, b);
            error_array = error_array/1000              # reducing error

            tempError = 0

            # sum up all the error values
            for i in xrange(len(error_array)):
                tempError += error_array[i][0] * error_array[i][0] / 1000

            tempError = tempError / 1000000

            # Check if current errorValue is less than minimum error, if so then update the solution & error value
            if(error_value > tempError):
                solution = x
                error_value = tempError

    # plot the value of solution using a circle to get an approx. vanishing point
    cv2.circle(img, (int(solution[0][0]), int(solution[1][0])), 50,(0, 0, 255), 10)
    cv2.imwrite("LeastSquareStandardHough.jpg", img);

# Function to plot the Probablistic Hough Lines with 75 as the threshold
def probablisticHough(edges):
    minLineLength = 100
    maxLineGap = 10
    lines1 = cv2.HoughLinesP(edges,1,np.pi/180,75,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines1[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imwrite("ProbablisticHough.jpg", img);


#Start
img = cv2.imread('ST2MainHall4030.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur= cv2.GaussianBlur(img,(7,7),1)             # Apply Gaussian Blur with kernel size of (7,7)

# Take the median of the image and calculate the upper and lower bound
v =  np.median(blur)
lower =  int(max(0 , (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))

edges = cv2.Canny(blur,lower,upper,apertureSize = 3)
#cv2.imwrite("Canny.jpg", edges);

# Calculate Standard Hough Transform
lines = cv2.HoughLines(edges,1,np.pi/180,150)
houghLinesArrary = []

# Iterate through the lines and plot them using line function
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

# Plot the lines
##    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
##    cv2.imwrite("StandardHough.jpg", img);
    houghLinesArrary.append(((x1,y1),(x2,y2)))
##----------------------------------    
#probablisticHough(edges)
##----------------------------------
##----------------------------------
filteredHoughLines = filteringHoughLines(edges, img, houghLinesArrary)
##----------------------------------

##----------------------------------
linePairVP(img, filteredHoughLines)
##----------------------------------

##------------------------------------
#leastSquareVP(img, filteredHoughLines)
##------------------------------------

