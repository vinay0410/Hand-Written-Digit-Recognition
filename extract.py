import numpy as np
import cv2
import collections

align = None


def show(imgDict, isVerbose):
    print imgDict
    if isVerbose:
        for key in imgDict:
            cv2.imshow(key, imgDict[key])
        cv2.waitKey(0)


def x_cord_contour(contour):
    # This function takes a contour from findContours
    # it then outputs the x centroid coordinates

    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return (int(M['m10']/M['m00']))

def y_cord_contour(contour):
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return (int(M['m01']/M['m00']))


def correctThickness(img):
    _, cnt, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print cnt
    flag = -1
    net_area = 0
    net_peri = 0
    h = h[0]
    for i, myh in enumerate(h):
        if h[i][3] == -1 and h[i][1] == -1:
            break

    net_area = cv2.countNonZero(img)
    net_peri = 0
    print cnt
    print h
    print i
    net_peri = cv2.arcLength(cnt[i], True)
    c = h[i][2]
    print c
    while c != -1:

        net_peri = net_peri + cv2.arcLength(cnt[c], True)

        c = h[c][0]
        print c
    print(net_area,net_peri)
    thickness = net_area/net_peri
    print thickness
    if thickness < 4:      # For 20 x 20 pixel image
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=3)
    if thickness < 7 and thickness > 4:
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=2)
    if thickness < 10 and thickness > 7:
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    return img

def getAlignedContours(contours):
    listx = [x_cord_contour(c) for c in contours]
    listy = [y_cord_contour(c) for c in contours]
    listx = abs(np.diff(listx))
    listy = abs(np.diff(listy))
    sumx = sum(listx)
    sumy = sum(listy)
    if (sumx >= sumy):
        align = 'x'
        sorted_cnt = sorted(contours, key = x_cord_contour, reverse = False)
    else:
        align = 'y'
        sorted_cnt = sorted(contours, key = y_cord_contour, reverse = False)

    return align, sorted_cnt

def makePaddedSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed

    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = int((height - width)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = int((width - height)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square

def getCentroid(img):
    cx = 0
    cy = 0
    mysum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            cx = cx + (j+1)*img[i, j]
            cy = cy + (i+1)*img[i, j]
            mysum = mysum + img[i, j]


    return (cx/mysum, cy/mysum)

def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions


    #print "dimension: " + str(dimensions) + "shape: " + str(squared.shape)
    #r = float(dimensions) / squared.shape[1]
    #dim = (dimensions, int(squared.shape[0] * r))
    dim = (dimensions, dimensions)
    #print "dimensions: " + str(dim)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized


def display(number, rect, img):

    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+ rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
    if (align == 'x'):
        #Put Text Below
        cv2.putText(img, number, (x , y + h),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    else:
        #Put Text On the Right
        cv2.putText(image, number, (x + w, y),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)


def reposition(img, centre):

    img_blank = np.zeros((28, 28), dtype=np.uint8)
    x = int(round(4 + (10 - centre[0])))
    y = int(round(4 + (10 - centre[1])))

    BLACK = [0, 0, 0]
    constant= cv2.copyMakeBorder(img,y,8-y,x,8-x,cv2.BORDER_CONSTANT,value=BLACK)
    return constant

def getImage(myfile, isVerbose):

    d = collections.OrderedDict();

    image = cv2.imread(myfile)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


    d["image"] = image
    d["gray"] = gray
    show(d, isVerbose)

    # Blur image then find edges using Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    kernel = np.ones((3,3), np.uint8)
    edged = cv2.Canny(blurred, 30, 100)


    dilate = cv2.dilate(edged, kernel, iterations=1)
    # Find Contours
    r_img, contours, _ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    align, sorted_contours = getAlignedContours(contours)


    d.clear()
    d["blurred"] = blurred
    d["edged"] = edged


    show(d, isVerbose)

    detected_images = []
    coords = []
    count = 0
    for c in sorted_contours:
        (x,y,w,h) = cv2.boundingRect(c)

        if w >= 5 and h >= 25:
            cropped = blurred[y:y+h, x:x+w]
            ret3,final = cv2.threshold(cropped,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            final_square = makePaddedSquare(final)



            final_resized = resize_to_pixel(100, final_square)

            corrected = correctThickness(final_resized)

            final_resized = resize_to_pixel(20, corrected)

            centre = getCentroid(final_resized)
            print final_resized.shape
            constant = reposition(final_resized, centre)

            print "Shape " + str(constant.shape)
            print "centroid " + str(getCentroid(constant))


            d.clear()
            d["final"] = final
            d["final_padded"] = final_square
            d["corrected"] = corrected
            d["UltimateFinal"] = final_resized
            d["repositioned"] = constant
            show(d, isVerbose)

            detected_images.append(constant)
            coords.append((x, y, w, h))
            count = count + 1
    cv2.destroyAllWindows()

    return detected_images, coords, align
