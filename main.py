# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 03:39:18 2020

@author: Marwan Eid
"""

import cv2
import imutils
import json
from matplotlib import pyplot as plt
import numpy as np
import os

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)    
    return (kps, features)

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []
    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    else:
        return None

def stitch_both(trainImg, queryImg):
    feature_extractor = 'sift'
    feature_matching = 'bf'
    # read images and transform them to grayscale
    # Make sure that the train image is the image that will be transformed
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
    # Opencv defines the color channel in the order BGR. 
    # Transform it to RGB to be compatible to matplotlib
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)
    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)
    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        print("Error!")
    (matches, H, status) = M
    # Apply panorama correction
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]
    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    # transform the panorama image to grayscale and threshold it 
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)
    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)
    # crop the image to the bbox coordinates
    result = result[y:y + h, x:x + w]
    # show the cropped image
    #plt.figure(figsize=(20,10))
    #plt.imshow(result)
    return result

def crop_image(image):
    temp = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bottommost = image.shape[0], image.shape[0]
    lower_blue = np.array([0,0,187], np.uint8)
    upper_blue = np.array([80,120,215], np.uint8)
    mask = cv2.inRange(image, lower_blue, upper_blue)
    image = cv2.bitwise_and(image, image, mask = mask)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    items = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = items[0] if len(items) == 2 else items[1]
    cv2.drawContours(image, cnts, 0, (0, 255, 0), 1)
    templeft = tuple(cnts[0][cnts[0][:,:,0].argmin()][0])
    tempright = tuple(cnts[0][cnts[0][:,:,1].argmax()][0])
    tempbottom = tuple(cnts[0][cnts[0][:,:,1].argmax()][0])
    for cnt in cnts:
        if ((tuple(cnt[cnt[:,:,0].argmin()][0]))[0] < templeft[0]):
            templeft = tuple(cnt[cnt[:,:,0].argmin()][0])
        if ((tuple(cnt[cnt[:,:,0].argmax()][0]))[0] > tempright[0]):
            tempright = tuple(cnt[cnt[:,:,0].argmax()][0])
        if ((tuple(cnt[cnt[:,:,1].argmax()][0]))[0] > tempbottom[0]):
            tempright = tuple(cnt[cnt[:,:,1].argmax()][0])
    leftmost = templeft
    rightmost = tempright
    bottommost = tempright
    img_cropped = temp[0: bottommost[0], leftmost[0]: rightmost[0]]
    return (img_cropped)

def crop_first_img(image):
    temp = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_yellow = np.array([24,21,129], np.uint8)
    upper_yellow = np.array([122,124,204], np.uint8)
    mask = cv2.inRange(image, lower_yellow, upper_yellow)
    image = cv2.bitwise_and(image, image, mask = mask)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    items = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = items[0] if len(items) == 2 else items[1]
    cv2.drawContours(image, cnts, 0, (0, 255, 0), 1)
    tempbottom = tuple(cnts[0][cnts[0][:,:,1].argmax()][0])
    for cnt in cnts:
        if ((tuple(cnt[cnt[:,:,1].argmax()][0]))[0] > tempbottom[0]):
            tempright = tuple(cnt[cnt[:,:,1].argmax()][0])
    bottommost = tempright
    img_cropped = temp[0: bottommost[0] - 10, 0 : image.shape[1]]
    return (img_cropped)

def crop_last_img(image):
    image = cv2.rotate(image, cv2.ROTATE_180)
    image = crop_first_img(image)
    image = cv2.rotate(image, cv2.ROTATE_180)
    return image

def read_and_crop_images(n):
    images = []
    for i in range (1, n+1):
        img_file_name = "img" + str(i) + ".png"
        temp = cv2.imread(img_file_name)
        if i == 1:
            temp = crop_first_img(temp)
        if i == n:
            pass
            #crop_last_img is working but is commented now since we don't need for now since the ending line is not in the pictures
            #temp = crop_last_img(temp)
        temp = crop_image(temp)
        images.append(temp)
    return images

def generate_actual_map(img, JSON_filename, n, no_of_coral_colony, no_of_coral_fragment_area, no_of_sea_star, no_of_sponge, no_of_horiznontal_squares):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    copy = img
    map_before = np.zeros([img.shape[0], img.shape[1], 3], dtype = np.uint8)
    map_before.fill(255)
    map_before = draw_grid(map_before, no_of_horiznontal_squares)
    with open(JSON_filename) as f:
        data = json.load(f)
    sorted_obj = dict(data)
    sorted_obj['predictions'] = sorted(data['predictions'], key=lambda x : x['probability'], reverse=True)
    left = []
    top = []
    width = []
    height = []
    coral_colony_counter = coral_fragment_area_counter = sea_star_counter = sponge_counter = 0
    for i in range(0, n):
        if (coral_colony_counter + coral_fragment_area_counter + sea_star_counter + sponge_counter == n):
            pass
        else:
            left.append(int((sorted_obj['predictions'][i])['boundingBox']['left'] * img.shape[1]))
            top.append(int((sorted_obj['predictions'][i])['boundingBox']['top'] * img.shape[0]))
            width.append(int((sorted_obj['predictions'][i])['boundingBox']['width'] * img.shape[1]))
            height.append(int((sorted_obj['predictions'][i])['boundingBox']['height'] * img.shape[0]))
            center = (left[i] + width[i] / 2, top[i] + height[i] / 2)
            if (center[0] <= map_before.shape[1] / 3):
                map_center_x = (map_before.shape[1] / 6)
            elif (center[0] <= map_before.shape[1] * 2 / 3):
                map_center_x = (map_before.shape[1] / 2)
            else:
                map_center_x = (map_before.shape[1] - map_before.shape[1] / 6)
            for j in range(1, no_of_horiznontal_squares + 1):
                if (center[1] <= map_before.shape[0] * j / no_of_horiznontal_squares):
                    map_center_y = map_before.shape[0] / no_of_horiznontal_squares * (j - 1) + map_before.shape[0] / (2 * no_of_horiznontal_squares)
                    break
            map_center = (map_center_x, map_center_y)
            map_center_oval_y = int(img.shape[0] / no_of_horiznontal_squares * round(float(center[1]) / (img.shape[0] / no_of_horiznontal_squares)))
            map_center_oval = (map_center_x, map_center_oval_y)
            radius = width[i] / 2
            thickness = 3
            map_radius = map_before.shape[1] / 6 - 10
            # We just need to accurately localize the shapes on the map_before image for the shapes to be nicely drawn on it
            if (((sorted_obj['predictions'][i])['tagName'] == "Coral Colony") and (coral_colony_counter < no_of_coral_colony)):
                color = (255, 0, 0)
                axes = ((left[i] - width[i]) / 2, (top[i] - height[i]) / 2)
                map_axes = ((radius + 5) * 2, radius + 5)
                angle = 90
                start_angle = 0
                end_angle = 360
                cv2.ellipse(copy, center, axes, angle, start_angle, end_angle, color, thickness)
                cv2.ellipse(map_before, map_center_oval, map_axes, angle, start_angle, end_angle, (0, 0, 255), thickness)
                coral_colony_counter += 1
            elif (((sorted_obj['predictions'][i])['tagName'] == "Coral Fragment Area") and (coral_fragment_area_counter < no_of_coral_fragment_area)):
                color = (255, 255, 0)
                cv2.circle(copy, center, radius, color, thickness)
                cv2.circle(map_before, map_center, map_radius, (0, 255, 255), thickness)
                coral_fragment_area_counter += 1
            elif (((sorted_obj['predictions'][i])['tagName'] == "Crown of Thorn Sea Star") and (sea_star_counter < no_of_sea_star)):
                color = (0, 0, 255)
                cv2.circle(copy, center, radius, color, thickness)
                cv2.circle(map_before, map_center, map_radius, (255, 0, 0), thickness)
                sea_star_counter += 1
            elif (((sorted_obj['predictions'][i])['tagName'] == "Sponge") and (sponge_counter < no_of_sponge)):
                color = (0, 255, 0)
                cv2.circle(copy, center, radius, color, thickness)
                cv2.circle(map_before, map_center, map_radius, color, thickness)
                sponge_counter += 1
    final_map = rotate_bound(map_before, 90)
    return copy, final_map

def draw_grid(map_before, no_of_squares):
    borders_thickness = 10
    lines_thickness = 5
    black = (0, 0, 0)
    cv2.line(map_before, (0,0), (map_before.shape[1], 0), black, borders_thickness) #top h
    cv2.line(map_before, (0, map_before.shape[0]), (map_before.shape[1], map_before.shape[0]), black, borders_thickness) #bottom v
    cv2.line(map_before, (0, 0), (0, map_before.shape[0]), black, borders_thickness) #left v
    cv2.line(map_before, (map_before.shape[1], 0), (map_before.shape[1], map_before.shape[0]), black, borders_thickness) #right v
    cv2.line(map_before, (int(map_before.shape[1] / 3), 0), (int(map_before.shape[1] / 3), map_before.shape[0]), black, lines_thickness)
    cv2.line(map_before, (int(map_before.shape[1] * 2 / 3), 0), (int(map_before.shape[1] * 2 / 3), map_before.shape[0]), black, lines_thickness)
    for i in range(1, no_of_squares + 1):
        cv2.line(map_before, (0, int(map_before.shape[0] * i / no_of_squares)), (map_before.shape[1], int(map_before.shape[0] * i / no_of_squares)), black, lines_thickness)
    return map_before

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def main():
    no_of_images = 3
    no_of_coral_colony = 1
    no_of_coral_fragment_area = 1
    no_of_sea_star = 1
    no_of_sponge = 1
    no_of_horiznontal_squares = 5
    JSON_filename = "finalJSONresults.json"
    delete_command = "rm " + JSON_filename
    images = read_and_crop_images(no_of_images)
    for i in range(0, no_of_images):
        images[0] = stitch_both(images[0], images[i])
    cv2.imwrite("finalImage.png", images[0])
    os.system("docker run -p 127.0.0.1:80:80 -d 4b6390ecbfc3")
    os.system(delete_command)
    os.system("curl -X POST http://127.0.0.1/image -F imageData=@finalImage.png >> finalJSONresults.json")
    finalImage, finalMapImage = generate_actual_map(images[0], JSON_filename, 4, no_of_coral_colony, no_of_coral_fragment_area, no_of_sea_star, no_of_sponge, no_of_horiznontal_squares)
    cv2.imwrite("finalImage.png", finalImage)
    cv2.imwrite("FinalMap.png", finalMapImage)
    plt.imshow(finalImage)
    plt.show()

main()
