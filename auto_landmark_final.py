import cv2
import numpy as np
import tkFileDialog

#################################################################################

def pad(array):
    padded = np.zeros(((int(size[0])/2)+2,(int(size[1])/2)+2),np.uint8)
    padded[1:(1+int(size[0])/2),1:(1+int(size[1])/2)] = array
    return padded

def convolve(arr1,arr2):
    s1=[int(np.shape(arr1)[0]),int(np.shape(arr1)[1])]
    s2=[int(np.shape(arr2)[0]),int(np.shape(arr2)[1])]
    
    if s1 > s2:
        max1 = s1[0]
        max2 = s1[1]
        min1 = s2[0]
        min2 = s2[1]
        temp1 = pad(arr1)
        temp2 = arr2
        temp3 = np.zeros(s1)
    else:
        max1 = s2[0]
        max2 = s2[1]
        min1 = s1[0]
        min2 = s1[1]
        temp1 = pad(arr2)
        temp2 = arr1
        temp3 = np.zeros(s2)
    for i in range(0,max1):
        for j in range(0,max2):
            temp3[i,j] = sum(sum(temp2*temp1[i:(i+min1),j:(j+min2)]))
    return temp3

def cand_sel(dog):
    s1=[int(np.shape(dog)[0]),int(np.shape(dog)[1])]
    temp =[]
    for i in range (18,s1[0]-18):
        for j in range(8,s1[1]-8):
            temp1 = dog[i-18:i+18,j-8:j+8]
            temp1_min = 0
            for k in range (len(temp1)):
                if temp1_min > min(temp1[k]):
                    temp1_min = min(temp1[k])
            if temp1_min == dog[i,j]:
                temp.append([i,j])
                
    return temp

def sel_mask(img):
    thresh = cv2.adaptiveThreshold(res,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,201,12)
    blur = cv2.GaussianBlur(res, (3, 3), 0)
    canny = cv2.Canny(blur,70,95)
    kernel = np.ones((3, 3),np.uint8)
    thresh = np.bitwise_not(thresh)
    thresh = cv2.dilate(thresh,kernel,2)
    thresh = cv2.erode(thresh,kernel,1)
    thresh = cv2.dilate(thresh,kernel,2)
    mask = cv2.erode(thresh,kernel,1)
    return mask

def f_draw(mask2):
    img11 = img1.copy()
    s1=[int(np.shape(img11)[0]),int(np.shape(img11)[1])]
    for i in range (0,s1[0]):
        for j in range(0,s1[1]):
            if mask2[i,j] > 0:
                output = cv2.circle(img11,(j,i), 1, (0,255,0), -1)
    return output
################################################################################

file_path_string = tkFileDialog.askopenfilename()
img = cv2.imread(str(file_path_string),0)
img1 = cv2.imread(str(file_path_string))
size = np.shape(img)
img1=cv2.resize(img1,(int(size[1])/2,int(size[0])/2))
res = cv2.resize(img,(int(size[1])/2,int(size[0])/2))

#################################################################################

G1 = np.zeros((3, 3))
G2 = np.zeros((3, 3))
sigma = 1.6  ##scaling factor
k = 1.2599 ##constant multiplicative factor
for i in range(0, 3):
    for j in range(0,3):
        temp = -(((i+1)**2)+((j+1)**2))/(2*(sigma**2))
        G1[i,j] = (np.exp(temp)*7)/(44*(sigma**2))
        G2[i,j] = (np.exp(temp/(k*k))*7)/(44*k*k*(sigma**2))
##discription function
temp = np.ones((int(size[0])/2,int(size[1])/2))
img_float = (res*temp)
L1 = convolve(G1,img_float)
L2 = convolve(G2,img_float)
####difference of Gaussian function DoG
D = L2-L1
candidate=cand_sel(D)
out = np.zeros((int(size[0])/2,(int(size[1])/2)),np.uint8)
for i in range (len(candidate)):
    cv2.circle(out,((candidate[i])[1],(candidate[i])[0]), 6, (255), -1)
out = out.astype(np.uint64)
mask = sel_mask(img);
mask = mask.astype(np.uint64)
mask2 = np.bitwise_and(mask,out)
mask2 = mask2.astype(np.uint8)
output = f_draw(mask2);
cv2.imshow("input",img1)
cv2.imshow("output",output)
#################################################################################
cv2.waitKey();cv2.destroyAllWindows()
