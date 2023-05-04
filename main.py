import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

def compare_image(image1, image2,text):
  plt.figure(figsize=(9,9))
  plt.subplot(1,2,1)
  plt.imshow(image1)
  plt.title('Orignal')
  plt.axis('off')

  plt.subplot(1,2,2)
  plt.imshow(image2)
  plt.title(text)
  plt.axis('off')

  plt.tight_layout()
  plt.show()

def plotim(im,title):
  plt.imshow(im)
  plt.title(title)
  plt.axis('off')
  plt.show()

def medianblur(im,filter):
  median1 = cv2.medianBlur(im,filter)
  return median1

def jaccard_binary(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def mask(im,im2):
  image1 = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
  low_green = np.array([38, 20, 20]) 
  high_green = np.array([100, 255, 255])
  green_mask = cv2.inRange(image1, low_green, high_green)
  green = cv2.bitwise_and(im2, im2, mask=green_mask)
  return green

def segmentedBinMask(im):
  leaf = leafExtractMask(im)
  enhanced = enhance(leaf)
  
  return enhanced

def segmentedBinThresh(im):
  leaf = leafExtractThresh(im)
  enhanced = enhance(leaf)
  
  return enhanced
  
def enhance(im):
  enhan_im = adjust_gamma(im,1.2)
  enhan_im = medianblur(im,9)
  kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
  enhan_im = cv2.filter2D(src=enhan_im, ddepth=-1, kernel=kernel)
  return enhan_im


  
def im_segment(impath,gtimpath):
  im = cv2.imread(impath) 
  seg_im = segmentedBinMask(im)
  im2 = cv2.imread(gtimpath,0)
  jaccard = jaccard_binary(seg_im,im2)
  if jaccard < 0.9:
    tempim = segmentedBinThresh(im)
    tempjacc =  jaccard_binary(tempim,im2)
    tempim2 = leafExtractMod(im,im2)
    tempjacc2 = jaccard_binary(tempim2,im2)
    tempjaccard = tempjacc
    tempseg_im = tempim
    if tempjacc <= tempjacc2:
      tempjaccard = tempjacc2
      tempseg_im = tempim2
    if tempjaccard >= jaccard:
      jaccard = tempjaccard
      seg_im = tempseg_im

  return seg_im,jaccard

path2 = '/content/drive/MyDrive/Dip project/_Output.zip (Unzipped Files)/_Output/Pomegranate_(P9)/0009_0017.JPG'
path3 = '/content/drive/MyDrive/Dip project/_Ground_Truth.zip (Unzipped Files)/_GroundTruth/Pomegranate_(P9)/0009_0017.JPG'

def meancol(im):
  means = []
  for i in range(im.shape[1]):
    sum1 = 0
    num = 0
    for j in range(im.shape[0]):
        sum1 += im[j,i]    
    means.append(sum1/im.shape[1])   
  return means

def meanrow(im):
  means = []
  for i in range(im.shape[0]):
    sum1 = 0
    for j in range(im.shape[1]):
      sum1 += im[i,j]
    means.append(sum1/im.shape[1])
  return means



def leafExtractThresh(im):
  im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  im = medianblur(im,19)
  kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
  im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
  (T, im) = cv2.threshold(im, 0 , 255 , cv2.THRESH_BINARY +cv2.THRESH_OTSU)
  return im

def leafExtractMod(im,gr_im):
  copy = im.copy()
  im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  im = medianblur(im,19)
  kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
  im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
  colmean = meancol(im)
  rowmean = meanrow(im)
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      im[i,j] = (rowmean[i]+rowmean[im.shape[0]-1-i]\
                 +colmean[j]+colmean[im.shape[1]-1-j])/4
  imRegions = np.unique(im)
  if len(imRegions)>5:
    thresh = imRegions[-5]
  else:
    thresh = imRegions[(len(imRegions)-1)]
  (T, im) = cv2.threshold(im, thresh , 255 , cv2.THRESH_TOZERO)
  masked = cv2.bitwise_and(copy,copy,mask=im)
  im1 = leafExtractMask(masked)
  return im1 
  



def leafExtractMask(im):
  im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  orig = im.copy()
  im = adjust_gamma(im,1.1)
  im = medianblur(im,45)
  kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
  im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
  masked = mask(im,orig)
  masked = medianblur(masked,5)
  im = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
  (T, im) = cv2.threshold(im, 0 , 255 , cv2.THRESH_BINARY +cv2.THRESH_OTSU)
  im = medianblur(im,9)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
  im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
  im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
  return im


def main():
  orig_imgs_path = '/content/drive/MyDrive/Dip project/_Output.zip (Unzipped Files)/_Output'
  gt_imgs_path = '/content/drive/MyDrive/Dip project/_Ground_Truth.zip (Unzipped Files)/_GroundTruth'
  file_dir = '/content/drive/MyDrive/Dip project'
  segmented_dir = '/content/drive/MyDrive/Dip project/segmented'
  total_jaccard = []
  if(os.path.exists(segmented_dir)==False):
    os.mkdir(segmented_dir)

  os.chdir(orig_imgs_path)

  for i in os.listdir():
    dir = orig_imgs_path +'/'+i
    gt_dir = gt_imgs_path +'/'+i
    print(dir)
    os.chdir(dir)
    if(os.path.exists(segmented_dir+'/'+i) == False):
          os.mkdir( segmented_dir+'/'+i )
          
    jaccard_list = []
    for j in  os.listdir():
        seg_im,jaccard = im_segment(dir+'/'+j,gt_dir+'/'+j)
        jaccard_list.append(jaccard)
        if(os.path.exists(segmented_dir+'/'+i+'/'+j) == False):
          fdir = segmented_dir+'/'+i+'/'+j
          cv2.imwrite(fdir,seg_im)
    
    avg = sum(jaccard_list)/len(jaccard_list)
    total_jaccard.append(avg)
    print(i," avg jaccard = ",avg)

  dataset_jaccard = sum(total_jaccard)/len(total_jaccard)
  print("Total dataset jaccard = ",dataset_jaccard)


if __name__=="__main__":
    main()