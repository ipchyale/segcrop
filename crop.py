import glob,os
from skimage.io import imread
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/damoncrockett/ivpy/src")
from ivpy import *
from ivpy.utils import resize
from ivpy.extract import extract
from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#-------------------------

def bbox2trubbox(bbox):
    left = bbox[0]
    top = bbox[1]
    right = left + bbox[2]
    bottom = top + bbox[3]
    
    return [left, top, right, bottom]

def mask2PIL(img, mask):
    bbox = bbox2trubbox(mask['bbox'])
    cropped_image = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    pil_image = Image.fromarray(cropped_image)
    return pil_image

def crop(img, mask):
    width = mask['bbox'][2]
    height = mask['bbox'][3]

    orientation  = 'horizontal' if width > height else 'vertical'
    imgwidth = img.shape[1]
    imgheight = img.shape[0]
    
    if orientation == 'horizontal':
        location = 'top' if mask['bbox'][1] < imgheight / 2 else 'bottom'
    else:
        location = 'left' if mask['bbox'][0] < imgwidth / 2 else 'right'

    trubbox = bbox2trubbox(mask['bbox'])

    if location == 'top':
        img_cropped = img[trubbox[3]:,:]
    elif location == 'bottom':
        img_cropped = img[:trubbox[1],:]
    elif location == 'left':
        img_cropped = img[:,trubbox[2]:]
    else:
        img_cropped = img[:,:trubbox[0]]

    return img_cropped

#-------------------------

imdir = sys.argv[1]
ext = sys.argv[2]
outdir = sys.argv[3]

if not os.path.exists(outdir):
    os.makedirs(outdir)

allfiles = glob.glob(os.path.join(imdir,f'*.{ext}'))
df = pd.DataFrame({"impath":allfiles})

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")

img = imread('SBI_scaled/4793_007.jpeg') # hardcoded for now
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)
maskims = pd.Series([mask2PIL(img, masks[i]) for i in range(len(masks))])

X = extract('neural',pathcol=maskims)
X = X.applymap(lambda x:x.item())
anchor = np.array(X.iloc[2]) # also hardcoded for now

#-------------------------

for i in df.index[:10]: # testing on first 10 for now
    impath = df.impath.loc[i]
    img = imread(impath)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)
    maskims = pd.Series([mask2PIL(img, masks[i]) for i in range(len(masks))])
    
    X = extract('neural',pathcol=maskims)
    X = X.applymap(lambda x:x.item()) # these are tensor objects, we want the values
    X = np.array(X)
    
    idx = np.argmin(np.linalg.norm(X - anchor, axis=1))
    mask = masks[idx]
    
    cropped_img = crop(img, mask)
    basename = os.path.basename(impath)
    outpath = os.path.join(outdir,basename)

    Image.fromarray(cropped_img).save(outpath)