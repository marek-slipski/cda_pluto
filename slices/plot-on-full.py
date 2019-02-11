import numpy as np
import glob
import pandas as pd

print '\n'
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
print '\n'


def centroid(z):
    return np.sum(z)/len(z)

slice_dirs =  sorted(glob.glob('slices/c20_output/*'))

basepix = [4400,6000,8900,10300]
fullimg = np.load('slices/4400_6000_8900_10300.npy')

import cv2
tif = cv2.imread('Pluto_NewHorizons_Global_Mosaic_300m_Jul2017_8bit.tif')
tifsl =tif[4400:6000,8900:10300]

rdb = pd.read_csv('Robbins/Supplemental Material - Pluto.csv')
scale = 69.132491671495  #pixels/deg
colpix = 360*scale
rowpix = colpix/2
rdb['col'] = rdb['MASTER_LON']/360.*colpix
rdb['row'] = (rdb['MASTER_LAT']/90.*rowpix/2.-rowpix/2)*-1
#big= rdb[rdb['MASTER_DIAM']<15]
big = rdb[rdb['MASTER_CONF']>3.]
big_range = big[(big['col']>8900)&(big['col']<10300)]
big_range = big[(big['row']>4400)&(big['row']<6000)]

fig,ax = plt.subplots(figsize=(10,10))
ax.imshow(fullimg)

fig2,ax2 = plt.subplots(figsize=(10,10))
ax2.imshow(tifsl)

for sld in slice_dirs:
    fname = sld.split('/')[-1]
    rp1,rp2,cp1,cp2,blur = fname.split('_')
    #print rp1,rp2,cp1,cp2
    shapes = glob.glob(sld+'/shapes/*_xypoints.npy')
    for shfile in shapes:
        x,y = np.load(shfile)
        adjx = x+int(cp1)-basepix[2]
        adjy = y+int(rp1)-basepix[0]
        centx, centy = centroid(adjx),centroid(adjy)

        ax.plot(centx,centy,'ro',ms=4)
        ax2.plot(centx,centy,'ro',ms=4)


ax.scatter(big_range['col']-8900,big_range['row']-4400,c='y',s=10,lw=0.1)
ax.set_xlim(0,1400)
ax.set_ylim(1600,0)
ax2.scatter(big_range['col']-8900,big_range['row']-4400,c='y',s=10,lw=0.1)
ax2.set_xlim(0,1400)
ax2.set_ylim(1600,0)
fig.savefig('slices/c20_ov')
fig2.savefig('slices/c20_im_ov.png',dpi=300)
plt.show()
