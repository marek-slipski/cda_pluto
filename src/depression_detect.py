import numpy as np
import sys
import os

from skimage import filters as skfilt

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import c_transform as ctr
import derivs as der

elev_img_name = sys.argv[1]
elev_image = np.load(sys.argv[1])
lam_blur = int(sys.argv[2])
save_dir = sys.argv[3]

os.system('mkdir -p '+save_dir) #make directory for files

base =  elev_img_name.split('.')[0]
name = base+'_c'+str(lam_blur)

Ct = ctr.C_transform_image(elev_image,lam_blur)
np.save(save_dir+'/'+name,Ct)

plt.figure()
plt.imshow(Ct,vmin=-1.e+4,vmax=1.e+4,cmap='seismic')
plt.savefig(save_dir+'/'+name+'.png',dpi=300)
plt.close()

Ct_blur = skfilt.gaussian(Ct,sigma=lam_blur,preserve_range=True)
np.save(save_dir+'/'+name+'_blur'+str(lam_blur),Ct_blur)

plt.figure()
plt.imshow(Ct_blur,vmin=-1.e+4,vmax=1.e+4,cmap='seismic')
plt.savefig(save_dir+'/'+name+'_blur'+str(lam_blur)+'.png',dpi=300)
plt.close()

deriv_list = der.derivs(Ct_blur)

fig,ax = plt.subplots(2,4)
for i,div in enumerate(deriv_list[0:4]):
    ax[0,i].imshow(div)
for j,div in enumerate(deriv_list[4:]):
    ax[1,j].imshow(div,cmap='seismic')

plt.savefig(save_dir+'/'+name+'_blur'+str(lam_blur)+'_derivs.png',dpi=300)
plt.close()

dep_map = der.depression(*deriv_list[4:])
np.save(save_dir+'/'+name+'blur'+str(lam_blur)+'_deps',dep_map)

plt.figure()
plt.imshow(dep_map,cmap='Greys')
plt.savefig(save_dir+'/'+name+'_blur'+str(lam_blur)+'_deps.png',dpi=300)
