import numpy as np
import scipy.ndimage as snd

################################################################################
# Functions to get 1st and 2nd derivs of C-transform in 4 dxns.
#
# Marek Slipski
# 20171113
# 20171117
################################################################################

kernel1d = np.array([0.5,0,-0.5]) #non-intuitive,does k[-1]*in[1] matches gradient
kernel_diag = np.array([[np.sqrt(2)/2,0,0],[0,0,0],[0,0,-np.sqrt(2)/2]]) #xy
kernel_diagf = np.fliplr(kernel_diag).T #flipped xy

def derivs(arr):
    dx = snd.convolve1d(arr,kernel1d,axis=1)
    dy = snd.convolve1d(arr,kernel1d,axis=0)
    dxy = snd.convolve(arr,kernel_diag)
    dyx = snd.convolve(arr,kernel_diagf)
    dx2 = snd.convolve1d(dx,kernel1d,axis=1)
    dy2 = snd.convolve1d(dy,kernel1d,axis=0)
    dxy2 = snd.convolve(dxy,kernel_diag)
    dyx2 = snd.convolve(dyx,kernel_diagf)
    return dx,dy,dxy,dyx,dx2,dy2,dxy2,dyx2

def depression(dx2,dy2,dxy2,dyx2):
    conditions = np.where(dx2<0.,1,0)*np.where(dy2<0.,1,0)*np.where(dxy2<0.,1.,0)*np.where(dyx2<0.,1,0)
    return conditions

#workflow should have "if blur Ct, blur and use that in derivs,
#  if blur derivs, blur dx,dy,dxy,dyx and then take second derivs"


if __name__=='__main__':
    import sys
    from skimage import filters as skfilt

    #print '\n'
    import matplotlib
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt
    #print '\n'

    Ctrans = np.load(sys.argv[1])
    sig = int(sys.argv[2])

    data = skfilt.gaussian(Ctrans,sigma=sig,preserve_range=True)

    deriv_list = derivs(data)

    dep_map = depression(*deriv_list[4:])
    if sys.argv[3]:
        np.save(sys.argv[3],dep_map)


    plt.figure()
    plt.imshow(data,vmin=-1.e+4,vmax=1.e+4,cmap='seismic')

    fig,ax = plt.subplots(2,4)
    for i,div in enumerate(deriv_list[0:4]):
        ax[0,i].imshow(div)
    for j,div in enumerate(deriv_list[4:]):
        ax[1,j].imshow(div,cmap='seismic')

    plt.figure()
    plt.imshow(dep_map,cmap='Greys')

    plt.show()
