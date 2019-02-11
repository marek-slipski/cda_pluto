import numpy as np
from planetaryimage import CubeFile

################################################################################
# Functions to use DEM to automatically detect craters.
#
# Marek Slipski
# 20171113
# 20171117
################################################################################



def slice_cube(cubfile,pixelrows,pixelcols,savef=False):
    '''
    Grab a slice of a cube file with known pixel rows and columns
    Inputs
    ------
    cubfile
    pixelrows/cols: tuple, (y1,y2) (x1,x2)

    Outputs
    -------
    cub_array: array
    '''
    cubimage = CubeFile.open(cubfile) #open cub file
    cubimage_fixed = cubimage.apply_numpy_specials() #fix nans for python
    cub_array = cubimage_fixed[0,pixelrows[0]:pixelrows[1],
                               pixelcols[0]:pixelcols[1]]
    if savef:
        np.save(savef,cub_array)
    return cub_array


if __name__ == '__main__':
    print '\n'
    import matplotlib
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt
    print '\n'

    import sys

    cub = 'Pluto_NewHorizons_Global_DEM_300m_Jul2017_32bit.cub'
    #img1 = slice_cube('Pluto_NewHorizons_Global_DEM_300m_Jul2017_32bit.cub',
    #           (5150,5350),(8750,8950),savef='img0')

    #try for 4400-6000, 8250,10250
    r1 = sys.argv[1]
    r2 = sys.argv[2]
    c1 = sys.argv[3]
    c2 = sys.argv[4]
    save_name = r1+'_'+r2+'_'+c1+'_'+c2
    img1 = slice_cube(cub,(int(r1),int(r2)),(int(c1),int(c2)),savef ='slices/'+save_name)


    #test2 = np.load('img0.npy')
    #lam = 5
    #Ct = C_transform_image(test2,lam)

    #cubimage = CubeFile.open(cub) #open cub file
    #cubimage_fixed = cubimage.apply_numpy_specials() #fix nans for python

    plt.figure()
    #plt.imshow(cubimage_fixed)
    ax = plt.imshow(img1)
    plt.colorbar(ax)
    plt.savefig('slices/'+save_name+'.png',dpi=300)
    plt.close()
    #plt.show()

    #plt.figure()
    #plt.imshow(Ct,cmap='seismic',vmin=-1.e+4,vmax=1.e+4)
    #plt.show()
