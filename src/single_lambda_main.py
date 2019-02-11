import sys
import numpy as np
import scipy, scipy.ndimage, scipy.interpolate
from skimage import filters as skfilt
import cv2
import os

import c_transform as ctr
import derivs
import shapes_and_coeffs as shp

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

def main(img,lam,name):
    elev = np.load(img)
    sig = lam

    # naming directories and files
    name_base = name +'_c'+ str(lam).zfill(2)
    os.system('mkdir -p '+ name_base)
    name_shape = name_base + '/shapes'
    os.system('mkdir -p '+ name_shape)

    print 'Calculating C-transform...\n'
    Ct = ctr.C_transform_image(elev,lam) # comvert elev pixels to artificial
    Ct_name = name_base+'/'+name_base
    np.save(Ct_name,Ct)

    plt.figure() #plot and save transform
    plt.imshow(Ct,vmin=-1.e+4,vmax=1.e+4,cmap='seismic')
    plt.savefig(Ct_name+'.png',dpi=300)
    plt.close()

    # Gaussian blur transformed image to remove some artifacts
    print 'Gaussian blur of transform...\n'
    Ct_blur = skfilt.gaussian(Ct,sigma=sig,preserve_range=True)
    Ct_blur_name = Ct_name + '_blur'+str(sig).zfill(2)
    np.save(Ct_blur_name,Ct_blur)

    plt.figure()
    plt.imshow(Ct_blur,vmin=-1.e+4,vmax=1.e+4,cmap='seismic')
    plt.savefig(Ct_blur_name+'.png',dpi=300)
    plt.close()

    # Calculate derivates
    print 'Calculating derivatives...\n'
    deriv_list = derivs.derivs(Ct_blur)  # first and second x,y,xy,yx derivs
    deriv_name = Ct_blur_name + '_derivs'

    fig,ax = plt.subplots(2,4)
    for i,div in enumerate(deriv_list[0:4]):
        ax[0,i].imshow(div)
    for j,div in enumerate(deriv_list[4:]):
        ax[1,j].imshow(div,cmap='seismic')
    fig.savefig(deriv_name+'.png',dpi=300)
    plt.close()

    # Find depressions
    print 'Finding depressions...\n'
    dep_map = derivs.depression(*deriv_list[4:]) #use second derivs
    dep_name = Ct_blur_name + 'deps'
    np.save(dep_name,dep_map)

    plt.figure()
    plt.imshow(dep_map,cmap='Greys')
    plt.savefig(dep_name+'.png',dpi=300)
    plt.close()

    # Find depression boundaries
    print 'Finding depression boundaries...\n'
    # get contours
    im2,contours,hier = cv2.findContours(np.copy(dep_map).astype('uint8'), \
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # fix up
    con_inds = shp.remove_edge_shapes(dep_map,contours) #remove objects on the edge
    cont_adj = [contours[z].reshape(len(contours[z]),2) for z in con_inds] #list of objs
    for c,check_cont in enumerate(cont_adj): #shapes with one point passing, need to rem
        if len(check_cont) <= 10: #lower limit to number of points on boundary
            del cont_adj[c] # delete from cont_adj

    # plot all contours
    plt.figure()
    for i in range(len(contours)):
        plt.imshow(cv2.drawContours(im2,contours,i,100,thickness=3),cmap='Greys')
    plt.savefig(name_shape+'/'+'contours_all.png',dpi=300)
    plt.close()

    # plot only those contours in fully in image
    plt.figure()
    for j in con_inds:
        plt.imshow(cv2.drawContours(dep_map,contours,j,100,thickness=3),cmap='Greys')
    plt.savefig(name_shape+'/'+'contours_adj.png',dpi=300)
    plt.close()

    # Loop through each object, get shape coefficients
    print 'Determining shape coefficients...\n'
    rec_fig, rec_ax = plt.subplots() #initialize figure, angles
    rec_ax.imshow(dep_map)
    arb_ang = scipy.linspace(-np.pi, np.pi, num=100, endpoint=False)
    for t,cont in enumerate(cont_adj): #go through each contour object
        cent_x,cent_y = shp.centroid(cont) #find the centroid
        np.save(name_shape+'/'+str(t)+'_centroid',np.array([cent_x,cent_y]))
        x,y = cont.T #separate x and y
        xdiff = x - cent_x # relative to centroid
        ydiff = y - cent_y
        np.save(name_shape+'/'+str(t)+'_xypoints',np.array([x,y]))

        # Get Fourier Coeffs
        f_c = shp.coeffs(xdiff,ydiff) #determine shape coefficients for
        f_c[10:] = 0 # ignore higher order coefficients

        #plot fourier coefficient amplitudes
        An = np.sqrt(f_c.real**2+f_c.imag**2) #amplitudes
        plt.figure()
        plt.plot(np.arange(len(An)),An)
        plt.xlim(1,10)
        plt.ylim(0,An[1:].max()+5)
        plt.savefig(name_shape+'/'+str(t)+'_CoeffAmps')
        plt.close()

        #check points
        r_fit = np.fft.irfft(f_c) # get back radius values from coefficients
        rec_ax.plot(cent_x,cent_y,'ro',ms=3)
        rec_ax.plot(r_fit*np.cos(arb_ang)+cent_x,r_fit*np.sin(arb_ang)+cent_y,'k',lw=3)
    rec_fig.savefig(name_shape+'/PointsExtracted.png',dpi=300)
    plt.close()


if __name__=='__main__':
    data = sys.argv[1]
    smooth = int(sys.argv[2])
    name = sys.argv[3]
    main(data,smooth,name)
