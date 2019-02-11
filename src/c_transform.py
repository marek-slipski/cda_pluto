import numpy as np

def C_transform(elevation, foc_pix_row, foc_pix_col, l_blur):
    '''
    Find C-transform for a given pixel in an image. Uses Gaussian-like blur to
    focus around lambda-scale neighborhood around the focal pixel along with
    gradient of elevation to determine how strongly the slope points toward
    the focal pixel. Final C-transform value is an "artificial elevation."

    Proposed by Stepinski et al. (2009) - https://doi.org/10.1016/j.icarus.2009.04.026
    Used also in Liu et al. (2015) - https://doi.org/10.1007/s11038-015-9467-9

    **Possible upgrade: fairly slow, doesn't seem necessary to use every pixel in image
    for each focal pixel - think I could use a smaller kernal that's dependent on
    the lambda length scale to possibly speed up (haven't actually tested speed of components)

    **change -1,0,1 part to skip division by 0

    Inputs
    ------
    image:
    foc_pix_x,y:
    l_blur:

    Returns
    -------
    C_tran: float, value of transform at foc_pixel
    '''
    indicies = np.indices(elevation.shape) #row [0] and col [1] index arrays for position
    gradient = np.gradient(elevation) #row [0] and col [1] gradients of elevation

    dist = (indicies[0] - foc_pix_row)**2 + (indicies[1] - foc_pix_col)**2 # square of vector diff
    blur = np.exp(-1*dist/(2.*l_blur**2)) # like Gaussian blur (lambda-dependent)

    direc_yx = np.ones(indicies.shape,dtype=int) # to find direction (y-comp [0], x-comp[1])
    direc_yx[0] = direc_yx[0]*foc_pix_row # y val to subtract from each row from point
    direc_yx[1] = direc_yx[1]*foc_pix_col # x val to subtract from each col from point
    with np.errstate(divide='ignore', invalid='ignore'): #ignore division by 0
        direc_vec = (indicies-direc_yx)/np.abs(indicies-direc_yx) #distance components from point

    prod = gradient*direc_vec #first part of dot product
    C_tran = np.sum(blur * (prod[0]+prod[1])) #add to complete dot, multiply by blur, sum over all pixels

    return C_tran #return "artificial elevation"

def C_transform_image(elevation,l_blur):
    '''
    Find C-transform for each pixel in image and
    retrun a new image in "artificial elevation" units
    New array represents how strongly (positive) the slopes within lambda
    of a given pixel point toward that pixel.

    Calls C_transform for each point

    Inputs
    ------
    elevation: array, DEM
    l_blur: scale of neighborhood to look around given pixel (blur)

    Returns
    -------
    C_t_array: array, C-transform for each pixel
    '''
    # Initialize new array
    C_t_array = np.zeros(elevation.shape)

    for row_y, col_x in np.ndindex(elevation.shape): # index iterator (row [y], col [x])
        C_t_array[row_y,col_x] = C_transform(elevation,row_y,col_x,l_blur)

    return C_t_array


if __name__=='__main__':
    import sys
    elev_array = np.load(sys.argv[1])
    lam = int(sys.argv[2])
    output_array_file = sys.argv[3]

    Ct = C_transform_image(elev_array,lam)
    np.save(output_array_file,Ct)
