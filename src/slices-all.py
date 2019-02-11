import glob
import single_lambda_main as slm

for i,elev in enumerate(glob.glob('slices/data/*')):
    print i, elev,'\n'
    fname = elev.split('/')[-1].split('.')[0]
    slm.main(elev,20,fname)
