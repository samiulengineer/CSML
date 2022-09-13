# import os
import numpy as np
# import matplotlib.pyplot as plt


def readShortComplex(fileName, width=1):
    """Read scomplex data
        
       Usage example:
       slc = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/rslc/20140611.rslc', width=1000)
    """
    return np.fromfile(fileName, '>i2').astype(np.float).view(np.complex).reshape(-1, width)


def readFloatComplex(fileName, width=1):
    """Read fcomplex data
        
       Usage example:
       ifg = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20131217_20131228.diff.orb_cor', width=1000)
    """
    return np.fromfile(fileName, '>c8').astype(np.complex).reshape(-1, width)


def readFloat(fileName, width=1):
    return np.fromfile(fileName, '>f4').astype(np.float).reshape(-1, width)


def writeShortComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.copy().view(np.float).astype('>i2').tofile(out_file)
    out_file.close()


def writeFloatComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>c8').tofile(out_file)
    out_file.close()


def writeFloat(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>f4').tofile(out_file)
    out_file.close()



#fg = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20131217_20131228.diff.orb_cor', width=1000)
#print(ifg)
'''[[-14972. +4234.j -14917. +3884.j -15232.+13812.j ... -15544.+26986.j
   17490.-13638.j -15334.-26879.j]
 [ 16918.+29380.j -15564.-31845.j  17560.+23695.j ...  18406.-32440.j
   18644.+11885.j  18558.+27695.j]- 
 [-15188.+24215.j -14933.+32497.j -14886.+25050.j ... -15280. +6430.j
  -15437.-29174.j  17009.-21744.j]
 ...1
 [-14859.-27125.j -15082.+25781.j -15171.+23120.j ... -14547.-29578.j
   18465.+25668.j -14432.-22658.j]
 [ 18070.+13382.j -14622.+24897.j  17959.-27988.j ...  17086.+20400.j
  -15323.+12023.j -15217.+16241.j]
 [-14878.-29076.j -14910.-25661.j  17197.+17242.j ... -14539.+11328.j
  -14751.-25702.j -14722.+29069.j]]
'''
#slc = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/rslc/20140611.rslc', width=1000)
#print(slc)
#print(type(slc))
'''[[  86. -40.j  -67.  +4.j  -86.  +4.j ...   13.  +8.j  -16. -56.j
   -20. -49.j]
 [  22. -63.j  -48. +11.j  -95. +58.j ...   40.  +8.j   12. -24.j
   -19. -18.j]
 [ -20. -94.j   17. -26.j  -44. +70.j ...   28. +19.j   25. -24.j
   -24. -10.j]
 ...
 [ 106. +29.j  -51.-134.j -110.-133.j ...  246. -99.j  714.-101.j
   571.-384.j]
 [ 114. -30.j  -33.-110.j  -48.-121.j ...  502. -51.j  975.+245.j
   619.-227.j]
 [ 122. +25.j   16. -73.j   46.-107.j ...  435.+192.j  770.+431.j
   270. -91.j]] <class 'numpy.ndarray'>'''

#te= readFloat('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/rslc/20140611.rslc', width=1000)
#print(te)

"""[[7.98963211e-39            nan            nan ... 1.19387266e-39
             nan            nan]
 [2.11212813e-39            nan            nan ... 3.67343106e-39
  1.19382782e-39            nan]
 [           nan 1.65300250e-39            nan ... 2.57142052e-39
  2.38768927e-39            nan]
 ...
 [9.73460323e-39            nan            nan ... 2.26832288e-38
  2.98279427e-37 1.38116281e-37]
 [1.05610400e-38            nan            nan ... 9.07331843e-38
  1.21665863e-36 1.73382872e-37]
 [1.12039656e-38 1.56110114e-39 4.31611838e-39 ... 6.57552914e-38
  3.82054991e-37 2.62646969e-38]]"""

#te= readFloat('//mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20131217_20131228.diff.orb_cor', width=1000)
#print(te)
#x=plt.imshow(te) dont work 
#plt.show()
"""[[-4.22606738e+03 -5.98589648e+03 -1.02568604e+03 ... -2.00411774e+02
   8.43167603e+02 -6.18359436e+02]
 [ 3.76120758e+01 -1.80514084e+02  1.21889246e+03 ...  1.18018562e+05
   4.34547406e+05  2.60528734e+05]
 [-1.37895593e+03 -5.48786768e+03 -6.98823145e+03 ... -8.32392456e+02
  -3.59109680e+02  6.04170532e+01]
 ...
 [-7.85875537e+03 -2.40629419e+03 -1.51482227e+03 ... -4.44284609e+04
   1.65265062e+05 -8.22549844e+04]
 [ 1.92261367e+04 -2.89766270e+04  1.07246680e+04 ...  9.51556396e+01
  -6.60733826e+02 -1.14598254e+03]
 [-7.24980273e+03 -6.22747021e+03  1.73263092e+02 ... -4.63802500e+04
  -1.44389004e+04 -1.62843877e+04]"""