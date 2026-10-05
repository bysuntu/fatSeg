import os
import numpy as np
import nibabel as nib

sourceDir = r'C:\Users\hctsbo\Desktop\Dec_Thigh'

for case in os.listdir(sourceDir):
    caseName = os.path.join(sourceDir, case)
    segF = os.path.join(caseName, 'seg.nii.gz')
    if not os.path.isfile(segF):
        continue
    nib_seg = nib.load(segF)
    seg = nib_seg.get_fdata()
    # print(np.max(seg))
    seg1 = np.sum(seg == 2) * 0.8 * 0.8 * 3
    seg2 = np.sum(seg == 1) * 0.8 * 0.8 * 3
    seg3 = np.sum(seg == 3) * 0.8 * 0.8 * 3
    print(case, seg1, seg2, seg3)