
import argparse
import os
from os.path import expanduser, join
from dipy.io.image import load_nifti, save_nifti
from scipy.optimize import curve_fit
import numpy as np
from dipy.io import read_bvals_bvecs
from tqdm import tqdm
from joblib import Parallel,delayed
import warnings

def has_negative_zero(arr):
    n = len(arr)
    for i in range(n):
        if arr[i] <= 0.01:
            return True

    return False

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c


def calc_abc(x, y):
    bounds = ([0, -0.007, 0], [np.inf, 0, np.inf])
    # ignore warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            popt, pcov = curve_fit(exponential_func, x, y, p0=[y[0], np.log(y[1] / y[0]) / x[1], 0], bounds=bounds)
        except:
            return 0, 0, 0
        return popt[0], popt[1], popt[2]


# parser
parser = argparse.ArgumentParser(
    description="Fit ABC",
    epilog="Written by WTL.\"")
parser.add_argument("-v", "--version",
                    action="version", default=argparse.SUPPRESS,
                    version='4.0',
                    help="Show program's version number and exit")
parser.add_argument(
    'subjectDirectory',
    help='A directory of study subjects.')
parser.add_argument(
    'dwiFile',
    help='Name of DWI.')
parser.add_argument(
    'bvalFile',
    help='Name of b-value.')
parser.add_argument(
    'bvecFile',
    help='Name of gradiet vectory.')
parser.add_argument(
    'maskFile',
    help='Name of brain mask.')

n_jobs = 32
args = parser.parse_args()

subjectDirectory = args.subjectDirectory

dir = subjectDirectory
fdwi = join(dir, args.dwiFile)
fbval = join(dir, args.bvalFile)
fbvec = join(dir, args.bvecFile)
mask = join(dir, args.maskFile)


data, affine = load_nifti(fdwi, return_img=False)
mask, mask_affine = load_nifti(mask, return_img=False)
new_mask = mask
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
# 112 112 50 136
print(data.shape)
print(mask.shape)
# round and sort
bvals = np.round(bvals / 100) * 100
bvals_sorted = sorted(list(set(bvals)))
print(bvals_sorted)

sms_list = []
# spherical mean signal
for i in range(len(bvals_sorted)):
    indices = np.where(bvals == bvals_sorted[i])[0]
    # print(indices)
    sms_i = np.mean(data[..., indices], axis=3)
    sms_list.append(sms_i)

# stack to 112 112 50 3 or 4
sms = np.stack(sms_list, axis=-1)
print(sms.shape)


def fit_ABC(x, voxel):
    ii, jj, kk, y = voxel[0], voxel[1], voxel[2], voxel[3]
    a, b, c = calc_abc(x, y)
    return [(ii, jj, kk), a, b, c]


# A B C
A = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
B = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
C = np.zeros([data.shape[0], data.shape[1], data.shape[2]])

count = 0
#
count_exception = []
count_total = np.count_nonzero(mask)

x = bvals_sorted

# for each voxel
task_list = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            if mask[i, j, k] == 0:
                continue

            y = sms[i, j, k, :]
            if has_negative_zero(y) or np.sum(y) <= 1:
                # print(y)
                count += 1
                new_mask[i, j, k] = 0
                continue

            task_list.append([i, j, k, y])

#
results = Parallel(n_jobs=n_jobs)(delayed(fit_ABC)(x, voxel) for voxel in tqdm(task_list))

#
for voxel in results:

    coordinate = voxel[0]
    i, j, k = coordinate[0], coordinate[1], coordinate[2]
    a = voxel[1]
    b = voxel[2]
    c = voxel[3]
    if a == 0 and b == 0 and c == 0:
        count += 1
        new_mask[i, j, k] = 0

    A[i, j, k] = a
    B[i, j, k] = b
    C[i, j, k] = c
#
B = -B
# save
output_path = dir
save_nifti(join(output_path, 'new_mask.nii.gz'), new_mask, mask_affine)
save_nifti(join(output_path, 'A.nii.gz'), A, affine)
save_nifti(join(output_path, 'B.nii.gz'), B, affine)
save_nifti(join(output_path, 'C.nii.gz'), C, affine)

print(f"fitting ABC complete.Bad voxel removal amount:{count},accounting for {count / count_total * 100}%")

