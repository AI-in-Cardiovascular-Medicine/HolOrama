import os
from enum import Enum

import pydicom as dcm
import pandas as pd
import nibabel as nib


class SupportedType(Enum):
    IVUS = "IVUS"
    NIRS = "NIRS"
    OCT = "OCT"


ds = dcm.read_file(r"D:\00_coding\AIVUS-CAA\test_cases\anonymized.dcm", force=True, defer_size=256)
array = ds.pixel_array

rows = []
for elem in ds:  # iterates main dataset, excludes file_meta
    if elem.name == 'Pixel Data':
        continue
    else:
        rows.append(
            {
                "Tag": str(elem.tag),
                "VR": elem.VR,
                "Description": elem.name,
                "Value": elem.value,
            }
        )

df = pd.DataFrame(rows)
print(df)
print(df[df['Description'] == 'Modality']['Value'])

nft = nib.load(r"E:\PostDoc_Anselm-Stark\08_oct_project\test_anselm\img\BE003_RCAdis.nii.gz")
array_nft = nft.get_fdata()

rows_nft = []
for field in nft.header.structarr.dtype.names:
    rows_nft.append(
        {
            "Tag": field,
            "VR": str(nft.header.structarr.dtype[field]),
            "Description": field,
            "Value": nft.header[field].tolist(),
        }
    )

df_nft = pd.DataFrame(rows_nft)
print(df_nft)

root, ext = os.path.splitext("E:\PostDoc_Anselm-Stark\08_oct_project\test_anselm\img\BE003_RCAdis.nii.gz")
print(root)
print(ext)
if ext == '.gz':
    root = os.path.splitext(root)[0]
print(root)


def _check_integrity(metadata: pd.DataFrame) -> bool:
    is_dicom = not metadata[metadata['Description'] == 'Modality'].empty
    if is_dicom:
        modality = metadata[metadata['Description'] == 'Modality']['Value']
        if modality.empty or not modality.isin([t.value for t in SupportedType]).any():
            return False
        num_frames = metadata[metadata['Description'] == 'Number of Frames']['Value']
        if not num_frames.empty and int(num_frames.iloc[0]) < 1:
            return False
        return True
    else:
        dim = metadata[metadata['Description'] == 'dim']['Value']
        if not dim.empty:
            d = dim.iloc[0]
            # d[0] = number of dimensions, d[3] = z/frame count
            if hasattr(d, '__len__') and (d[0] < 3 or d[3] < 1):
                return False
        pixdim = metadata[metadata['Description'] == 'pixdim']['Value']
        if not pixdim.empty:
            p = pixdim.iloc[0]
            if hasattr(p, '__len__') and len(p) > 1 and p[1] <= 0:
                return False
        return True


print(_check_integrity(df))
print(_check_integrity(df_nft))
