import os
from src.pages.fusion import pipeline

ROOT_CCTA = (
    "E:/PostDoc_Anselm-Stark/07_ccta_data/21_ccta_images/V1/NARCO_306/DICOM/0000965F/AAC3C2AA/AAF47571/000012EB_nifti"
)
ROOT_IVUS = "E:/PostDoc_Anselm-Stark/06_ivus_data/NARCO_306/20250108/091324/Run1"
ROOT_CCTA_MM = "D:/00_coding/multimoda-rs/examples/data"
ROOT_IVUS_MM = "D:/00_coding/multimoda-rs/examples/data"

path_ao = os.path.join(ROOT_CCTA, "ao_cl.vtp")
path_rca = os.path.join(ROOT_CCTA, "rca_cl.vtp")
path_lca = os.path.join(ROOT_CCTA, "lca_cl.vtp")

path_ao_mm = os.path.join(ROOT_CCTA_MM, "ao_cl.vtp")
path_rca_mm = os.path.join(ROOT_CCTA_MM, "rca_cl.vtp")
path_lca_mm = os.path.join(ROOT_CCTA_MM, "lca_cl.vtp")

cl_ao = pipeline.read_centerline_vtp(path_ao)
cl_rca = pipeline.read_centerline_vtp(path_rca)
cl_lca = pipeline.read_centerline_vtp(path_lca)

cl_ao_mm = pipeline.read_centerline_vtp(path_ao)
cl_rca_mm = pipeline.read_centerline_vtp(path_rca).cleanup_vtp_data(smooth=True)
cl_lca_mm = pipeline.read_centerline_vtp(path_lca)

print(len(cl_rca.points))
print(cl_rca.points[0])
print(cl_rca.points[-1])

print(len(cl_rca.points))
print(cl_rca_mm.points[0])
print(cl_rca_mm.points[-1])
