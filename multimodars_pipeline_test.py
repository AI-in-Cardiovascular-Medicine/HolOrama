import os
from src.pages.fusion import pipeline

ROOT_CCTA = (
    "E:/PostDoc_Anselm-Stark/07_ccta_data/21_ccta_images/V1/NARCO_306/DICOM/0000965F/AAC3C2AA/AAF47571/000012EB_nifti"
)
ROOT_IVUS = "E:/PostDoc_Anselm-Stark/06_ivus_data/NARCO_306/20250108/091324/Run1"
ROOT_CCTA_MM = "D:/00_coding/multimoda-rs/examples/data"
ROOT_IVUS_MM = "D:/00_coding/multimoda-rs/examples/data"

path_ccta = os.path.join(ROOT_CCTA, "aortic_root.stl")
path_ao = os.path.join(ROOT_CCTA, "ao_cl.vtp")
path_rca = os.path.join(ROOT_CCTA, "rca_cl.vtp")
path_lca = os.path.join(ROOT_CCTA, "lca_cl.vtp")

cl_ao = pipeline.read_centerline_vtp(path_ao).cleanup_vtp(smooth=True)
cl_rca = pipeline.read_centerline_vtp(path_rca).cleanup_vtp(smooth=True)
cl_lca = pipeline.read_centerline_vtp(path_lca).cleanup_vtp(smooth=True)

print(len(cl_rca.points))
print(cl_rca.points[0])
print(cl_rca.points[-1])

results, (ao_cl_new, rca_cl_new, lca_cl_new) = pipeline.run_label_geometry(
    path_ccta,
    cl_ao,
    cl_rca,
    cl_lca,
)
