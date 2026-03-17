import pydicom as dcm


oct = dcm.read_file(r"D:\BERN-090FU\IMG001")
ivus = dcm.read_file(r"D:\00_coding\AIVUS-CAA\test_cases\anonymized.dcm")
print(oct)
print("\n\n==============================================================\n\n")
print(ivus)

n_frames   = oct[0x0028, 0x0008].value   # Number of Frames
frame_time = oct[0x0018, 0x1063].value   # Frame Time (ms)
duration_s = (n_frames * frame_time) / 1000
print(duration_s)

n_frames       = int(oct[0x0028, 0x0008].value)   # Number of Frames
frame_time_ms  = float(oct[0x0018, 0x1063].value) # Frame Time in ms
pullback_speed = float(oct[0x0018, 0x3101].value) # Pullback Speed in mm/s

duration_s     = (n_frames * frame_time_ms) / 1000
distance_mm    = pullback_speed * duration_s

print(f"Duration : {duration_s:.2f} s")
print(f"Distance : {distance_mm:.2f} mm")