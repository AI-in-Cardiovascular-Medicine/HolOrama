---
name: Bug report
about: Report something that does not work as expected
title: '[Bug] '
labels: bug
assignees: ''

---

**Which module?**
<!-- Tick the one where the problem occurs. -->
- [ ] Intravascular (IVUS / OCT segmentation, gating, breathing correction)
- [ ] CCTA (segmentation, 3D render, cut geometry, centerlines)
- [ ] Fusion
- [ ] Installation / startup
- [ ] Other / not sure

**Describe the bug**
A clear and concise description of what goes wrong.

**To reproduce**
Steps to reproduce the behaviour:
1. Open file / perform action '...'
2. Click '...'
3. See error

**Expected behaviour**
What you expected to happen instead.

**Screenshots**
If applicable, add screenshots — especially of the error dialog and of the state of the 3D view or the contours.

**Log file (please attach)**
The application writes a log file on every run. This is the single most helpful thing you can provide.

> **Not familiar with code? Here is how to find it:**
> The files are named `app_YYYYMMDD_HHMMSS.log` (e.g. `app_20260311_143022.log`). Attach the most recent one using the paperclip icon below the comment box.
>
> - **Installed with `HolOrama-Setup-<version>.exe`:** press <kbd>Win</kbd>+<kbd>R</kbd>, paste `%LOCALAPPDATA%\HolOrama` and press Enter.
> - **Running from source:** the `logs/` folder inside the repository you cloned.

**Environment**
- OS: [e.g. Windows 11, Ubuntu 22.04]
- HolOrama version: [e.g. 0.9.0]
- How you installed it: [Windows installer / from source]
- Python version (source installs only): [e.g. 3.11.8]
- GPU + CUDA version (only if the problem involves Automatic Segmentation): [e.g. RTX 4070, CUDA 11.8]

> **Where to find the version:** the installer file name (`HolOrama-Setup-0.9.0.exe`) or *Apps & features* on Windows. When running from source, it is printed in the startup banner in the terminal and stored in `src/version.py`.

**Data**
- Modality: [IVUS / OCT / CCTA]
- File format: [DICOM / NIfTI]
- Acquisition system or vendor, if known:

**Additional context**
Anything else that might matter — settings changed in `config.yaml`, whether a saved contour/mask file was loaded, whether it worked in an earlier version, and (for centerline problems) your vmtk setup.
