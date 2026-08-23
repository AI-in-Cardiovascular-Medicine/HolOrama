.. docs/contents/installation.rst

Installation
============

There are two ways to get HolOrama:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - :ref:`Pre-built installer <install-binary>`
     - :ref:`From source <install-source>`
   * - Who it is for
     - Anyone who just wants to use the application (no coding experience needed)
     - Developers, and anyone who needs automatic segmentation
   * - Requirements
     - Windows, no Python, ~1 GB disk space
     - Python toolchain, ~10 GB disk for the GPU stack
   * - Automatic AI segmentation
     - Not included
     - Included

Everything else (contouring, gating, breathing correction, CCTA segmentation, the 3D
tools and the whole fusion pipeline) is identical in both.

.. _install-binary:

Pre-built Windows installer (recommended)
-----------------------------------------

Download the latest ``HolOrama-Setup-<version>.exe`` from the `Releases
<https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/releases/latest>`_ page and run
it.

- Installs **per-user**, no administrator rights, no UAC prompt (should work on hospital systems).
- Adds Start-Menu and Desktop shortcuts.
- Logs and ``config.yaml`` live in ``%LOCALAPPDATA%\HolOrama``; your analysis outputs are
  written next to the files you open.

.. note::
   On first launch Windows may show a **"Windows protected your PC"** (SmartScreen) prompt,
   because the installer is not code-signed. Click **More info → Run anyway**.

.. warning::
   The packaged build deliberately excludes PyTorch/TensorFlow/nnUNetv2, so the
   **Automatic Segmentation** button is not available. Install from source if you need it.

.. _install-source:

Install from source
-------------------

The project uses `uv <https://docs.astral.sh/uv/>`_ for dependency management. The
provided install scripts handle every platform-specific step; use them first and fall back
to the :ref:`step-by-step instructions <install-manual>` only if a script fails.

Windows script
~~~~~~~~~~~~~~~~

#. Install the `Visual C++ Redistributable 2022 (x64)
   <https://aka.ms/vs/17/release/vc_redist.x64.exe>`_ if it is not already present.

#. Clone the repository and run the installer script from PowerShell:

   .. code-block:: powershell

      git clone https://github.com/AI-in-Cardiovascular-Medicine/HolOrama.git
      cd HolOrama
      .\install.ps1

The script creates a ``.venv``, installs all dependencies and applies both Windows-specific
fixes automatically (the missing OpenMP DLL and the ``optree`` version pin, see
:ref:`install-manual`). The default install is GPU-ready (CUDA 11.8).

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Flag
     - Effect
   * - ``-Dev``
     - Also install the development and documentation dependencies (linters, test runner,
       Sphinx)
   * - ``-NnUZoo``
     - Install nnUZoo from GitHub (required for automatic segmentation)
   * - ``-Cpu``
     - CPU-only PyTorch instead of the CUDA build
   * - ``-Cuda 121``
     - Use the CUDA 12.1 build instead of the default cu118

With ``-Dev`` the script also installs the ``pre-commit`` git hooks.

Linux / macOS script
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/AI-in-Cardiovascular-Medicine/HolOrama.git
   cd HolOrama
   bash install.sh

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Flag
     - Effect
   * - ``--dev``
     - Also install the development and documentation dependencies (linters, test runner,
       Sphinx)
   * - ``--nnuzoo``
     - Install nnUZoo from GitHub (required for automatic segmentation)
   * - ``--cpu``
     - CPU-only PyTorch instead of the CUDA build
   * - ``--cuda 121``
     - Use the CUDA 12.1 build instead of the default cu118

The usage line at the top of each install script lists the same options.

Running the application
~~~~~~~~~~~~~~~~~~~~~~~

With the virtual environment active:

.. code-block:: bash

   python3 src/main.py

The graphical interface opens maximised on the Intravascular module.

.. _install-manual:

Step-by-step installation (if the scripts fail)
-----------------------------------------------

Windows
~~~~~~~

**1. Visual C++ Redistributable**

Download and install the `Visual C++ Redistributable 2022 (x64)
<https://aka.ms/vs/17/release/vc_redist.x64.exe>`_ if it is not already present.

**2. Base install**

.. code-block:: powershell

   pip install uv
   uv sync
   .\.venv\Scripts\Activate.ps1

**3. Fix the missing LLVM OpenMP runtime (**\ ``libomp140.x86_64.dll``\ **)**

PyTorch 2.4.0 on Windows depends on ``libomp140.x86_64.dll``, which is not bundled in the
pip wheel. Run this once after installation:

.. code-block:: python

   import urllib.request, tarfile, io, os, sys

   url = 'https://conda.anaconda.org/conda-forge/win-64/llvm-openmp-14.0.0-h2d74725_0.tar.bz2'
   data = urllib.request.urlopen(url).read()
   dest = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib', 'libomp140.x86_64.dll')

   with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as t:
       f = t.extractfile('Library/bin/libomp.dll')
       with open(dest, 'wb') as out:
           out.write(f.read())
   print('Done:', dest)

.. note::
   This file is lost whenever torch is reinstalled; re-run the snippet afterwards.

**4. Fix the** ``optree`` **version incompatibility**

``optree >= 0.14`` is incompatible with ``torch 2.4.0`` and causes a C-level access
violation. Downgrade it:

.. code-block:: powershell

   uv pip install "optree==0.13.1"

**5. GPU acceleration (optional)**

Install the CUDA-enabled torch build matching your driver. With a CUDA driver ≤ 12.0
(check with ``nvidia-smi``) use the CUDA 11.8 build:

.. code-block:: powershell

   uv pip install --reinstall "torch==2.4.0+cu118" "torchvision==0.19.0+cu118" `
       --index-url https://download.pytorch.org/whl/cu118

After installing the CUDA build, re-run the ``libomp140.x86_64.dll`` snippet from step 3.

Linux / macOS
~~~~~~~~~~~~~

.. code-block:: bash

   pip install uv
   uv sync                      # creates .venv and installs all dependencies
   source .venv/bin/activate

Development and documentation dependencies:

.. code-block:: bash

   uv sync --group dev --group docs

nnUZoo, for automatic segmentation:

.. code-block:: bash

   uv pip install git+https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo@main

GPU (CUDA 11.8):

.. code-block:: bash

   uv pip install --reinstall "torch==2.4.0+cu118" "torchvision==0.19.0+cu118" \
       --index-url https://download.pytorch.org/whl/cu118

If you plan to use GPU acceleration, install the NVIDIA drivers and CUDA toolkit
beforehand:

.. code-block:: bash

   sudo apt update && sudo apt upgrade
   sudo apt install build-essential dkms
   sudo ubuntu-drivers autoinstall
   sudo reboot
   nvidia-smi  # verify the driver installation
   sudo apt install nvidia-cuda-toolkit

.. _install-vmtk:

Optional: vmtk, for CCTA centerlines
------------------------------------

**Calculate Centerlines** in the :doc:`CCTA module <modules/ccta>` drives `vmtk
<https://vmtk.github.io>`_, the Vascular Modelling Toolkit. vmtk is **not** bundled with
HolOrama. Its build is too involved to ship inside a binary, so installing it is left to
you. Everything else in the CCTA module works without it; only centerline computation is
disabled.

If you do not want to install vmtk, you can still use the Fusion module by supplying
centerlines (``.vtp``) computed elsewhere.

Current integration
~~~~~~~~~~~~~~~~~~~

HolOrama expects a **WSL-native Linux vmtk build** and invokes it through ``wsl.exe``.
Before each call it sources two scripts, exactly as a manual vmtk session would:

#. the vmtk Python venv's ``bin/activate``
#. the vmtk build's ``vmtk_env.sh``

and then runs ``vmtkcenterlines`` → ``vmtkcenterlinesmoothing`` → ``vmtksurfacewriter``.

Point HolOrama at your install in ``config.yaml``:

.. code-block:: yaml

   vmtk:
     venv_path: 'D:\path\to\vmtk-env'                 # the venv containing bin/activate
     build_path: 'D:\path\to\vmtk-build\Install'      # the build containing vmtk_env.sh
     wsl_distro: 'Ubuntu'                             # '' = whatever wsl.exe defaults to

``wsl_distro`` matters when several distributions are installed side by side: the default
one is not necessarily the one vmtk was built against.

Before running anything, HolOrama verifies that both activation scripts exist, that WSL
starts, and that all three vmtk executables resolve on ``PATH`` after sourcing them. If any
check fails you get a specific error message instead of a silent failure.

.. note::
   Because the runner always shells out to ``wsl.exe``, centerline computation currently
   requires Windows with WSL. On Linux, compute the centerlines with your native vmtk
   install and load the resulting ``.vtp`` files directly into the Fusion module.

Verifying the installation
--------------------------

- The window title reads **HolOrama Software** and three buttons (Intravascular, CCTA,
  Fusion) are visible on the left edge.
- **Help → Documentation** opens this site; **Help → GitHub Page** opens the repository.
- Open ``test_cases/patient_example`` with :kbd:`Ctrl+O`: the first frame of the example
  pullback should appear.
- Source installs log to ``./logs``; the packaged application logs to
  ``%LOCALAPPDATA%\HolOrama``. Attach the log when reporting a problem.

Building the binary yourself
----------------------------

The release binary is compiled with `Nuitka <https://nuitka.net/>`_ and wrapped by Inno
Setup. From a source checkout with the dev dependencies installed:

.. code-block:: powershell

   .\build_nuitka.ps1      # compile the standalone (inference-free) application
   .\build_installer.ps1   # wrap it into HolOrama-Setup-<version>.exe

The Inno Setup script is ``installer/HolOrama.iss``.
