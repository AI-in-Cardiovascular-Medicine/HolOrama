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
     - Developers; Linux and macOS users
   * - Requirements
     - Windows x64, no Python, ~1 GB disk space
     - `uv <https://docs.astral.sh/uv/>`_ (the install scripts fetch it), Python 3.11–3.13
   * - Automatic AI segmentation
     - Included (CPU)
     - Included (CPU, or NVIDIA GPU with ``-Gpu``)

Both contain the same features. Only **Calculate Centerlines** in the CCTA module needs an
extra component, vmtk, which HolOrama downloads itself the first time you use it; see
:ref:`install-vmtk`.

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

.. _install-source:

Install from source
-------------------

The project uses `uv <https://docs.astral.sh/uv/>`_ for dependency management; every
dependency is pinned in ``uv.lock``. The install scripts only call ``uv sync``, so a manual
install is the same one command.

Windows
~~~~~~~

#. Install the `Visual C++ Redistributable 2022 (x64)
   <https://aka.ms/vs/17/release/vc_redist.x64.exe>`_ if it is not already present.

#. Clone the repository and run the install script from PowerShell:

   .. code-block:: powershell

      git clone https://github.com/AI-in-Cardiovascular-Medicine/HolOrama.git
      cd HolOrama
      .\install.ps1

Linux / macOS
~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/AI-in-Cardiovascular-Medicine/HolOrama.git
   cd HolOrama
   bash install.sh

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag (``install.ps1`` / ``install.sh``)
     - Effect
   * - ``-Dev`` / ``--dev``
     - Also install the development and documentation dependencies (tests, linters, Nuitka,
       Sphinx) and the ``pre-commit`` git hooks.
   * - ``-Gpu`` / ``--gpu``
     - Run **Automatic Segmentation** on an NVIDIA GPU (``segmentation-gpu`` extra instead of
       ``segmentation``). See :ref:`segmentation-gpu`.
   * - ``-NoSegmentation`` / ``--no-segmentation``
     - Skip the optional ``segmentation`` extra (``onnxruntime``). **Automatic Segmentation**
       then tells you it is unavailable; everything else works.
   * - ``-Centerlines`` / ``--centerlines``
     - Set up the vmtk environment for **Calculate Centerlines** right away instead of on
       first use (see :ref:`install-vmtk`).

Without the scripts, the equivalent is:

.. code-block:: bash

   uv sync --extra segmentation                              # users
   uv sync --extra segmentation-gpu                          # users with an NVIDIA GPU
   uv sync --extra segmentation --group dev --group docs     # developers

Running the application
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run src/main.py

(or activate ``.venv`` and run ``python src/main.py``; on Windows ``run.bat`` does the same).
The graphical interface opens maximised on the Intravascular module.

Automatic segmentation
----------------------

The IVUS lumen model (nnU-Net, U2Net-P architecture) is bundled as a 5 MB ONNX file and runs
with `ONNX Runtime <https://onnxruntime.ai>`_: no PyTorch and no nnU-Net installation. The
results match the original nnU-Net inference (including its test-time mirroring) to a few
pixels per pullback, on the CPU and on the GPU alike.

On the CPU expect a few seconds per frame (≈3.4 s on a 16-core desktop with mirroring, ≈1 s
with ``segmentation.use_mirroring: False``), so segment gated frames or a frame range rather
than whole pullbacks. See :doc:`configuration` for the settings and for using a custom
model.

.. _segmentation-gpu:

On an NVIDIA GPU (source install)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install with ``install.ps1 -Gpu`` / ``install.sh --gpu`` (or ``uv sync --extra
segmentation-gpu``). This installs ``onnxruntime-gpu`` together with the CUDA 12 and cuDNN
runtimes as pip packages (about 2.5 GB), so no CUDA toolkit has to be installed; the only
requirement is an NVIDIA driver ≥ 527.41 on Windows (≥ 525.60 on Linux), i.e. any driver
that reports *CUDA Version: 12.0* or higher in ``nvidia-smi``. With mirroring, a frame takes
≈0.2 s on an RTX 3060 laptop GPU, ~16× the CPU.

With ``segmentation.device: 'auto'`` (the default) HolOrama runs a test inference on the
GPU when the model is loaded and falls back to the CPU if that fails; the progress dialog
says which one is used and why the GPU was not. ``'cuda'`` refuses to run without the GPU
instead. The pre-built installer is CPU-only.

.. _install-vmtk:

vmtk, for CCTA centerlines
--------------------------

**Calculate Centerlines** in the :doc:`CCTA module <modules/ccta>` runs `vmtk
<https://vmtk.github.io>`_, the Vascular Modelling Toolkit. vmtk is only distributed through
conda-forge, so it cannot be part of HolOrama's own environment or of the compiled
application. Instead HolOrama keeps it in a separate environment managed by `pixi
<https://pixi.sh>`_:

- The first time you click **Calculate Centerlines**, HolOrama asks whether to set vmtk up.
  It downloads pixi (if it is not already installed) and the exact, pinned vmtk environment
  into ``%LOCALAPPDATA%\HolOrama\vmtk-env`` (``~/HolOrama/vmtk-env`` on Linux/macOS). This
  needs internet access, takes a few minutes and about 1.5 GB of disk space, and happens once.
- When a new HolOrama version pins a different vmtk, the environment is re-synced the same
  way on next use.
- vmtk then runs natively (no WSL, no compiling). Available on Windows x64, Linux x64 and
  Intel macOS; vmtk has no conda-forge build for Apple Silicon or Linux ARM.

The location and the pixi executable can be changed in ``config.yaml`` (``centerlines``
section, see :doc:`configuration`). From a source checkout, ``install.ps1 -Centerlines`` /
``install.sh --centerlines`` sets the environment up ahead of time.

Everything else in the CCTA module works without vmtk, and the Fusion module also accepts
centerlines (``.vtp``) computed elsewhere.

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
Setup. From a source checkout:

.. code-block:: powershell

   .\build_nuitka.ps1      # syncs its own build\venv (CPU onnxruntime + Nuitka), then compiles
   .\build_installer.ps1   # wrap it into HolOrama-Setup-<version>.exe

The Inno Setup script is ``installer/HolOrama.iss``. vmtk is not compiled in; the
application sets it up on first use from the pixi manifest it ships with (see
:ref:`install-vmtk`).

Updating the segmentation model
-------------------------------

The bundled ``src/segmentation/weights/ivus_lumen_u2netp.onnx`` was exported from the
trained nnU-Net checkpoint with ``scripts/model_export/export_onnx.py`` and checked against
nnU-Net's own predictor with ``scripts/model_export/verify_onnx.py``. Those two scripts are
the only part of the project that still needs PyTorch and nnUZoo; see
``scripts/model_export/README.md``.
