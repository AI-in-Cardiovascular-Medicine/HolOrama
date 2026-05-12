.. docs/contents/installation.rst

Installation
============

Basic
-----

.. code-block:: bash

    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install

Sometimes the nnUZoo can be problematic to install over GitHub, so as a default it is commented out
in ``pyproject.toml``. In this case the installation should be performed like this:

.. code-block:: bash

    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install
    poetry run pip install git+https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo@main

For developers download additionally the dev dependencies:

.. code-block:: bash

    poetry install --with dev

If you plan on using GPU acceleration for model training and inference, make sure to install the
required tools (NVIDIA toolkit, etc.) and the corresponding version of PyTorch/TensorFlow.

The program was tested on Ubuntu 22.04.5 with Python 3.10.12. We tested it on different hardware;
NVIDIA drivers and CUDA tended to cause problems cross-platform. Make sure to download the
corresponding drivers and CUDA toolkit, e.g.:

.. code-block:: bash

    sudo apt update
    sudo apt upgrade
    sudo apt install build-essential dkms
    sudo ubuntu-drivers autoinstall
    sudo reboot
    # verify the installation of the driver
    nvidia-smi
    sudo apt install nvidia-cuda-toolkit

Potentially extra steps are needed.

Windows
-------

Windows requires several manual fixes after installation due to packaging issues with PyTorch and
library conflicts.

1. Install Visual C++ Redistributable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download and install the `Visual C++ Redistributable 2022 (x64) <https://aka.ms/vs/17/release/vc_redist.x64.exe>`_
if not already present.

2. Fix missing LLVM OpenMP runtime (``libomp140.x86_64.dll``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch 2.4.0 on Windows depends on ``libomp140.x86_64.dll`` which is not bundled in the pip
wheel. Run this once after installation:

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

    This file will be lost if torch is reinstalled — re-run the script afterwards.

3. Fix ``optree`` version incompatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``optree >= 0.14`` is incompatible with ``torch 2.4.0`` and causes a C-level access violation.
Downgrade it:

.. code-block:: bash

    pip install "optree==0.13.1"

4. Fix torch + TensorFlow DLL conflict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When both torch and TensorFlow are loaded in the same process, their OpenMP runtimes conflict on
Windows. The fix is already applied in ``src/main.py`` via:

.. code-block:: python

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

This must appear before any torch or TensorFlow imports.

5. GPU acceleration (CUDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the CUDA-enabled torch build matching your driver. With CUDA driver ≤ 12.0
(check with ``nvidia-smi``), use the CUDA 11.8 build:

.. code-block:: bash

    pip install "torch==2.4.0+cu118" "torchvision==0.19.0+cu118" --index-url https://download.pytorch.org/whl/cu118

After installing the CUDA build, re-run the ``libomp140.x86_64.dll`` script from step 2.
Also uncomment the torch and torchvision entries in ``pyproject.toml`` (default is CPU only).

Precompiled version
-------------------

A precompiled version is pinned to the release (compiled with ``nuitka``).
To compile the project yourself:

.. code-block:: bash

    python -m nuitka --standalone --plugin-enable=pyqt6 --include-package=pydicom --include-package=scipy --include-package=numpy --follow-imports --show-progress main.py

Running the program
-------------------

After installation, run the main program with:

.. code-block:: bash

    python3 src/main.py

The graphical user interface (GUI) should appear. If you encounter issues, review the README or
submit an issue on GitHub.
