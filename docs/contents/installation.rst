.. docs/contents/installation.rst

Installation
============

Basic
-----

The project uses `uv <https://docs.astral.sh/uv/>`_ for dependency management.

Linux / macOS:

.. code-block:: bash

    ./install.sh

Windows (PowerShell):

.. code-block:: powershell

    .\install.ps1

Both scripts create a ``.venv``, install all dependencies, and apply any platform fixes automatically.
Run with ``--help`` (Linux) or ``Get-Help .\install.ps1`` (Windows) to see available options.

Options
~~~~~~~

.. code-block:: text

    --dev            install dev dependencies (linters, test runner)
    --nnuzoo         install nnUZoo from GitHub (required for automatic segmentation)
    --cpu            install CPU-only PyTorch (default is CUDA 11.8)
    --cuda 121       switch to CUDA 12.1 PyTorch build

Manual install
--------------

If you prefer not to use the install script:

.. code-block:: bash

    pip install uv
    uv sync                      # creates .venv and installs all dependencies
    source .venv/bin/activate    # Linux/macOS
    # .\.venv\Scripts\Activate.ps1  # Windows

For GPU (CUDA 11.8):

.. code-block:: bash

    uv pip install --reinstall "torch==2.4.0+cu118" "torchvision==0.19.0+cu118" \
        --index-url https://download.pytorch.org/whl/cu118

Windows notes
-------------

The install script handles all Windows-specific fixes automatically:

- Downloads ``libomp140.x86_64.dll`` required by PyTorch 2.4.0 (sourced from conda-forge).
- Pins ``optree==0.13.1`` to avoid a C-level crash with torch 2.4.0.
- ``KMP_DUPLICATE_LIB_OK=TRUE`` is set in ``src/main.py`` to prevent an OpenMP conflict when torch and TensorFlow are both loaded.

Install Visual C++ Redistributable 2022 (x64) if not already present before running the script.

Precompiled version
-------------------

A precompiled version is pinned to the release (compiled with ``nuitka``).
To compile the project yourself:

.. code-block:: bash

    python -m nuitka --standalone --plugin-enable=pyqt6 --include-package=pydicom \
        --include-package=scipy --include-package=numpy --follow-imports --show-progress main.py

Running the program
-------------------

After installation, run the main program with:

.. code-block:: bash

    python3 src/main.py

The graphical user interface (GUI) should appear. If you encounter issues, review the README or
submit an issue on GitHub.
