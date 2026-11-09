.. docs/contents/related_projects.rst

Related projects
================

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Project
     - Relation to HolOrama
   * - `multimodars <https://pypi.org/project/multimodars>`_
     - The fusion engine. HolOrama's :doc:`modules/fusion` module is a graphical front end
       for it; use the package directly if you want to script the same pipeline.
   * - `nnUZoo <https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo>`_
     - Segmentation model zoo the **Automatic Segmentation** model was trained with. HolOrama
       ships it exported to ONNX, so nnUZoo is not needed to run it.
   * - `vmtk <https://vmtk.github.io>`_
     - The Vascular Modelling Toolkit, used for centerline extraction in the
       :doc:`modules/ccta` module. HolOrama installs it on first use, see
       :ref:`install-vmtk`.
