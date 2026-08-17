.. HolOrama documentation master file.

========================
HolOrama documentation
========================

*A unified platform for cardiac image analysis — segment, quantify and fuse intravascular
and coronary CT imaging.*

.. image:: https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/HolOrama/main/media/logo_holOrama.jpg
   :alt: HolOrama logo
   :align: center
   :width: 640px

HolOrama is a desktop application (PyQt6) for annotating and analysing cardiac images.
Its name mixes the greek *holo* ("full") and *orama* ("vision"): the promise of visualising
the whole heart from different modalities and, by fusing them, creating a full picture.

The application is organised into three modules, switched with the vertical navigation bar
on the far left of the window:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - What it does
   * - :doc:`Intravascular <contents/modules/intravascular>`
     - Frame-by-frame segmentation of IVUS and OCT pullbacks with several contour tools
       (lumen, EEM, calcium, side branch, lipid, macrophage), open/closed splines and
       explicit uncertainty markers. Exports contours and masks as NIfTI, ready for
       training an AI model. For IVUS it adds :doc:`image-based gating
       <contents/modules/gating>` and :doc:`breathing-motion detection and correction
       <contents/modules/breathing>`.
   * - :doc:`CCTA <contents/modules/ccta>`
     - Multi-label segmentation of coronary CT angiography with a brush tool, 3-D volume
       rendering, a lasso tool for removing structures, and construction of a combined
       aortic-root-and-coronaries model from user-defined cut planes — which can then be
       smoothed, decimated and (with a user-supplied vmtk install) turned into centerlines.
   * - :doc:`Fusion <contents/modules/fusion>`
     - A GUI wrapper around the `multimodars <https://pypi.org/project/multimodars>`_
       package: takes the cut CCTA geometry, its centerlines and an IVUS/OCT pullback and
       merges them into one geometry.

.. warning::
   This software is provided *as is*, for research use only. It is not a medical device.
   Users should independently verify all results.

New here? Start with the :doc:`contents/overview`, then :doc:`contents/installation`,
then the tutorial for the module you need.

.. toctree::
   :caption: Getting Started
   :titlesonly:

   contents/overview
   contents/installation
   contents/configuration

.. toctree::
   :caption: Modules & Tutorials
   :titlesonly:

   contents/modules/index

.. toctree::
   :caption: Reference
   :titlesonly:

   contents/shortcuts
   contents/outputs
   contents/api_references
   contents/related_projects
   CONTRIBUTING.md
   CHANGELOG.md
   Github <https://github.com/AI-in-Cardiovascular-Medicine/HolOrama>

Citation
--------

Please kindly cite the following papers if you use this software.

.. code-block:: bibtex

   @article{stark2025automated,
     title={Automated intravascular ultrasound image processing and quantification of
            coronary artery anomalies: the HolOrama software},
     author={Stark, Anselm W and Kazaj, Pooya Mohammadi and Balzer, Sebastian and Ilic, Marc
             and Bergamin, Manuel and Kakizaki, Ryota and Giannopoulos, Andreas and
             Haeberlin, Andreas and R{\"a}ber, Lorenz and Shiri, Isaac and others},
     journal={Computer Methods and Programs in Biomedicine},
     pages={109065},
     year={2025},
     publisher={Elsevier},
     doi={10.1016/j.cmpb.2025.109065},
     url={https://doi.org/10.1016/j.cmpb.2025.109065}
   }

   @article{kazaj2026unified,
     title  = {A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping:
               Human-in-the-Loop Data Annotation, Vision Foundation Model Development,
               Multicenter Evaluation and Clinical Validation},
     author = {Mohammadi Kazaj, Pooya and Weber, Leo Fridolin and Xie, Wen and
               Safavi-Naini, Seyed Amir Ahmad and Stark, Anselm and Baj, Giovanni and
               Mokhtari, Ali and Yoshida, Toshiya and Ryffel, Christoph and Okuno, Taishi and
               Akashi, Yoshihiro and Buechel, Ronny R. and Pilgrim, Thomas and Valenzuela, Waldo
               and Siontis, George C. M. and Xu, Xiaowei and Hundertmark, Moritz and
               Windecker, Stephan and Grani, Christoph and Shiri, Isaac},
     journal = {arXiv preprint arXiv:2607.11287},
     year   = {2026},
     doi    = {10.48550/arXiv.2607.11287},
     url    = {https://arxiv.org/abs/2607.11287}
   }

License
-------

This package is covered by the open source `MIT License
<https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/blob/main/LICENSE>`_.

Developers
----------

 - `Anselm Stark <https://github.com/yungselm>`_:sup:`1,2`
 - `Sebastian Balzer <https://github.com/cardionaut>`_:sup:`1,2`
 - `Pooya Mohammadi Kazaj <https://github.com/pooya-mohammadi>`_:sup:`1,2`
 - `Isaac Shiri <https://github.com/Isaacshiri>`_:sup:`1`

:sup:`1`\ Department of Cardiology, Inselspital, Bern University Hospital, University of Bern, Switzerland

:sup:`2`\ Graduate School for Cellular and Biomedical Sciences, University of Bern, Bern, Switzerland

Contributing
------------

Contributions are welcome. Please read the `contributing guidelines
<https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/blob/main/CONTRIBUTING.md>`_
before opening a pull request, and use the issue templates to
`report a problem <https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues/new?template=bug_report.md>`_
or `request a feature <https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues/new?template=feature_request.md>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
