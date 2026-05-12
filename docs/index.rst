.. AIVUS-CAA documentation master file, created by
   sphinx-quickstart on Tue Sep 9 10:57:43 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================
AIVUS-CAA documentation
========================

*AIVUS-CAA: Automated IntraVascular UltraSound image processing and coronary artery analysis.* 

AIVUS-CAA is a Python application for intravascular ultrasound (IVUS) image processing. It enables frame-by-frame inspection of IVUS sequences and quantitative analysis of lumen contours. Users can manually draw contours or use automatic tools to measure lumen area and other metrics. 

.. warning::
   This software is provided *as is*, for research use only. Users should independently verify all results.

.. include:: contents/key_features.rst
.. include:: contents/related_projects.rst

.. toctree::
   :caption: Getting Started
   :titlesonly:

   contents/key_features
   contents/installation

.. toctree::
   :caption: User Guide
   :titlesonly:

   contents/usage
   contents/tutorial
   contents/video_tutorial

.. toctree::
   :caption: Reference
   :titlesonly:

   contents/api_references
   CONTRIBUTING.md
   CHANGELOG.md
   contents/related_projects
   Github <https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA>

License
-------

This package is covered by the open source `MIT License <https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/blob/main/LICENSE>`_.

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
We'd welcome your contributions to AIVUS-CAA. Please read the
`contributing guidelines <https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/blob/main/CONTRIBUTING.md>`_ on how to contribute to multimodars.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`