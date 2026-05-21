# Proposed module structure after refactor
```tree
src--
    |- gating
    |- gui
    |- input_output
    |   |- input
    |   |- output
    |- segmentation
    |- data
    |   |- types
    |   |- math
    |- tools
```

- Clear seperation of concerns. 
- All data hold by the program during runtime is stored in it's own struct -> better control to clean up data
- input layer needs to be restructured: should check the data, decide for an input path and then use a clear strategy (DICOM, nifti etc.)
- output layer should especially seperate the math concepts from the writing -> can be reused throughout the program like this.
- The tools used for segmentation should be seperated from the main window. Like this they can be reused in different instances of windows.

# Progress update and thoughts
...