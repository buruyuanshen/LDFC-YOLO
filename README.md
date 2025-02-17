# Project Introduction
    This project is established for the detection of cashmere and wool in electron microscope.
 
    It can extract features from cashmere and wool fibers and achieve high-precision detection through experiments.
 
# Environmental dependencies
    PyTorch 2.1.2
    CUDA 11.8
    torchvision 0.16.0
    numpy 1.26.4
    seaborn 0.13.2
 
# Description of the directory structure
    ├── ReadMe.md                // Help documentation
    
    ├── ultralytics              // The ultralytics folder
     
    │   ├── cfg                  // Storing configuration files

    │       ├── data             // Store data
    
    │       ├── models           // Model configuration
    
    │           ├── 11
    
    │           └── yolo11.yaml // YOLO11 Configuration files related to the target detection model

    │       ├── default.yaml    // Hyperparameter configuration file

    │   ├── nn                  // Basic module

    │       ├── addmodules

    │           ├── W-LDConv.py // Wool-Linear Deformable Convolution

    │           ├── LDFC.py     // Linear Deformable Feature Capture Module

    │       ├── tasks.py        //  Task definition and encapsulation
    
    │   ├── utils               // Tools and supporting documents
    
    │       ├── loss.py         // Defined loss function
    
    └── train.py                // Files for training purposes
 
# Instructions for use
 
    First configure the environment, then import W-LDConv and LDFC into the tasks.py file for registration, 
    create the configuration file and add the LDFC module, W-LDConv can also be combined with modules such as C3k2. 
    Then create the data set file. Finally, train using train.py.
 
# Version content update
    None

 
 
