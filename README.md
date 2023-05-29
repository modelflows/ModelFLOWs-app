# ModelFLOWs-app
Welcome to the ModelFLOWs-app repository!

## Table of contents
* [About ModelFLOWs-app](#about-Modelflows-app)
* [Versions](#versions)
* [Requirements](#requirements)
* [Setup](#setup)
* [Run](#run)
* [Resources](#resources)
* [More info](#more-info)

## About ModelFLOWs-app
ModelFLOWs-app, an open source Software for data post-processing, patterns identification and development of reduced order models using modal decomposition and deep learning architectures.

## Versions
There are currently two versions available:
* ModelFLOWs-app desktop version
* ModelFLOWs-app web-browser version (demo)

The main files are also provided.
	
## Requirements
ModelFLOWs-app has been developed with:
* Python: 3.9

Other relevant libraries that require installation:
* tensorflow: 2.10
* scikit-learn: 1.2
* ffmpeg: latest version
* hdf5storage: latest version
* numba: 0.56.4
* scipy: 1.9.3
* keras-tuner: latest version
* protobuf: 3.20
	
## Setup
There are two ways to install all required libraries:

#### Option 1. Install all libraries:
```
$ cd ../v0.1_ModelFLOWs-app
$ sudo pip install -r Requirements.txt
```

#### Option 2. Install each library individually:
```
$ cd ../Desktop
$ sudo pip install **insert library name**
```

## Run
To open ModelFLOWs-app, run the following command:

#### For the desktop version:
```
$ cd ../v0.1_ModelFLOWs-app
$ python ModelFLOWs_app.py
```

#### For the web-browser demo:
```
$ cd ../v0.1_ModelFLOWs-app_web
$ streamlit run ModelFLOWs_web.py
```

## Resources
Check out the *Tutorial* folder for a more in depth tutorial explaining how to install, run and use ModelFLOWs-app. This folder also contains advanced tutorials explaining how each module of the application works.

## More info
For more information please visit ModelFLOWs-app's official website.[ModelFLOWs-app's official website](https://modelflows.github.io/modelflowsapp/). You can also find us on [LinkedIn](https://www.linkedin.com/in/company/modelflows/)
