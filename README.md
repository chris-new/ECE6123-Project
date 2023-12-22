# 3D human parsing via multi-camera 2D analysis and 2D-to-3D reprojection

## Group Members
- Enji Hu (eh3094@nyu.edu)
- Chen Yang (cy2478@nyu.edu)
- Fengze Zhang (fz2244@nyu.edu)
- Mentor: Yueyu Hu (yh3986@nyu.edu)

## Notes
The Jypyter notebook "Experiment.ipynb" was only used to experiment with the SCHP 2D segmentation model. It is not a part of our automated pipeline. However, the code in this notebook is the same as the code in this repository except minor changes.

The 3D data we used for testing were provided by our mentor.

Some experiment results can be found under the "ExperimentResults" folder.

## How to Run Our Code
### Environment Setup
Our pipeline is expected to be run on Linux with Python 3.8.

For the 2D segmentation model SCHP, we use the paper authors' implementation. Their implementation requires a GPU to run. Their GitHub repository can be found [here](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing).

Run the following commands to install necessary dependencies.
```
pip install -r requirements.txt
git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git
```

### Run Pipeline
Download this repository to the computer.
```
git clone https://github.com/chris-new/ECE6123-Project.git
cd ECE6123-Project
mkdir Inputs
mkdir Outputs
```

Put input 3D data under the "Inputs" folder.

Run the python script.
```
python run_pipeline.py
```

The output data (3D segmentation results) will be under the "Outputs" folder. The 2D segmentation masks are saved under the "segmentation/2d_outputs" folder.
