#!/usr/bin/env python

# Run this Python to run the whole pipeline with a cdcl 2D segmentation core.

import os
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['OPEN3D_CPU_RENDERING'] = 'true'

import argparse
import matplotlib.pyplot as plt


import open3d as o3d
import subprocess

from projection.projection import project, reproject, save_pcd, knn_for_all_conflict_points, mesh_to_point_cloud

setting_projection = {
    'img_width': 1920,
    'img_height' : 1080,
    'Mesh': "/scratch/yh3986/pointersect-modified-repo2/pointersect-modified-repo5/dataset/THuman-uniform-800K",
    'Output_Pcd': "./Outputs",
    'view_num': 4,
    'depth_path': "./projection/depth",
    'view_path': "./projection/view",
    'radius': 5
}
def list_all_files(directory):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files


def run_pipeline():
    
    files = list_all_files(setting_projection['Mesh'])

    # use the ply files
    models = [file for file in files if file.endswith('.ply')]
    
    print("Found models: {}".format(models))

    intrinsic =  o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)
    
    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(setting_projection['img_width'], setting_projection['img_height'])
    
    # generate the eye and center list from projection
    radius = setting_projection['radius']
    eyes = [
        [0,0,1.35],
        # [1*radius,0,1.35],
        # [0,1*radius,1.35],
        # [-1*radius,0,1.35],
        # [0,-1*radius,1.35],
        [0.71*radius,-0.71*radius,1.35],
        [0.71*radius,0.71*radius,1.35],
        [-0.71*radius,-0.71*radius,1.35],
        [-0.71*radius,0.71*radius,1.35],
        
        [0,0,-1.35],
        # [1*radius,0,-1.35],
        # [0,1*radius,-1.35],
        # [-1*radius,0,-1.35],
        # [0,-1*radius,-1.35],
        [0.71*radius,-0.71*radius,-1.35],
        [0.71*radius,0.71*radius,-1.35],
        [-0.71*radius,-0.71*radius,-1.35],
        [-0.71*radius,0.71*radius,-1.35],

    ]
    centers = []
    for i in len(eyes):
        centers.append([0,0,0])
    for model in models:
        print("Processsing model {}".format(model))

        ############ 3D to 2D Projection #############
        Image, Depth, Extrinsic = project(os.path.join(setting_projection['Mesh'], model), centers, eyes,renderer_pc, intrinsic)
        print("Retrived {} views of model {}".format(len(Image), model))
        
        
        
        ############ 2D Segmentation #############
        
        print("Passed to 2D segmentation")
        
        save_image_path = os.path.abspath(setting_projection.view_path)
        save_depth_path = os.path.abspath(setting_projection.depth_path)
        
        # first change pwd to make sure the 2d segmentation parser works properly
        module_path = os.path.dirname(os.path.abspath("./cdcl"))
        os.chdir(module_path)
        
        # save images to input folder of cdcl (because of the environment problem)
        # cdcl is using python3.6 while open3d is supporting python version >= 3.8
        for index, pair in enumerate(list(zip(Image, Depth))):
            save_view_name = f"{os.path.basename(model)[:-4]}_image{index}.png"
            save_depth_name = f"{os.path.basename(model)[:-4]}_depth{index}.png"
            plt.imsave(os.path.join(infer15.parser.input_folder, save_view_name), pair[0])
            plt.imsave(os.path.join(save_image_path, save_view_name), pair[0])
            plt.imsave(os.path.join(save_depth_path, save_depth_name), pair[1])  
        # call inference_15parts.py in command line after adjust conda environment               
        command_conda_cdcl = 'conda activate cdcl'
        command_cdcl_infer15 = 'python3 inference_15parts.py --scale=1 --scale=0.75 --scale=0.5'
        subprocess.run(command_conda_cdcl, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)       
        subprocess.run(command_cdcl_infer15, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # change back pwd
        os.chdir(os.path.dirname(module_path))
        print("2D segmentation completed")


        ############ 2D to 3D Projection #############
        print("Start reprojection")
        Pcd = reproject(combined_results, Depth, intrinsic, Extrinsic)
        print("Start merging conflict points")
        Pcd_merged = knn_for_all_conflict_points(Pcd)
        save_file_name = f"{os.path.basename(model)[:-4]}_pcd.pcd"
        save_pcd(Pcd_merged, os.path.join(setting_projection["Output_Pcd"], save_file_name))
        print("Output of {} saved to {}".format(model, save_file_name))

if __name__ == '__main__':
    run_pipeline()