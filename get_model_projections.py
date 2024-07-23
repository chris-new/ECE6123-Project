import os
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import subprocess
import numpy as np
import random


from tqdm import tqdm
import pickle
import time
import re

from projection.projection import project, reproject, save_pcd, knn_for_all_conflict_points, mesh_to_point_cloud, recenter_model
parser = argparse.ArgumentParser(description='get model projection via open3d headless rendering')

parser.add_argument('--path', type=str, default='/home/fengze/ECE6123-Project', help='path to the project')
parser.add_argument('--input', type=str, default='/home/huzi/code/pointersect-modified/dataset/THuman-uniform-800K/', help='path to the project')
parser.add_argument('--radius', type=float, default=0.8, help='radius of camera sphere')
parser.add_argument('--distance', type=float, default=2.6, help='distance of camera to the center')
parser.add_argument('--img-width', type=int, default=1920, help='the width of each view')
parser.add_argument('--img-height', type=int, default=1080, help='the height of each view')
parser.add_argument('--view-output-path', type=str, default='/home/fengze/ECE6123-Project/cdcl/input', help='path where the view outputs to be')
parser.add_argument('--depth-output-path', type=str, default='/home/fengze/ECE6123-Project/depths', help='path where the depth outputs to be')
parser.add_argument('--extrinsic-output-path', type=str, default='/home/fengze/ECE6123-Project/extrinsics', help='path where the extrinsics outputs to be')




    
def list_all_files(directory):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files

def main(args):
    files = list_all_files(args.input)
    models = [file for file in files if file.endswith('.ply')]
    
    print("Found models: {}".format(models))
    print(f"total {len(models)} models")
    intrinsic =  o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)
    
    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(args.img_width, args.img_height)
    radius = args.radius
    
    eyes = [
        [0,0,args.distance],
        # [1*radius,0,1.35],
        # [0,1*radius,1.35],
        # [-1*radius,0,1.35],
        # [0,-1*radius,1.35],
        [0.71*radius,-0.71*radius,args.distance],
        [0.71*radius,0.71*radius,args.distance],
        [-0.71*radius,-0.71*radius,args.distance],
        [-0.71*radius,0.71*radius,args.distance],
        
        [0,0,-args.distance],
        # [1*radius,0,-1.35],
        # [0,1*radius,-1.35],
        # [-1*radius,0,-1.35],
        # [0,-1*radius,-1.35],
        [0.71*radius,-0.71*radius,-args.distance],
        [0.71*radius,0.71*radius,-args.distance],
        [-0.71*radius,-0.71*radius,-args.distance],
        [-0.71*radius,0.71*radius,-args.distance],

    ]
    centers = []
    for i in range(len(eyes)):
        centers.append([0,0,0])
    for model in tqdm(models, desc="Processing", unit="model"):
        print("Processsing model {}".format(model))
        id = model.split(os.sep)[-2]
        
        
        ############ 3D to 2D Projection #############
        Image, Depth, Extrinsic = project(model, centers, eyes,renderer_pc, intrinsic, args.img_width, args.img_height)
        print("Retrived {} views of model {}".format(len(Image), model))

        for i in range(len(Image)):
            plt.imsave(os.path.join(args.view_output_path,f"{id}_image{i+1}.png"), Image[i])
            print(f"{id}: save view{i+1} to {args.view_output_path}")
            np.save(os.path.join(args.depth_output_path, f"{id}_depth{i+1}"), Depth[i])
            print(f"{id}: save depth{i+1} to {args.depth_output_path}")
            np.save(os.path.join(args.extrinsic_output_path, f"{id}_extrinsic{i+1}"), Extrinsic[i])
            print(f"{id}: save extrinsic{i+1} to {args.extrinsic_output_path}")

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    main(FLAGS)