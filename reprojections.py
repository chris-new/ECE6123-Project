import os
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import subprocess
import numpy as np
import random
from scipy.spatial import cKDTree
from functools import reduce
from tqdm import tqdm
import pickle
import time
import re

from projection.projection import read_point_cloud, reproject, save_pcd, knn_for_all_conflict_points, mesh_to_point_cloud, recenter_model, read_segmentation2,rgbd_to_pointcloud2, knn
parser = argparse.ArgumentParser(description='get model projection via open3d headless rendering')

parser.add_argument('--path', type=str, default='/home/fengze/ECE6123-Project', help='path to the project')
parser.add_argument('--input', type=str, default='/home/huzi/code/pointersect-modified/dataset/THuman-uniform-800K/', help='path to the project')
parser.add_argument('--mask-input-path', type=str, default='/home/fengze/ECE6123-Project/cdcl/output/', help='path to the masks')
parser.add_argument('--img-width', type=int, default=1920, help='the width of each view')
parser.add_argument('--img-height', type=int, default=1080, help='the height of each view')
# parser.add_argument('--view-output-path', type=str, default='/home/fengze/ECE6123-Project/cdcl/input/', help='path where the view outputs to be')
parser.add_argument('--depth-input-path', type=str, default='/home/fengze/ECE6123-Project/depths/', help='path where the depth inputs are')
parser.add_argument('--extrinsic-input-path', type=str, default='/home/fengze/ECE6123-Project/extrinsics/', help='path where the extrinsics inputs are')
parser.add_argument('--num-views', type=int, default=10, help="number of views")
parser.add_argument('--k', type=int, default=500, help="knn coefficient")
parser.add_argument('--output-path',type=str, default='/home/fengze/ECE6123-Project/output', help='output path of the point clouds')
    
def list_all_files(directory):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files

def main(args):
    files = list_all_files(args.input)
    models = [file for file in files if file.endswith('.ply')]
    
    # print("Found models: {}".format(models))
    print(f"total {len(models)} models")
    intrinsic =  o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)
    
    for model in tqdm(models, desc="Processing", unit="model"):
        print("Processsing model {}".format(model))
        id = model.split(os.sep)[-2]
        model_point_cloud = read_point_cloud(model)
        points = np.asarray(model_point_cloud.points)
        points 
        points -= points.mean(0)
        model_point_cloud.transform(np.array([[1,0,0,0], 
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]))
        recenter_model(model_point_cloud)
        ############ 2D to 3D Repxrojection #############
        
        pcds = []
        logits = []
        for i in range(args.num_views):
            view_path = os.path.join(args.mask_input_path, f"{id}_image{i+1}_cdcl_segmask.npy")
            logit_path = os.path.join(args.mask_input_path, f'{id}_image{i+1}_cdcl_seg_output.npy"')
            depth_path = os.path.join(args.depth_input_path, f"{id}_depth{i+1}.npy")
            extrinsic_path = os.path.join(args.extrinsic_input_path, f"{id}_extrinsic{i+1}.npy")
            rgbd_image_seg = read_segmentation2(view_path=view_path, depth_path=depth_path)
            logit = read_segmentation2(view_path=logit_path, depth_path=depth_path)
            print(f"{id} get view from {id}_image{i+1}_cdcl_segmask.npy")
            pcd = rgbd_to_pointcloud2(rgbd_image_seg, intrinsic, extrinsic_path)   
            logit = rgbd_to_pointcloud2(logit, intrinsic, extrinsic_path)     
            pcds.append(pcd)
            logits.append(logit)
        assert(len(pcds) == args.num_views)
        Pcd_merged = knn_for_all_conflict_points(reduce(lambda x, y: x + y, pcds))
        seg_points = np.array(Pcd_merged.points)
        tree = cKDTree(seg_points)

        seg_colors = np.array(Pcd_merged.colors)
        points = np.asarray(model_point_cloud.points)
        colors = []
        for point in tqdm(points, desc="Processing", unit="point") :
            color = knn(args.k, point, tree, seg_colors)
            colors.append( color)
            
        colors_stacked = np.vstack(colors)

        pcd_result = o3d.geometry.PointCloud()
        pcd_result.points = o3d.utility.Vector3dVector(points)
        pcd_result.colors = o3d.utility.Vector3dVector(colors_stacked)
        o3d.io.write_point_cloud(os.path.join(args.output_path,f'{id}_cdcl_pcd_result.ply'),pcd_result )
        print(f"{id} result point cloud output to {args.output_path}")
if __name__ == '__main__':
    FLAGS = parser.parse_args()
    main(FLAGS)