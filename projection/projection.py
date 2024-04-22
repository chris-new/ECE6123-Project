import numpy as np 
import os
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import time
from scipy.spatial import cKDTree

def mesh_to_point_cloud(mesh_path, num_points=1000):
    mesh = read_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def read_mesh(filename:str):
    #pcd = o3d.io.read_triangle_mesh(filename, True)
    pcd = o3d.io.read_triangle_model(filename, True)
    return pcd

def read_point_cloud(filename:str):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd


def render(geometries):
    o3d.visualization.draw_geometries(geometries)


# side effect: will set renderer_pc's extrinsic camera
# side effect: will set renderer_pc's extrinsic camera
def get_view(renderer_pc, intrinsic, model, center, eye, up=[0,1,0], extrinsic=None, img_width=400, img_height=500):


    # grey = o3d.visualization.rendering.MaterialRecord()
    # grey.base_color = [0.7, 0.7, 0.7, 1.0]
    # grey.shader = "defaultLit"
    
    material = o3d.visualization.rendering.MaterialRecord()
    # texture = np.asarray(model.colors).copy()
    # texture = texture.astype(np.float32)
    # texture = o3d.geometry.Image(texture)
    # material.albedo_img = texture
    material.aspect_ratio = 1.0
    material.shader = "defaultLit"



    # renderer_pc.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    renderer_pc.scene.set_lighting(renderer_pc.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    renderer_pc.scene.scene.enable_sun_light(True)

    renderer_pc.scene.add_geometry("model", model, material)
    
    # # just for reference 
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=([0., 0., 0.]))
    # renderer_pc.scene.add_geometry("axis", axis, material)

    
    # for triangle meshes
    # renderer_pc.scene.add_model("model", model)

    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 30  # between 5 and 90 degrees
    aspect_ratio = img_width / img_height  # azimuth over elevation
    near_plane = 0.05
    far_plane = 50.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    # renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)
    
    renderer_pc.scene.camera.set_projection(intrinsic.intrinsic_matrix, near_plane, far_plane, img_width, img_height)

    # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
    # camera orientation
    # extrinsic = lookat(np.asarray(center), np.asarray(eye), np.asarray(up))
    renderer_pc.scene.camera.look_at(center, eye, up)
    if extrinsic is not None:
        renderer_pc.setup_camera(intrinsic, extrinsic)
    # renderer_pc.setup_camera(intrinsic, extrinsic)

    depth_image = np.asarray(renderer_pc.render_to_depth_image(True) )
    image = np.asarray(renderer_pc.render_to_image())


    # plt.imshow(depth_image)
    renderer_pc.scene.remove_geometry("model")
    # renderer_pc.scene.remove_geometry("axis")
    return (image, depth_image, renderer_pc.scene.camera.get_view_matrix())


def lookat(center, eye, up):
    f = (eye - center)
    f /= np.linalg.norm(f)
    
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    
    u = np.cross(f, r)
    u /= np.linalg.norm(u)

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = np.column_stack((r, u, -f))
    view_matrix[:3, 3] = -np.dot(np.column_stack((r, u, -f)), eye)
    
    return view_matrix


def flip(matrix):
    result = np.copy(matrix)
    result[1:3,:] = -1 * result[1:3,:]
    return result


def rgbd_to_pointcloud(rgbd, intrinsic, extrinsic_camera):
    # extrinsic_camera = np.array(extrinsic_camera)
    # extrinsic_camera = np.linalg.inv(extrinsic_camera)
    pcd =  o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, flip(extrinsic_camera))
    # Flip it, otherwise the pointcloud will be upside down
    #pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return pcd


def downsample_rescale(pcd, voxel_size=0.00001, scale=10000):
    downsampled_pc = pcd.voxel_down_sample(voxel_size)
    # use mean as the center for now
    a1 = np.asarray(downsampled_pc.points)
    scaled_downsampled_pc = downsampled_pc.scale(scale, a1.mean(0))
    return scaled_downsampled_pc


def downsample(pcd, voxel_size=0.00001):
    downsampled_pc = pcd.voxel_down_sample(voxel_size)
    return downsampled_pc


def read_segmentation(file_name, depth):
    # for read npy files
    rgb = np.load(file_name).astype(np.uint8)
    rgb = o3d.geometry.Image(rgb)
    ## for read image files
    #rgb = o3d.io.read_image(file_name)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, o3d.geometry.Image(depth), convert_rgb_to_intensity=False, depth_scale=1)
    return rgbd_image


def project(model_path, centers, eyes, renderer_pc, intrinsic):
    model = read_point_cloud(model_path)
    #IDEA recenter and rescale the model 
    points = np.asarray(model.points)
    points /= 450
    points -= points.mean(0)
    model.transform(np.array([[1,0,0,0], 
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]))
    Image = []
    Depth = []
    Extrinsic = []
    for i in range(len(eyes)):
        image, depth, extrinsic = get_view(renderer_pc, intrinsic, model, center=centers[i], eye=eyes[i])
        Image.append(image)
        Depth.append(depth)
        Extrinsic.append(extrinsic)
    return Image, Depth, Extrinsic

def reproject(Masks, Depth, intrinsic, Extrinsic):
    # Pcd = []
    Pcd = o3d.geometry.PointCloud()
    for i in range(len(Masks)):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(Masks[i].astype(np.float32)), o3d.geometry.Image(Depth[i]), convert_rgb_to_intensity=False, depth_scale=1)
        pcd = rgbd_to_pointcloud(rgbd_image, intrinsic, Extrinsic[i])
        Pcd = pcd + Pcd
    return Pcd

def save_pcd(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)

def knn_for_all_conflict_points(Pcd):
    K = 100

    Pcd_points = np.asarray(Pcd.points)
    Pcd_colors = np.asarray(Pcd.colors)
    # knn for one point
    def knn(location):
        points = Pcd_points
        distances = np.linalg.norm(points - location, ord=2, axis=1)
        inds = np.argsort(distances)[:K]
        labels = Pcd_colors[inds]
        unique_labels, counts = np.unique(labels, axis=0, return_counts=True)
        return unique_labels[np.argsort(counts)[-1]]
    
    uniques, counts = np.unique(Pcd_points, axis=0, return_counts=True)
    duplicate_points = uniques[counts > 1]
    all_inds = np.arange(0, len(Pcd_points))
    print("Find {} duplicate points".format(len(duplicate_points)))
    for point in duplicate_points:
        inds = all_inds[np.all(Pcd_points == point, axis=1)]
        colors = Pcd_colors[inds]
        if len(np.unique(colors, axis=0)) == 1:
            continue
        else:
            label = knn(point)
        Pcd_colors[inds][:] = label
    
    pcd_ret = o3d.geometry.PointCloud()
    pcd_ret.points = o3d.utility.Vector3dVector(Pcd_points)
    pcd_ret.colors = o3d.utility.Vector3dVector(Pcd_colors)

    return pcd_ret

def knn(K, location, tree, colors):
        _, inds = tree.query(location, k=K)
        labels = colors[inds]
        # distances = np.linalg.norm(points - location, ord=2, axis=1)
        # inds = np.argsort(distances)[:K]
        # labels = colors[inds]
        unique_labels, counts = np.unique(labels, axis=0, return_counts=True)
        return unique_labels[np.argsort(counts)[-1]]