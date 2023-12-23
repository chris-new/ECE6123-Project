import numpy as np 
import os
import open3d as o3d
import matplotlib.pyplot as plt
import time

def read_mesh(filename:str):
    #pcd = o3d.io.read_triangle_mesh(filename, True)
    pcd = o3d.io.read_triangle_model(filename, True)
    return pcd


def render(geometries):
    o3d.visualization.draw_geometries(geometries)


# side effect: will set renderer_pc's extrinsic camera
def get_view(renderer_pc, intrinsic, model, center, eye, extrinsic=None, img_width=400, img_height=500):

    # grey = o3d.visualization.rendering.MaterialRecord()
    # grey.base_color = [0.7, 0.7, 0.7, 1.0]
    # grey.shader = "defaultLit"
    
    # material = o3d.visualization.rendering.MaterialRecord()
    # texture = np.asarray(model.textures[0]).copy()
    # texture = o3d.geometry.Image(texture)
    # material.albedo_img = texture
    # material.aspect_ratio = 1.0
    # material.shader = "defaultLit"
    
    renderer_pc.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    renderer_pc.scene.scene.enable_sun_light(True)

    #renderer_pc.scene.add_geometry("pcd", model, material)
    
    renderer_pc.scene.add_model("model", model)

    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 15.0  # between 5 and 90 degrees
    aspect_ratio = img_width / img_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 50.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    #renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)
    renderer_pc.scene.camera.set_projection(intrinsic.intrinsic_matrix, near_plane, far_plane, img_width, img_height)

    # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
    up = [0, 1, 0]  # camera orientation
    # extrinsic = lookat(np.asarray(center), np.asarray(eye), np.asarray(up))
    renderer_pc.scene.camera.look_at(center, eye, up)
    if extrinsic is not None:
        renderer_pc.setup_camera(intrinsic, extrinsic)
    # renderer_pc.setup_camera(intrinsic, extrinsic)

    depth_image = np.asarray(renderer_pc.render_to_depth_image(True) )
    image = np.asarray(renderer_pc.render_to_image())


    # plt.imshow(depth_image)
    renderer_pc.scene.remove_geometry("model")
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


def project(model_path, img_width, img_height, intrinsic):
    img_width, img_height = (400, 500)
    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    model = read_mesh(model_path)
    
    Center = [[0.25,0,0], [-0.25,0,0], [-0.25,0,0], [0.25,0,0]]
    Eye = [[0, 0, 1.25], [0.5, 0, -1.25], [0.5, 0, -1.25], [0.75, 0, 0.75]]
    Image = []
    Depth = []
    Extrinsic = []
    for i in range(4):
        image, depth, extrinsic = get_view(renderer_pc, intrinsic, model, center=Center[i], eye=Eye[i])
        Image += image
        Depth += depth
        Extrinsic += extrinsic
    return Image, Depth, Extrinsic

def reproject(Masks, Depth, intrinsic, Extrinsic):
    # Pcd = []
    Pcd = o3d.geometry.PointCloud()
    for i in range(len(Masks)):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            Masks[i], o3d.geometry.Image(Depth[i]), convert_rgb_to_intensity=False, depth_scale=1)
        pcd = rgbd_to_pointcloud(rgbd_image, intrinsic, Extrinsic[i])
        Pcd += pcd
    return Pcd

def save_pcd(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd, True)

def knn_for_all_conflict_points(Pcd):
    K = 100

    # knn for one point
    def knn(location):
        points = Pcd.points
        distances = np.linalg.norm(points - location, ord=2, axis=1)
        inds = np.argsort(distances)[:K]
        labels = Pcd.colors[inds]
        unique_labels, counts = np.unique(labels, axis=0, return_counts=True)
        return unique_labels[np.argsort(counts)[-1]]
    
    uniques, counts = np.unique(Pcd.points, axis=0, return_counts=True)
    duplicate_points = uniques[counts > 1]
    all_inds = np.arange(0, len(Pcd.points))
    for point in duplicate_points:
        inds = all_inds[np.all(Pcd.points == point, axis=1)]
        colors = Pcd.colors[inds]
        if len(np.unique(colors, axis=0)) == 1:
            continue
        else:
            label = knn(point)
        Pcd.colors[inds][:] = label
    
    return Pcd