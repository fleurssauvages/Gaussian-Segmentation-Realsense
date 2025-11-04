#!/usr/bin/env python3
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import open3d as o3d
from scripts.ellipsoid_calculation import build_gaussians_from_points

# ====== Helpers ======
class UIControl:
    def __init__(self):
        self.mode = 'init'  # init, select, track
        self.input_point = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.mode == 'init':
            self.input_point.append((x, y))
            print(f"Point selected: ({x}, {y})")

def show_mask(mask, frame, random_color=False):
    if random_color:
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    else:
        color = (30, 144, 255)
    overlay = frame.copy()
    mask_indices = mask.astype(bool)
    overlay[mask_indices] = color
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    return frame, color

def ellipsoid_from_gaussian(mu, cov, r=1.0):
    """
    Create an Open3D TriangleMesh ellipsoid corresponding to a Gaussian.
    mu: (3,)
    cov: (3,3)
    r: Mahalanobis radius scaling
    color: RGB list [0-1]
    """
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)  # avoid negatives

    # Semi-axes
    axes = r * np.sqrt(eigvals)

    # Create a unit sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=12)
    sphere.compute_vertex_normals()

    # Scale by axes lengths
    scale = np.diag(axes)
    R = eigvecs @ scale  # apply scale in eigenbasis
    sphere_vertices = np.asarray(sphere.vertices)
    sphere_vertices = (sphere_vertices @ R.T) + mu  # rotate + scale + translate
    sphere.vertices = o3d.utility.Vector3dVector(sphere_vertices)

    return sphere


# ====== Model Parameters ======
sam_checkpoint = "model/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# ====== Ellipsoid Parameters ======
K_ellipses = 4  # Number of ellipsoids to fit
prune_weight=1e-3 # Weight threshold for pruning
prune_points=10 # Minimum points per Gaussian
merge_thresh=0.05 # Mahalanobis distance for merging
radius_method='percentile' # Method for radius calculation
radius_percentile=99.0 # Percentile value
outlier_removal=True # Enable outlier removal
outlier_thresh=1.5 # Threshold for outlier removal
clamp_scale_min=0.5 # Minimum radius scale clamp for ellipsoids
clamp_scale_max=30.0 # Maximum scale clamp for ellipsoids
verbose=True

# ====== Image Parameters ======
font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0,150,255)
thickness = 1

# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0] # In this example we are only using one camera
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153 # Grey

# ====== Enable Streams ======
config.enable_device(device)
stream_res_x = 640
stream_res_y = 480
stream_fps = 15
config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

sensors = profile.get_device().query_sensors()
for sensor in sensors:
    if sensor.supports(rs.option.auto_exposure_priority):
        ae = sensor.set_option(rs.option.enable_auto_exposure, 1) 
        aep = sensor.set_option(rs.option.auto_exposure_priority, 0)

color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

K = [[intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]]
camera_matrix = np.array(K, dtype=np.float32)
axis_3D = np.float32([[0,0,0],
                    [0.25,0,0],  # X-axis
                    [0,0.25,0],  # Y-axis
                    [0,0,0.1]]) # Z-axis

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")# ====== Set clipping distance ======
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Create filters ======
hole_filling = rs.hole_filling_filter()

print("Press a key to keep the image for segmentation.")
i = 0
while True:
    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        continue
    
    if i < 30: # Drop first 30 frames to allow auto-exposure to settle
        i += 1
        continue
    
    # Process images
    aligned_depth_frame = hole_filling.process(aligned_depth_frame)
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    color_image = np.asanyarray(color_frame.get_data())
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    cv2.imshow("Image", color_image)
    k = cv2.waitKey(1)
    if k != -1:
        break
    
# ====== Load Model ======
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)
print("Model loaded successfully.")

# ====== Start UI ======
ui_control = UIControl()
cv2.imshow("Image", color_image)
cv2.setMouseCallback("Image", ui_control.mouse_callback)
print("Select points on the image. Press a key to go next.")
cv2.waitKey(0)

# Chose a point to start the segmentation
input_point = np.array(ui_control.input_point)
input_label = np.array([1] * len(ui_control.input_point))

# Initialize masks
predictor.set_image(color_image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

mask, score = masks[-1], scores[-1]
print(f"Mask selected with score: {score}")
frame, color_bgr = show_mask(mask, color_image, random_color=True)
cv2.imshow("Mask", frame)
cv2.waitKey(0)

# ====== Full scene point cloud (context) ======
h, w = depth_image.shape
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
depths_full = depth_image.astype(np.float32) * depth_scale

fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy

X_full = (xx - cx) * depths_full / fx
Y_full = (yy - cy) * depths_full / fy
Z_full = depths_full

valid = Z_full > 0
full_pts3d = np.vstack((X_full[valid], Y_full[valid], Z_full[valid])).T

# Get colors from the RGB image
color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
colors_full = color_image_rgb.reshape(-1, 3)[valid.flatten()] / 255.0  # normalize to [0,1]

# Create point cloud
pcd_full = o3d.geometry.PointCloud()
pcd_full.points = o3d.utility.Vector3dVector(full_pts3d)
pcd_full.colors = o3d.utility.Vector3dVector(colors_full)

# ====== Segmented cloud ======
ys, xs = np.where(mask)
points_3d = []

depths = depth_image[ys, xs] * depth_scale  # convert to meters

fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy

X = (xs - cx) * depths / fx
Y = (ys - cy) * depths / fy
Z = depths

# Stack into Nx3 points
points_segmented = np.vstack((X, Y, Z)).T  # flip Y to match Open3D coords
pcd_seg = o3d.geometry.PointCloud()
pcd_seg.points = o3d.utility.Vector3dVector(points_segmented)
pcd_seg.colors = o3d.utility.Vector3dVector(colors_full)

# ====== Fit Ellipsoids ======
print("Fitting ellipsoids to segmented point cloud...")
mus, covs_scaled, weights, radii = build_gaussians_from_points(
    points_segmented,
    K=K_ellipses,
    prune_weight=prune_weight,
    prune_points=prune_points,
    merge_thresh=merge_thresh,
    radius_method=radius_method, 
    radius_percentile=radius_percentile,
    outlier_removal=outlier_removal,
    outlier_thresh=outlier_thresh,  
    clamp_scale_min=clamp_scale_min,
    clamp_scale_max=clamp_scale_max,
    verbose=verbose
)
print("Fitting complete.")

# ====== Create ellipsoid meshes ======
meshes = []
for mu, cov in zip(mus, covs_scaled):
    ellipsoid = ellipsoid_from_gaussian(mu, cov, r=1.0)
    meshes.append(ellipsoid)

# ====== Visualize ======
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ====== Open3D Visualization Setup ======
# Create O3DVisualizer window
app = o3d.visualization.gui.Application.instance
app.initialize()
vis = o3d.visualization.O3DVisualizer("Gaussian Ellipsoids", 1600, 900)
vis.show_settings = True

# Add point cloud (as small dots)
mat_points = o3d.visualization.rendering.MaterialRecord()
mat_points.shader = "defaultUnlit"
mat_points.point_size = 3.0
vis.add_geometry("pointcloud", pcd_full, mat_points)

# Add each ellipsoid as a semi-transparent mesh
for i, (mu, cov) in enumerate(zip(mus, covs_scaled)):
    try:
        # Eigen-decomposition to get ellipsoid radii and orientation
        eigvals, eigvecs = np.linalg.eigh(cov)
        radii = np.sqrt(np.maximum(eigvals, 1e-10))

        # Create a sphere and transform it to an ellipsoid
        ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        ellipsoid.compute_vertex_normals()

        # Scale by radii
        scale = np.diag(radii)
        transform = np.eye(4)
        transform[:3, :3] = eigvecs @ scale
        transform[:3, 3] = mu
        ellipsoid.transform(transform)

        # Semi-transparent material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency"
        color_rgb = color_bgr[::-1] / 255.0
        mat.base_color = [color_rgb[0], color_rgb[1], color_rgb[2], 0.5]
        mat.base_roughness = 0.8
        mat.base_reflectance = 0.1

        vis.add_geometry(f"ellipsoid_{i}", ellipsoid, mat)
    except Exception as e:
        print(f"[Warning] Skipping ellipsoid {i}: {e}")

vis.show_settings = True
# vis.add_geometry(...) etc.

# ====== Compute camera view from RealSense intrinsics/extrinsics ======
intr = color_stream.as_video_stream_profile().get_intrinsics()
extrinsic = np.eye(4)

R = extrinsic[:3, :3]
t = extrinsic[:3, 3]

# In Open3D coordinates: camera looks toward -Z by default.
eye = t
center = t + R[:, 2] * 1.0     # or use -R[:, 2] depending on your depth direction
up = -R[:, 1]

# ====== Apply to O3DVisualizer ======
vis.reset_camera_to_default()

# "look_at" works in all GUI versions
vis.scene.camera.look_at(center, eye, up)

# Field of view setup — only if available
fov_x = 2 * np.degrees(np.arctan(intr.width / (2 * intr.fx)))
fov_y = 2 * np.degrees(np.arctan(intr.height / (2 * intr.fy)))
fov = float((fov_x + fov_y) / 2)

if hasattr(vis.scene.camera, "set_field_of_view"):
    vis.scene.camera.set_field_of_view(fov)
else:
    print(f"[Info] Skipping FOV (not supported in this Open3D version).")

# Zoom setup — also version-guarded
if hasattr(vis.scene.camera, "set_zoom"):
    vis.scene.camera.set_zoom(0.25)
else:
    print(f"[Info] Skipping zoom (not supported in this Open3D version).")

app = o3d.visualization.gui.Application.instance
app.initialize()
app.add_window(vis)
app.run()