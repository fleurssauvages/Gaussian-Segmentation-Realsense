#!/usr/bin/env python3
from sam2.build_sam import build_sam2_camera_predictor
from scripts.ellipsoid_calculation import build_gaussians_from_points
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import open3d as o3d
from collections import defaultdict

# ====== Helpers ======
class UIControl:
        def __init__(self):
            self.mode = 'init'
            self.objects = []          # list of dicts: {points, color}
            self.current_points = []
            self.current_color = tuple(np.random.randint(0, 255, size=3).tolist())

        def next_object(self):
            if self.current_points:
                self.objects.append({
                    'points': self.current_points.copy(),
                    'color': self.current_color
                })
                print(f"✅ Finalized object {len(self.objects)} with {len(self.current_points)} points.")
                self.current_points.clear()
                self.current_color = tuple(np.random.randint(0, 255, size=3).tolist())

        def mouse_callback(self, event, x, y, flags, param):
            if self.mode != 'init':
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append((x, y))
                print(f"Added point ({x}, {y}) for current object.")
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.next_object()

def show_mask(mask, frame, color=None):
    if color is None:
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    overlay = frame.copy()
    mask_indices = mask.astype(bool)
    overlay[mask_indices] = color
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    return frame, color

def preprocess_frame(frame, device="cuda"):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

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

def project_ellipsoid(mu, cov, intr, n_points=100):
    """
    Projects a 3D Gaussian ellipsoid to 2D image coordinates.
    Returns 2D contour points (Nx2).
    """
    # Eigen decomposition for axes
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes = np.sqrt(np.maximum(eigvals, 1e-12))
    
    # Sample sphere points
    phi = np.linspace(0, np.pi, int(np.sqrt(n_points)))
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(n_points)))
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    sphere_points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    # Transform sphere → ellipsoid → world
    ellipsoid_points = sphere_points @ (eigvecs * axes) + mu

    # Project to image plane
    X, Y, Z = ellipsoid_points[:,0], ellipsoid_points[:,1], ellipsoid_points[:,2]
    u = intr.fx * X / Z + intr.ppx
    v = intr.fy * Y / Z + intr.ppy
    pts_2d = np.stack((u, v), axis=-1)
    
    # Keep only visible points
    mask = (Z > 0)
    pts_2d = pts_2d[mask]

    return pts_2d.astype(np.int32)

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
verbose=False

max_distance = 2.0 # meters

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

# ====== Load Model ======
sam2_checkpoint = "model/sam2_hiera_tiny.pt"  # path to SAM2 checkpoint
model_cfg = "sam2_hiera_t"  # pick model type (e.g., t, s, b, b+, l, l+)
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)
print("Model loaded on device:", device)

# ====== Main Loop ======
i = 0
frame_idx = 0
while True:
    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        continue
    
    # Process images
    aligned_depth_frame = hole_filling.process(aligned_depth_frame)
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    color_image = np.asanyarray(color_frame.get_data())
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    if i < 30: # Drop first 30 frames to allow auto-exposure to settle
        i += 1
        continue
    elif i == 30:
        i += 1
        predictor.load_first_frame(color_image)
        
        ui = UIControl()
        cv2.namedWindow("Select Points")
        cv2.setMouseCallback("Select Points", ui.mouse_callback)
        
        print("\n=== POINT SELECTION MODE ===")
        print("Left-click: add point")
        print("Right-click: next object")
        print("Press ENTER when done\n")
        
        while True:
            temp_img = color_image.copy()

            # Draw already finalized objects
            for obj in ui.objects:
                for (x, y) in obj['points']:
                    cv2.circle(temp_img, (x, y), 3, obj['color'], -1)

            # Draw current object in progress
            for (x, y) in ui.current_points:
                cv2.circle(temp_img, (x, y), 3, ui.current_color, -1)

            cv2.imshow("Select Points", temp_img)
            key = cv2.waitKey(20)
            if key == 13:  # Enter
                ui.next_object()  # finalize last object
                break
            elif key == 27:  # ESC
                print("Selection cancelled.")
                cv2.destroyAllWindows()
                exit(0)

        cv2.destroyWindow("Select Points")
            
        # now add all selected objects to predictor
        for obj_id, obj in enumerate(ui.objects, start=1):
            points = np.array(obj['points'], dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            color = obj['color']
            print(f"Adding object {obj_id} with {len(points)} points, color={color}")
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, points=points, labels=labels
            )
        object_colors = {j: obj['color'] for j, obj in enumerate(ui.objects)}

    else:
        frame_idx += 1
    
    out_obj_ids, out_mask_logits = predictor.track(color_image)
    
    # Generate masks
    for j, mask in enumerate(out_mask_logits):
        mask = (mask > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        mask = mask[:, :, 0]
        # Visualize
        color_bgr = object_colors[j]
        color_image, color_bgr = show_mask(mask, color_image, color=color_bgr)

    gaussians = defaultdict(list)
    # Fit Gaussians
    for j, mask in enumerate(out_mask_logits):
        mask = (mask > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        mask = mask[:, :, 0]
        depth_masked = np.where(mask, depth_image, 0).astype(np.float32)
        points_3d = []
        # Vectorized conversion of valid depth pixels to 3D points
        depth_m = depth_masked.astype(np.float32) * depth_scale  # meters
        valid = (depth_m > 0) & (depth_m <= max_distance)
        if np.any(valid):
            vs, us = np.nonzero(valid)        # row (v) and col (u) indices
            zs = depth_m[vs, us]
            xs = (us - intr.ppx) * zs / intr.fx
            ys = (vs - intr.ppy) * zs / intr.fy
            points_3d = np.stack((xs, ys, zs), axis=-1)
        else:
            points_3d = np.empty((0, 3), dtype=np.float32)
        points_3d = np.array(points_3d)
        if points_3d.shape[0] < 10:
            continue
        mus, covs_scaled, weights, radii = build_gaussians_from_points(
            points_3d,
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
        gaussians[j] = list(zip(mus, covs_scaled))
    
    print(f"Generated {len(out_mask_logits)} masks")
    print(f"Fitted Gaussians per object: " + ", ".join([f"{len(gaussians[j])}" for j in range(len(out_mask_logits))]))

    cv2.imshow("Segmented Image", color_image)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # ESC to exit
        cv2.destroyAllWindows()
        break
