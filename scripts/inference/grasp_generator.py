import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp


class GraspGenerator:
    def __init__(self, saved_model_path, cam_id, visualize=False):
        self.saved_model_path = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97'
        self.camera = RealSenseCamera(device_id=cam_id)

        self.model = None # load_model에서 불러옴
        self.device = None # load_model에서 불러옴

        self.cam_data = CameraData(include_depth=True, include_rgb=True)

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        # self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        # self.cam_depth_scale = np.loadtxt('saved_data/camera_depth_scale.txt', delimiter=' ')
        self.cam_pose = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], dtype=float) #Identity matric로 생성
        self.cam_depth_scale = np.array([1.0]) #scale은 1

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        print(homedir)
        self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        np.save(self.grasp_request,np.array([]))
        self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        np.save(self.grasp_available,np.array([]))
        self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")
        np.save(self.grasp_pose,np.array([]))

        if visualize: #false
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):
        # Get RGB-D image from camera
        # image_bundle의 rgb aligned_depth를 가져옴
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img) #파지점 추출

        # Get grasp position from model output
        pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale - 0.04
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)

        if pos_z == 0:
            return

        target = np.asarray([pos_x, pos_y, pos_z]) #position
        target.shape = (3, 1)
        print('target: ', target)

        # Convert camera to robot coordinates
        camera2robot = self.cam_pose #항등행렬
        target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:] 
        target_position = target_position[0:3, 0]
        print('target_position: ', target_position) #1x3 matrix

        # Convert camera to robot angle
        angle = np.asarray([0, 0, grasps[0].angle])
        angle.shape = (3, 1)
        target_angle = np.dot(camera2robot[0:3, 0:3], angle)
        print('target_angle: ',target_angle)

        # Concatenate grasp pose with grasp angle
        grasp_pose = np.append(target_position, target_angle[2]) #target_angle의 두번째원소가 rotation값

        print('grasp_pose: ', grasp_pose)

        np.save(self.grasp_pose, grasp_pose)

        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)

    def run(self):
        while True:
            print(np.load(self.grasp_request,allow_pickle=True),123123)
            self.generate()
            np.save(self.grasp_request, 0)
            np.save(self.grasp_available, 1)
            time.sleep(0.1)
            # if np.load(self.grasp_request, allow_pickle=True):
            #     print(222222)
            #     self.generate()
            #     np.save(self.grasp_request, 0)
            #     np.save(self.grasp_available, 1)
            # else:
            #     time.sleep(0.1)
