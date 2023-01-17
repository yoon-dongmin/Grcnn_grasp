#!/usr/bin/env python
import sys
sys.path.append('/home/user/grasp_ws/src')
import os
import time
import rospy

import matplotlib.pyplot as plt
import numpy as np
import torch

from grcnn.srv import GraspPrediction, GraspPredictionResponse

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp
from tf.transformations import *


from grcnn.srv import GraspPrediction, GraspPredictionResponse


class GRCNNService:
    def __init__(self):
        self.saved_model_path = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97'
        self.camera = RealSenseCamera(device_id=143322072540)

        self.model = torch.load(self.saved_model_path) # load_model에서 불러옴
        self.device = get_device(force_cpu=False) # load_model에서 불러옴

        self.cam_data = CameraData(include_depth=True, include_rgb=True)

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        # self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        # self.cam_depth_scale = np.loadtxt('saved_data/camera_depth_scale.txt', delimiter=' ')
        self.cam_pose = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], dtype=float) #Identity matric로 생성
        self.cam_depth_scale = np.array([1.0]) #scale은 1
        rospy.Service('/predict', GraspPrediction, self.compute_service_handler) 





    def compute_service_handler(self, req):
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

        #quat = quaternion_from_euler(target_angle)
        # Concatenate grasp pose with grasp angle
        grasp_pose = np.append(target_position, target_angle[2]) #target_angle의 두번째원소가 rotation값

        print('grasp_pose: ', grasp_pose)

        ret = GraspPredictionResponse() # bool sussess 
        ret.success = True
        g = ret.best_grasp # 
        print(g,11111)
        g.position.x = target_position[0]
        g.position.y = target_position[1]
        g.position.z = target_position[2]
        # print(target_position[0],target_position[1],target_position[2],44444444)
        #print(target_angle,3333333333333)
        quat = quaternion_from_euler(0,0,target_angle[2])
        g.orientation.x = quat[0]
        g.orientation.y = quat[1]
        g.orientation.z = quat[2]
        g.orientation.w = quat[3]
        print(g,22222)

        return ret

if __name__ == '__main__':
    rospy.init_node('grcnn_service')
    GRCNN = GRCNNService()
    rospy.spin()