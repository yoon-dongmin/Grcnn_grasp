#-*- encoding: utf-8 -*-
import tf
import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from moveit_commander.conversions import pose_to_list, list_to_pose

def pose_list_to_mat(pose_list):
    trans_mat = tf.transformations.translation_matrix(pose_list[:3])
    rot_mat = tf.transformations.quaternion_matrix(pose_list[3:])
    pose_mat = tf.transformations.concatenate_matrices(trans_mat, rot_mat)
    return pose_mat


a = [-0.03,-0.4,0.6,2**0.5/2,-2**0.5/2,0,0]
b = pose_list_to_mat(a)
print(b)