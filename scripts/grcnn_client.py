#! /usr/bin/env python
import sys
sys.path.append('/home/user/grasp_ws/src/practice')
import rospy

import os
import time
import datetime
import moveit_commander

from std_msgs.msg import Int16
from geometry_msgs.msg import Twist, Pose
from franka_msgs.msg import FrankaState, Errors as FrankaErrors


#from msg import Grasp #경로 수정
from grcnn.srv import GraspPrediction #경로 수정


class PandaOpenLoopGraspController(object):
    """
    Perform open-loop grasps from a single viewpoint using the Panda robot.
    """
    def __init__(self):
        # Initialize moveit_commander and node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('d2_move_target_pose', anonymous=False)

        # Get instance from moveit_commander
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        # Get group_commander from MoveGroupCommander
        group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        #grcnn_service_name = '/grcnn_service'
        rospy.wait_for_service('/predict')
        grcnn_srv = rospy.ServiceProxy('/predict', GraspPrediction) #grasp정보 받음
        ret = grcnn_srv()
        print(ret.best_grasp)
        self.pose = ret.best_grasp
        #self.max_velo = 0.10 # max vel
        #self.best_grasp = Grasp() # grasp값

        #self.pc = PandaCommander(group_name='panda_arm_hand') #panda group

        #self.robot_state = None
        #rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

        #self.pregrasp_pose = [-0.03,-0.4,0.6,2**0.5/2,-2**0.5/2,0,0]



    def move_target_pose(self):
        self.move_group.set_pose_target(self.pose)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        quit()


if __name__ == '__main__':
    rospy.init_node('panda_grasp')
    a = PandaOpenLoopGraspController()
    a.move_target_pose()