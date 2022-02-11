#!/usr/bin/env python

#TODO clean up imports
import numpy as np
import copy

import cv2
from cv_bridge import CvBridge


import rospy
import tf
import tf2_ros
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge, CvBridgeError


from image_processing import preprocess_input, postprocess_output
from softdrone_target_pose_estimator.msg import Keypoints2D
# from pose_detection import make_transformation_matrix, transform_kpts, estimate_pose
import message_filters


class SoftDroneInterface():
    def __init__(self):
        # self.drone_controller = Geometric_Controller()
        self.target_center= None
        self.currentState = None
        self.target_location = np.array([2,0,2])
        self.trajectory_initialized = False
        self.kimera_ready = False
        self.count = 0

        rgb_camera_sub = message_filters.Subscriber("/camera/color/image_raw", ImageMsg)
        rgb_camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)
        depth_camera_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", ImageMsg)
        depth_camera_info_sub = message_filters.Subscriber("/camera/depth/camera_info", CameraInfo)
        # keypoints_sub = rospy.Subscriber("/tesse/keypoints", Keypoints2D, self.keypoints_cb)

        ts = message_filters.TimeSynchronizer([rgb_camera_sub, rgb_camera_info_sub,
                                               depth_camera_sub, depth_camera_info_sub], 10)
        ts.registerCallback(self.cameras_cb)

        self.process_img_pub = rospy.Publisher("/tesse/preproc_img", ImageMsg)
        self.kpts_img_pub = rospy.Publisher("/tesse/kpts_img", ImageMsg)

        self.bridge = CvBridge()
        self.last_rgb_img = None
        self.last_depth_img = None
        self.rgb_K = None
        self.depth_K = None
        self.is_ready = False

    def cameras_cb(self, rgb_img, rgb_info, depth_img, depth_info):
        bridge = CvBridge()
        rgb_img = bridge.imgmsg_to_cv2(rgb_img, "bgr8")
        depth_img = bridge.imgmsg_to_cv2(depth_img, "32FC1")

        # Publish preprocessed image to cpp NN node (temporary).
        img_preproc = preprocess_input(rgb_img)
        self.cur_preproc_img = img_preproc
        # Convert back into sensor message and publish for C++ node.
        msg_preproc = bridge.cv2_to_imgmsg(img_preproc, "bgr8")
        self.process_img_pub.publish(msg_preproc)

        # Save data for use in keypoints cb (temporary).
        self.last_rgb_img = rgb_img
        self.last_depth_img = depth_img
        self.rgb_K = rgb_info.K
        self.depth_K = depth_info.K
        self.is_ready = True

    def keypoints_cb(self, msg):
        
        # Something went wrong.
        if not self.is_ready:
            return

        num_kpts = len(msg.keypoints_2D)
        kpts_np = np.zeros((num_kpts, 2))
        # TODO change keypoint message so for loop is not needed.
        for i in range(num_kpts):
            row = np.array([msg.keypoints[i].x, msg.keypoints[i].y])
            kpts_np[i] = row

        # TODO: race condition.
        rescaled_kpts = postprocess_output(self.cur_preproc_img.shape,
                                           self.last_rgb_img.shape,
                                           kpts_np)

        self.kpts_px = rescaled_kpts

        # Publish annotated image.
        img = self.last_rgb_img.copy()
        for kpt in rescaled_kpts:
            cv2.circle(img, (kpt[0], kpt[1]), 3, [255,102,0], -1)

        kpts_img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        cv2.imshow("Keypoints", img)
        cv2.waitKey(1)
        self.kpts_img_pub.publish(kpts_img_msg)

        ## Convert keypoints from 2D pixels to world coordinates.
        #est_kpts_wrt_cam = np.zeros((rescaled_kpts.shape[0], 3))
        #K = np.asarray(self.depth_K).reshape((3,3))
        #for kpt_idx in range(len(rescaled_kpts)):
        #    kpt = rescaled_kpts[kpt_idx]
        #    #TODO make this into matrix equation.
        #    v = kpt[1]
        #    u = kpt[0]
        #    pz = self.last_depth_img[v, u]
        #    fx = K[0,0]
        #    fy = K[1,1]
        #    cx = K[0,2]
        #    cy = K[1,2]
        #    px = (u - cx) * pz / fx
        #    py = -(v - cy) * pz / fy
        #    est_kpts_wrt_cam[kpt_idx] = np.array([px, py, pz])

        ##TODO use ros tf.
        #pos = np.array([self.currentState.pose.pose.position.x,
        #                self.currentState.pose.pose.position.y,
        #                self.currentState.pose.pose.position.z])

        #quat = np.array([self.currentState.pose.pose.orientation.x,
        #                          self.currentState.pose.pose.orientation.y,
        #                          self.currentState.pose.pose.orientation.z,
        #                          self.currentState.pose.pose.orientation.w])

        #rot = (Rotation.from_quat(quat))
        #R = rot.as_dcm()
        #t = pos.reshape((3,))
        #gt_cam_wrt_world_tf = make_transformation_matrix(R, t)
        #est_kpts_wrt_world = transform_kpts(est_kpts_wrt_cam.T, gt_cam_wrt_world_tf)

        ## Target's tf relative to the keypoints CAD frame.
        #est_tar_wrt_src_pos, est_tar_wrt_src_rot = estimate_pose(est_kpts_wrt_world)
        #est_tar_wrt_src_tf = make_transformation_matrix(est_tar_wrt_src_rot, est_tar_wrt_src_pos)

        ## Estimated pose of target wrt camera frame.
        #gt_src_wrt_world_tf = np.eye(4)
        #est_tar_wrt_cam_tf = np.matmul(est_tar_wrt_src_tf, np.matmul(gt_src_wrt_world_tf, np.linalg.inv(gt_cam_wrt_world_tf)))
        #print est_tar_wrt_cam_tf







    #def depth_camera_info_cb(self, msg):
    #    self.depth_K = msg.K

    #def depth_camera_cb(self, msg):
    #    if self.depth_K is None or self.kpts_px is None:
    #        return
    #    # Convert sensor msg to cv.
    #    img = self.bridge.imgmsg_to_cv2(msg,"32FC1")




    #    print img
    #    cv2.imshow('depth image', img)
    #    cv2.waitKey(1)

    def rgb_camera_cb(self, msg):
        # TODO Image preprocessing should be done in C++ node.
        # Convert sensor msg to cv.
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        self.cur_img = img
        # Pre-process image for NN input.
        img_preproc = preprocess_input(img)
        self.cur_preproc_img = img_preproc
        # Convert back into sensor message and publish for C++ node.
        msg_preproc = bridge.cv2_to_imgmsg(img_preproc, "bgr8")
        self.process_img_pub.publish(msg_preproc)


        #print "img called"

        ##TODO dont hard code
        #if self.currentState is None:
        #    return

        #bridge = CvBridge()
        #img = bridge.imgmsg_to_cv2(msg, "bgr8")
        #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #lower_color_bound = np.array([20, 100, 100])
        #upper_color_bound = np.array([30, 255, 255])
        #mask = cv2.inRange(img_hsv, lower_color_bound, upper_color_bound)
        #yellow_cnts = cv2.findContours(mask.copy(),
        #                      cv2.RETR_EXTERNAL,
        #                      cv2.CHAIN_APPROX_SIMPLE)[-2]

        #if len(yellow_cnts) > 0:
        #    cnt = yellow_cnts[0]
        #    x,y,w,h = cv2.boundingRect(cnt)
        #    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #    self.target_center = ((x+w/2), (y+h/2))
        #    cv2.circle(img, self.target_center, 3,[255,102,0],-1)

        #def showImage(img):
        #    cv2.imshow('rgb_image', img)
        #    cv2.waitKey(1)

        #showImage(img)
        ## cv2.imwrite("/home/subella/test/medkit_image_{}.png".format(self.count), img)
        #self.count +=1



    def depth_Camera_cb(self, msg):

        if self.target_center is None:
            return

        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, "32FC1")

        #TODO dont harcode camera matrix, camera rotation
        K = np.array([415.69219381653056, 0.0, 360.0, 0.0, 415.69219381653056, 240.0, 0.0, 0.0, 1.0]).reshape((3,3))

        # TODO I dont understand conversion between unity camera frame and sofa
        # this rotation denotes 30deg along the y axis in an enu frame.
        # which is equivalent to what is being set in unity but in a DIFFERENT FRAME
        # FIX THIS!!
        R_c = Rotation.from_quat([0., 0.259, 0., 0.966]).as_dcm()

        u = self.target_center[0]
        v = self.target_center[1]
        pz = img[v, u]

        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        px = (u - cx) * pz / fx
        py = -(v - cy) * pz / fy
        p_w = np.array([px, py, pz])

        pos = np.array([self.currentState.pose.pose.position.x,
                        self.currentState.pose.pose.position.y,
                        self.currentState.pose.pose.position.z])

        rotation_quat = np.array([self.currentState.pose.pose.orientation.x,
                                  self.currentState.pose.pose.orientation.y,
                                  self.currentState.pose.pose.orientation.z,
                                  self.currentState.pose.pose.orientation.w])

        rot = (Rotation.from_quat(rotation_quat))
        # TEST THIS! Probably faces same problems as camera frame, but doesnt show up
        # because its almost identity
        R = rot.as_dcm()

        t = pos.reshape((3,1))
        T_quad_wrt_world = np.block([[R, t],[np.zeros((1,3)), 1]])
        # R_c=np.eye(3)
        T_camera_wrt_quad = np.block([[R_c, np.zeros((3,1))],[np.zeros((1,3)), 1]])

        p_world_wrt_camera = np.array([pz, px, py, 1])
        p_world = T_quad_wrt_world.dot(T_camera_wrt_quad.dot(p_world_wrt_camera))
        self.target_location = p_world[:3]
        print "World", p_world

        # print "depth", img[self.target_center[0], self.target_center[1]]
        # cv2.circle(img, self.target_center, 3,[255,102,0],2)
        # cv2.imshow('depth_image', img)
        # cv2.waitKey(1)
        # print "distance", img[self.target_center[0], self.target_center[1]]
        # print self.target_center

    def init_traj_controller(self, currentState):
        start = np.array([currentState.pose.pose.position.x,
                        currentState.pose.pose.position.y,
                        currentState.pose.pose.position.z])
        grasp_time = 3
        open_distance = 0.2
        grasp_target = self.target_location.copy()
        #grasp_target = np.array([2, 0, 1])


        #heuristic to try to get to 3d centroid
        #grasp_target += np.array([0.1, 0.075*grasp_target[1], -0.1])

        grasp_velocity = 0.5
        grasp_velocity = np.array([grasp_velocity, 0, 0])

        end_time = 3
        trajectories = [
        {
         "grasp_time": grasp_time, "grasp_position": grasp_target + [0, 0, 0.2],
         "grasp_velocity": grasp_velocity,
         "open_distance": open_distance,
         "grasp_args": {"L_r": 0.25}, "grasp_target": grasp_target,
         "open_args": {"L_r": 2.5, "log_level":3},
         "grasp_axis":0,
         # "end_time": end_time, "end_position": grasp_target + [8.0, 0, 8.0]},]
         "end_time": grasp_time, "end_position": grasp_target+ [1.0, 0, 1.0]},]
         # {
         # "grasp_time": 24, "grasp_position": [13,2,-0.3],
         # "grasp_velocity": grasp_velocity,
         # "open_distance": open_distance,
         # "grasp_args": {"L_r": 0.25}, "grasp_target": grasp_target,
         # "open_args": {"L_r": 2.5, "log_level":3},
         # "grasp_axis":0,
         # # "end_time": end_time, "end_position": grasp_target + [8.0, 0, 8.0]},]
         # "end_time": 24, "end_position": [16,2,3]},]

        start_time = currentState.header.stamp.to_sec()
        self.drone_controller = Trajectory_Controller(start_position=start,
                                                                    trajectories=trajectories,
                                                                    start_time=start_time)

        self.drone_controller.start()
        self.trajectory_initialized=True

    def on_gt_state(self, currentState):
        # if self.kimera_ready:
        #     return
        print "odom called"
        self.currentState = currentState
        # hover until we get target location
        wrench = self.drone_controller.do_step(currentState, goal_pos=0, goal_vel=0, goal_acc=0)
        lengths = [ [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744],
                        [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744],
                        [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744],
                        [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744]]

        # TODO replace message type
        control_msg = SoftDroneControlInput()


        control_msg.quadrotor_wrench.wrench = wrench
        # control_msg.finger_lengths.reduced_lengths = [lengths[0][0], lengths[0][2],
        #                                               lengths[2][0], lengths[2][2]]

        # control_msg.quadrotor_wrench.wrench = [0,0,0,0]
        control_msg.finger_lengths.reduced_lengths = [0, 0,
                                                      0, 0]
        #TODO replace this wait
        while(self.control_pub.get_num_connections() == 0):
            pass
                # print "waiting for control sub"
                # print self.odom_pub.get_num_connections()
        self.control_pub.publish(control_msg)

    def onCurrentState(self, currentState):
        #print "kimera cb called"
        # return
        self.kimera_ready = True
        self.currentState = currentState
        # hover until we get target location
        if self.target_location is None:
            wrench = self.drone_controller.do_step(currentState, goal_pos=0, goal_vel=0, goal_acc=0)
            lengths = [ [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744],
                            [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744],
                            [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744],
                            [0.19030669762326577, 0.19030669762326577, 0.2082230580323744, 0.2082230580323744]]
        else:
            if not self.trajectory_initialized:
                self.init_traj_controller(currentState)
            wrench = self.drone_controller.do_step(currentState)


        # TODO replace message type
        control_msg = SoftDroneControlInput()


        control_msg.quadrotor_wrench.wrench = wrench
        # control_msg.finger_lengths.reduced_lengths = [lengths[0][0], lengths[0][2],
        #                                               lengths[2][0], lengths[2][2]]

        # control_msg.quadrotor_wrench.wrench = [0,0,0,0]
        control_msg.finger_lengths.reduced_lengths = [0, 0,
                                                      0, 0]
        #TODO replace this wait
        while(self.control_pub.get_num_connections() == 0):
            pass
                # print "waiting for control sub"
                # print self.odom_pub.get_num_connections()
        self.control_pub.publish(control_msg)



if __name__ == "__main__":
    rospy.init_node("SoftDroneController_node")
    node = SoftDroneInterface()
    rospy.spin()
