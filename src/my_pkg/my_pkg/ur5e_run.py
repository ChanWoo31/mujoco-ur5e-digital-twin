#!/usr/bin/env python3

import time
import os
import numpy as np
from ur_analytic_ik import ur5e

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

import mujoco
import mujoco.viewer

home_dir = os.path.expanduser("~")
xml_path = os.path.join(home_dir, 'mujoco_menagerie/universal_robots_ur5e/scene.xml')

def deg2rad(x):
    y = x * np.pi / 180
    return y

class Ur5eRun(Node):
    def __init__(self):
        super().__init__('Ur5e_Run')
        self.ur5e_run = self.create_subscription(Int32MultiArray, 'topic', self.subscribe_topic, 10)
        
        
        self.l = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
        self.a = [0, -0.425, -0.3922, 0, 0, 0]
        self.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.renderer = mujoco.Renderer(self.m)
        self.angle = np.zeros(6)
        self.T06 = np.eye(4)

        self.target_angle = np.zeros(6)
        self.camera_to_base = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 1, -0.5],
            [0, 0, 0, 1]
        ])

        # 수정.
        self.d.qpos[:6] = [0, 0, 0, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    def subscribe_topic(self, msg):
        q = msg.data
        dir = [1, 1, -1, 1, 1, 1]
        offset = [0, -90, 0, -90, 0, 180]
        self.T = np.eye(4)
        for i in range(6):
            self.angle[i] = deg2rad((q[i] - 2048) * dir[i] * 360 / 4096 + offset[i])
            
        for i in range(6):
            self.T_i = self.Trans_mat(self.angle[i], self.l[i], self.a[i], self.alpha[i])
            # self.T06 = self.T06 @ self.T[i]
            self.T = self.T @ self.T_i
        
        # self.T06 = self.T @ np.array([
        #     [1, 0, 0, 0],
        #     [0, -1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])

        self.T06 = self.T
        # real_position = self.T[:3, 3]
        # real_orientation = self.T[:3, :3]
        # deg = 180
        # cam_orientation = np.array([
        #     [np.cos(deg2rad(deg)), -np.sin(deg2rad(deg)), 0],
        #     [np.sin(deg2rad(deg)), np.cos(deg2rad(deg)), 0],
        #     [0, 0, 1]
        # ])

        # target_position = cam_orientation @ real_position
        # target_orientation = cam_orientation @ real_orientation

        # self.T06[:3, :3] = target_orientation
        # self.T06[:3, 3] = target_position

        # self.T06 = self.T
        

        all_solutions = ur5e.inverse_kinematics(self.T06)
        
        if all_solutions is None or len(all_solutions) == 0:
            self.get_logger().warn("Ik 해 없음")
            return
        
        current_q = self.d.qpos[:6]

        best_sol = None
        min_dist = float('inf')

        for sol in all_solutions:
            dist = np.linalg.norm(np.array(sol) - current_q)
            if dist < min_dist:
                min_dist = dist
                best_sol = sol
        
        if best_sol is not None:
            self.target_angle = np.array(best_sol)

    def Trans_mat(self, theta, d, a, alpha):
        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
             [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
             [0, np.sin(alpha), np.cos(alpha), d],
             [0, 0, 0, 1]])
        return T
        
def main():
    rclpy.init()
    node=Ur5eRun()
    with mujoco.viewer.launch_passive(node.m, node.d) as viewer:
            cam_id = mujoco.mj_name2id(node.m, mujoco.mjtObj.mjOBJ_CAMERA, 'robotview1')
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
            while viewer.is_running():
                step_start = time.time()

                rclpy.spin_once(node, timeout_sec=0)
                # print(node.T06)

                node.d.ctrl[:6] = node.target_angle

                mujoco.mj_step(node.m, node.d)

                viewer.sync()

                # 기본적인 시간 측정 기능.
                time_until_next_step = node.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()