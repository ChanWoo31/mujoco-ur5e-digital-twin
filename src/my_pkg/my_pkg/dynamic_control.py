#!/usr/bin/env python3

import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32MultiArray

import mujoco
import mujoco.viewer

xml_path = 'universal_robots_ur5e/scene_motor.xml'

def deg2rad(x):
    y = x * np.pi / 180
    return y

class Ur5eRun(Node):
    def __init__(self):
        super().__init__('Ur5e_Run')
        self.ur5e_run = self.create_subscription(UInt32MultiArray, 'topic', self.subscribe_topic, 10)
        
        
        self.d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
        self.a = [0, -0.425, -0.3922, 0, 0, 0]
        self.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

        self.m_mujoco = mujoco.MjModel.from_xml_path(xml_path)
        self.d_mujoco = mujoco.MjData(self.m_mujoco)
        self.renderer = mujoco.Renderer(self.m_mujoco)

        # x_target은 다이나믹셀(목표값), x_current는 Ur5e
        self.dynamixel_angle = np.zeros(6)
        self.x_target = np.eye(4)
        self.x_current = np.eye(4)

        self.q_new = np.zeros(6)
        
        # 수정.
        self.d_mujoco.qpos[:6] = [0, 0, 0, 0, 0, 0]
        mujoco.mj_forward(self.m_mujoco, self.d_mujoco)

    def subscribe_topic(self, msg):
        q = msg.data
        dir = [1, 1, -1, 1, 1, 1]
        offset = [0, -90, 0, -90, 0, 180]
        
        for i in range(6):
            self.dynamixel_angle[i] = deg2rad((q[i] - 2048) * dir[i] * 360 / 4096 + offset[i])
            
        # x_target은 다이나믹셀(목표값), x_current는 Ur5e
        self.x_target = self.forward_kinematic(self.dynamixel_angle, self.d, self.a, self.alpha)
        
    def control_step(self):
        delta_x_1d = np.zeros(6)
        kp = 1
        ko = 1
        epsilon = 0.02
        damping_factor_max = 0.01

        self.q_current = self.d_mujoco.qpos[:6]
        self.x_current = self.forward_kinematic(self.q_current, self.d, self.a, self.alpha)
        
        error_position = self.x_target[:3, 3] - self.x_current[:3, 3]
        delta_x_1d[:3] = np.transpose(kp * error_position)

        # scipy 이용해서 쿼터니언 구하기 (기본값: [x, y, z, w] 순서.)
        rot_mat_target = R.from_matrix(self.x_target[:3, :3])
        rot_mat_current = R.from_matrix(self.x_current[:3, :3])
        quat_target = rot_mat_target.as_quat()
        quat_current = rot_mat_current.as_quat()

        if np.dot(quat_target, quat_current) < 0:
            quat_current = -quat_current
        
        error_orientation = quat_current[3] * quat_target[:3] - quat_target[3] * quat_current[:3] - np.cross(quat_target[:3], quat_current[:3])
        delta_x_1d[3:6] = np.transpose(ko * error_orientation)

        delta_x_1d_trans = np.transpose(delta_x_1d)
        jacobian_matrix = self.get_jacobian(self.q_current, self.d, self.a, self.alpha)
        # j_inv = np.linalg.pinv(jacobian_matrix)
        # delta_q = j_inv @ delta_x_1d_trans

        # damped least squares를 추가. (계속 휘릭휘릭 튀는 현상이 특이점 때문인거 같아서.)
        U, sigma, Vt = np.linalg.svd(jacobian_matrix)

        if sigma[-1] >= epsilon:
            damping_factor = 0
        else:
            damping_factor = damping_factor_max * np.sqrt(1 - (sigma[-1] / epsilon)**2)

        delta_q = np.transpose(jacobian_matrix) @ np.linalg.inv(jacobian_matrix @ np.transpose(jacobian_matrix) + (damping_factor**2) * np.eye(6)) @ delta_x_1d_trans

        self.q_new = self.q_current + delta_q

    def forward_kinematic(self, theta, d, a, alpha):
        T = np.eye(4)
        for i in range(6):
            T_i = self.Trans_mat(theta[i], d[i], a[i], alpha[i])
            T = T @ T_i
        return T

    def get_jacobian(self, theta, d, a, alpha):
        T01 = self.Trans_mat(theta[0], d[0], a[0], alpha[0])
        T12 = self.Trans_mat(theta[1], d[1], a[1], alpha[1])
        T23 = self.Trans_mat(theta[2], d[2], a[2], alpha[2])
        T34 = self.Trans_mat(theta[3], d[3], a[3], alpha[3])
        T45 = self.Trans_mat(theta[4], d[4], a[4], alpha[4])
        T56 = self.Trans_mat(theta[5], d[5], a[5], alpha[5])
        T02 = T01 @ T12
        T03 = T02 @ T23
        T04 = T03 @ T34
        T05 = T04 @ T45
        T06 = T05 @ T56
        
        # 왜 외적하면 1차원으로 나오지
        Jv1 = np.cross(np.transpose(np.array([0, 0, 1])) , T06[:3, 3])
        Jomega1 = np.eye(3) @ np.transpose(np.array([0, 0, 1]))
        Jv2 = np.cross(T01[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T01[:3, 3])
        Jomega2 = T01[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv3 = np.cross(T02[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T02[:3, 3])
        Jomega3 = T02[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv4 = np.cross(T03[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T03[:3, 3])
        Jomega4 = T03[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv5 = np.cross(T04[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T04[:3, 3])
        Jomega5 = T04[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv6 = np.cross(T05[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T05[:3, 3])
        Jomega6 = T05[:3, :3] @ np.transpose(np.array([0, 0, 1]))

        Jacobian_matrix = np.array([
            [Jv1[0], Jv2[0], Jv3[0], Jv4[0], Jv5[0], Jv6[0]],
            [Jv1[1], Jv2[1], Jv3[1], Jv4[1], Jv5[1], Jv6[1]],
            [Jv1[2], Jv2[2], Jv3[2], Jv4[2], Jv5[2], Jv6[2]],
            [Jomega1[0], Jomega2[0], Jomega3[0], Jomega4[0], Jomega5[0], Jomega6[0]],
            [Jomega1[1], Jomega2[1], Jomega3[1], Jomega4[1], Jomega5[1], Jomega6[1]],
            [Jomega1[2], Jomega2[2], Jomega3[2], Jomega4[2], Jomega5[2], Jomega6[2]],
        ])
        
        return Jacobian_matrix

    def Trans_mat(self, theta, d, a, alpha):
        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
             [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
             [0, np.sin(alpha), np.cos(alpha), d],
             [0, 0, 0, 1]])
        return T
     
def main():
    rclpy.init()
    node=Ur5eRun()
    with mujoco.viewer.launch_passive(node.m_mujoco, node.d_mujoco) as viewer:
            cam_id = mujoco.mj_name2id(node.m_mujoco, mujoco.mjtObj.mjOBJ_CAMERA, 'robotview1')
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
            while viewer.is_running():
                step_start = time.time()

                rclpy.spin_once(node, timeout_sec=0)
                node.control_step()

                node.d_mujoco.ctrl[:6] = node.q_new

                mujoco.mj_step(node.m_mujoco, node.d_mujoco)

                viewer.sync()

                # 기본적인 시간 측정 기능.
                time_until_next_step = node.m_mujoco.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()