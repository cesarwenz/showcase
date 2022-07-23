#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import pandas as pd
import sys

trial_number = sys.argv[1]

def unwrapAngle(angle, prevAngle, prevAngle_2, phase):
    switch = angle - prevAngle
    if abs(switch) > 5.6:
        diff = abs(angle - prevAngle_2)
        if diff > 5.6:
            diff = phase - diff
        else:
            diff = diff  + 0.1
        if switch < 0:
            angle = angle + phase*np.floor((abs(angle - prevAngle) + diff)/(2*np.pi))
        else:
            angle = angle - phase*np.floor((abs(angle - prevAngle) + diff)/(2*np.pi))
    return angle


def controller_input(u, pub):
    vel_msg = Twist()
    vel_msg.linear.x = u[0]
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = u[1]
    pub.publish(vel_msg)

def states():
    msg = rospy.wait_for_message("/odometry/filtered_map", Odometry)
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    th = tf.transformations.euler_from_quaternion(quaternion)[2]
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    omega = msg.twist.twist.angular.z
    states = np.array([x, y, th, vx, vy, omega])[np.newaxis].T
    return states

def odom_pub(xd, pub):
    ref = Odometry()
    ref.header.stamp = rospy.Time.now()
    ref.header.frame_id = '/map'
    ref.child_frame_id = '/base_link'

    ref.pose.pose.position.x = xd[0]
    ref.pose.pose.position.y = xd[1]
    ref.pose.pose.position.z = 0

    odom_quat = tf.transformations.quaternion_from_euler(0, 0, xd[2])
    ref.pose.pose.orientation.x = odom_quat[0]
    ref.pose.pose.orientation.y = odom_quat[1]
    ref.pose.pose.orientation.z = odom_quat[2]
    ref.pose.pose.orientation.w = odom_quat[3]

    p_cov = np.array([0.0]*36).reshape(6,6)
    P = np.mat(np.diag([0.0]*3))
    p_cov = np.array([0.0]*36).reshape(6,6)

    # position covariance
    p_cov[0:2,0:2] = P[0:2,0:2]
    # orientation covariance for Yaw
    # x and Yaw
    p_cov[5,0] = p_cov[0,5] = P[2,0]
    # y and Yaw
    p_cov[5,1] = p_cov[1,5] = P[2,1]
    # Yaw and Yaw
    p_cov[5,5] = P[2,2]
    ref.pose.covariance = tuple(p_cov.ravel().tolist())
    pub.publish(ref)

n = 0

df = pd.read_csv('../Data/field_test_v4.csv')
waypoints_x = df['xd'].to_numpy()[n:]
waypoints_y = df['yd'].to_numpy()[n:]
theta = df['thetad'].to_numpy()[n:]
vel_x = df['dxd'].to_numpy()[n:]
vel_y = df['dyd'].to_numpy()[n:]
dtheta = df['dthetad'].to_numpy()[n:]
num_points = np.shape(waypoints_x)[0] 


if __name__ == '__main__':
    try:
        time_threshold = 5
        rospy.init_node('FBL', anonymous=True)
        velocity_publisher = rospy.Publisher('/icerobot_velocity_controller/cmd_vel', Twist, queue_size=10)
        reference_publisher = rospy.Publisher('/reference_trajectory', Odometry,  queue_size=10)
        n=6 # States
        m=2 # Controls
        dt = 0.2
        # v = 1
        kp_v = 0.8
        kd_v = 0.1
        kp_y = 0.3
        kp_omega = 0.6 # y velocity
        kd_omega = 0.1 # omega
        # Dynamic SSMR Parameters
        max_vel = 0.4
        max_turn = 0.3

        x_ic = states().flatten()
        xd_ic = states().flatten()

        xe_temp = lambda x, xd: np.array([
            [np.cos(x[2])*(xd[0]-x[0]) + np.sin(x[2])*(xd[1]-x[1])], 
            [-np.sin(x[2])*(xd[0]-x[0]) + np.cos(x[2])*(xd[1]-x[1])], 
            [xd[2]-x[2]]
        ])

        xed = lambda x, xd: np.array([
            [np.sqrt(x[3]**2 + x[4]**2)*np.cos(xe_temp(x, xd)[2]) + xd[5]*xe_temp(x, xd)[1] - np.sqrt((xd[3])**2 + (xd[4])**2)],
            [np.sqrt(x[3]**2 + x[4]**2)*np.sin(xe_temp(x, xd)[2]) - xd[5]*xe_temp(x, xd)[0]],
            [x[5]-xd[5]]
        ])

        xe = lambda x, xd: np.vstack((xe_temp(x, xd), xed(x, xd)))
        sat = lambda val, x: min(val, max(-val, x))
        t_f = num_points*dt

        t = np.arange(0, t_f, dt) 

        t_k = np.zeros(num_points+1)

        e = np.zeros((n, num_points+1))
        x_sys = np.zeros((n, num_points+1))
        angle = np.zeros((1, num_points+1))
        x_sys[:,0] = x_ic
        xd = np.zeros((n, num_points+1))
        u = np.zeros((m, num_points+1))
        u_tot = np.zeros((m, num_points+1))
        t0 = rospy.Time.now().to_sec()
        
        for k in range(num_points):
            t1 = rospy.Time.now().to_sec()
            x_ref = waypoints_x[k] # x_ref for circle traj
            y_ref = waypoints_y[k] # y_ref for circle traj
            theta_ref = theta[k] # theta_ref for circle traj
            dx_ref = vel_x[k]
            dy_ref = vel_y[k]
            dtheta_ref = dtheta[k]

            xd[:,k+1] = np.vstack((x_ref, y_ref, theta_ref, dx_ref, dy_ref, dtheta_ref)).flatten()
            odom_pub(xd[:,k+1], reference_publisher)
            

            e[:,k+1] = xe(x_sys[:,k], xd[:,k]).T
            v = kp_v*e[0,k+1] + kd_v*e[3,k+1]
            v = sat(max_turn, v)
            omega = kp_omega*e[2,k+1] + kp_y*np.arctan(e[1,k+1]) + kd_omega*e[5,k+1]
            omega = sat(max_turn, omega)


            u[:,k+1] = [v, omega]
            controller_input(u[:,k+1], velocity_publisher)
            u_tot[:,k+1] = u[:,k+1] # control is from optimal auxiliary controller
            x_sys[:,k+1] = states().flatten()
            angle[:,k+1] = np.copy(x_sys[2,k+1])
            x_sys[2,k+1] = unwrapAngle(x_sys[2,k+1], x_sys[2,k], angle[0,k], 2*np.pi)
            t2 = rospy.Time.now().to_sec()
            t_k[k+1] = t2 - t0

            elapsed_time = t2 - t1
            
            rate = rospy.Rate(1/(dt-elapsed_time)) 
            rate.sleep()
            t2 = rospy.Time.now().to_sec()
            elapsed_time = t2 - t1
            print("execution time:{}".format(elapsed_time))
            if elapsed_time > time_threshold:
                quit()
    except rospy.ROSInterruptException:
        pass


df_t = pd.DataFrame({'time':t_k})
df_x_sys = pd.DataFrame({'x':x_sys[0],'y':x_sys[1], 'theta':x_sys[2], 'dx':x_sys[3],'dy':x_sys[4], 'dtheta':x_sys[5]})
df_xd = pd.DataFrame({'xd':waypoints_x,'yd':waypoints_y, 'thetad':theta})
df_u = pd.DataFrame({'ux':u[0],'uw':u[1]})
df1 = pd.concat([df_t, df_x_sys, df_xd, df_u], axis=1)
df1.to_csv('FBL_field'+ str(trial_number) + '.csv', index=False)
