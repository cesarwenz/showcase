#%%
# #!/usr/bin/env python
import numpy as np
import rospy
import tf
import matplotlib.pyplot as plt
from numpy import sin, cos
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import pandas as pd
import sys

trial_number = sys.argv[1]

def controller_input(u, pub):
    vel_msg = Twist()
    vel_msg.linear.x = u[0]
    vel_msg.linear.y = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = u[1]
    pub.publish(vel_msg)

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


def wrapAngle(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

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

def bl_sat(s,phi):
    x = s/phi
    if np.abs(x) <= 1:
        y = x
    else:
        y = np.sign(x)
    return y

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
        time_threshold = 0.8
        rospy.init_node('SMC', anonymous=True)
        reference_publisher = rospy.Publisher('/reference_trajectory', Odometry, queue_size=10)
        velocity_publisher = rospy.Publisher('/icerobot_velocity_controller/cmd_vel', Twist, queue_size=10)
        lam = 1*np.array([1, 0.6, 0.9]) # Controls gains of sign(ye)theta_e, xed, and yed respectively 
        Q = 1e-1*np.array([1, 1]) #  s -- Controls the ability to reach the traj (Q1 is straight, Q2 is turning) -- (mostly control input)
        P = 1*np.array([1, 1])
        phi = np.array([1, 1])

        dt = 0.2
        # Constants 
        n = 6 # states
        m = 2 # controls

        N = 10

        t_f = num_points*dt
        t = np.arange(0, t_f, dt) 

        sat = lambda val, x: min(val, max(-val, x))

        # Dynamic SSMR Parameters
        max_vel = 0.4	
        max_turn = 0.3

        Q_kin = lambda x: np.array([[cos(x[2]), 0], [sin(x[2]), 0], [0, 1]])

        kin_sys = lambda x, u_kin: np.matmul(Q_kin(x), u_kin) #where x is x, y, theta and u is vx and omega
        v_x = lambda x1, x2: np.sqrt(max(1e-4, x1**2+x2**2))

        xe_temp = lambda x, xd: np.array([
            [np.cos(x[2])*(xd[0]-x[0]) + np.sin(x[2])*(xd[1]-x[1])], 
            [-np.sin(x[2])*(xd[0]-x[0]) + np.cos(x[2])*(xd[1]-x[1])], 
            [xd[2]-x[2]]
        ])

        xed = lambda x, xd: np.array([
            [np.sqrt(xd[3]**2 + xd[4]**2)*np.cos(xe_temp(x, xd)[2]) + x[5]*xe_temp(x, xd)[1] - np.sqrt((x[3])**2 + (x[4])**2)],
            [np.sqrt(xd[3]**2 + xd[4]**2)*np.sin(xe_temp(x, xd)[2]) - x[5]*xe_temp(x, xd)[0]],
            [xd[5]-x[5]]
        ])

        xe = lambda x, xd: np.vstack((xe_temp(x, xd), xed(x, xd)))

        surface = lambda xd, xe: np.array([
            [xe[0]], 
            [xe[2] + (lam[1]/2)*np.arctan(xe[1]) + (lam[2]/2)*np.abs(xe[2])*np.sign(xe[1])]
        ])  


        x_sys = np.zeros((n, num_points))
        angle = np.zeros((1, num_points))
        u_sys = np.zeros((m, num_points))

        s = np.zeros((m, num_points))
        e = np.zeros((n, num_points)) # error
        v_c = np.zeros((1, num_points))
        omega_c = np.zeros((1, num_points))
        x_ref=np.zeros((9, num_points))

        t_k = np.zeros(num_points)

        ## Setup Simulation
        dv_r = 0
        domega_c = 0

        
        t0 = rospy.Time.now().to_sec()
        while t0 == 0:
            t0 = rospy.Time.now().to_sec()
            print("Initializing Clock")
            rate = rospy.Rate(1/(1))
            rate.sleep()

        ## Simulation
        for k in range(num_points-1):
        ## Primary Controller
            t1=rospy.Time.now().to_sec()
            t_k[k+1] = t1 - t0
            # Get the reference trajectory
            e[:,k+1] = xe(x_sys[:,k], x_ref[:,k]).T
            s[:,k+1] = surface(x_ref[:,k], e[:,k+1]).T

            v_d = np.sqrt(x_ref[3,k]**2 + x_ref[4,k]**2)

            v_c[:,k+1] = Q[0]*s[0,k+1] + P[0]*bl_sat(s[0,k+1], phi[0]) + e[1,k+1]*omega_c[:,k+1] + v_d*cos(e[2,k+1])
            v_c[:,k+1] = sat(max_vel, v_c[:,k+1])
            
            omega_c[:,k+1] = (Q[1]*s[1,k+1] + P[1]*bl_sat(s[1,k+1], phi[1]) + x_ref[5,k]*(1+(lam[2]/2)*np.sign(e[2,k+1])) +
                (lam[1]/2)*v_d*np.sin(e[2,k+1])/(1+(e[1,k+1])**2))/ (1+(lam[2]/2)**np.sign(e[2,k+1])+(lam[1]/2)*e[0,k+1]/(1+(e[1,k+1])**2))
            omega_c[:,k+1] = sat(max_turn, omega_c[:,k+1])

            
    
            u_sys[:,k+1] = np.hstack((v_c[:,k+1], omega_c[:,k+1]))

            x_ref[:,k+1] = np.hstack((
                    waypoints_x[k], # x_ref for circle traj
                    waypoints_y[k], # y_ref for circle traj
                    theta[k], # theta_ref for circle traj
                    vel_x[k],
                    vel_y[k],
                    dtheta[k],
                    0,
                    0,
                    0
            ))
            x_sys[:,k+1] = states().flatten()
            angle[:,k+1] = np.copy(x_sys[2,k+1])
            x_sys[2,k+1] = unwrapAngle(x_sys[2,k+1], x_sys[2,k], angle[:,k], 2*np.pi)
            dv_r = (np.sqrt(x_sys[3,k]**2 + x_sys[4,k]**2)-np.sqrt(x_sys[3,k+1]**2 + x_sys[4,k+1]**2))*dt

            odom_pub(x_ref[:,k+1], reference_publisher)

            
            controller_input(u_sys[:,k], velocity_publisher)
            t2 = rospy.Time.now().to_sec()
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
df_x_sys = pd.DataFrame({'x':x_sys[0],'y':x_sys[1], 'theta':x_sys[2],'dx':x_sys[3], 'dy':x_sys[4],'dtheta':x_sys[5]})
df_xd = pd.DataFrame({'xd':x_ref[0],'yd':x_ref[1], 'thetad':x_ref[2],'dxd':x_ref[3], 'dyd':x_ref[4],'dthetad':x_ref[5]})
df_e = pd.DataFrame({'xer':e[0], 'yer':e[1], 'thetaer':e[2],'dxer':e[3], 'dyer':e[4],'dthetaer':e[5]})
df_s = pd.DataFrame({'s1':s[0], 's2':s[1]})
df_u = pd.DataFrame({'ux':u_sys[0],'uw':u_sys[1]})
df1 = pd.concat([df_t, df_x_sys, df_xd, df_s, df_u], axis=1)
df1.to_csv('SMC_mod3_field'+ str(trial_number) + '.csv', index=False)
# %%
