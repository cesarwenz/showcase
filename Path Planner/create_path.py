#%%
# import path_planner_map_back
# reload(path_planner_map_back)
from path_planner_map_back import *
import pandas as pd
import numpy as np
import os
import cv2

# Path planner
# meter_per_pix = 0.05
# Sim orchard params
# map_origin_x = -10.4
# map_origin_y = -14.35
# start_node = np.array([0, 0])/meter_per_pix
# circ_radius = 2.5/meter_per_pix
# x_offset = 50
# y_offset = 30

# Real orchard params
meter_per_pix = 0.05
# map_origin_x = -4.70000
# map_origin_y = -6.20000
# w_r = (np.pi)/30
# map_file = 'real_orchard_1.pgm'

map_origin_x = -10.40
map_origin_y = -14.35

start_node = np.array([-map_origin_x, -map_origin_y])/meter_per_pix
start_node[0] = start_node[0] + map_origin_x
start_node[1] = start_node[1] + map_origin_y
circ_radius = 3/meter_per_pix
x_offset = 50
y_offset = 40

linear_vel = 0.5
c_r = circ_radius

w_r = (np.pi)/30 # to get integer in t_k of path planner (only change denom)
dt = 0.2



map_file = 'orchard.pgm'
map_clr = cv2.imread(map_file)[::-1]

# Define grey scale map
map_grey = cv2.cvtColor(map_clr, cv2.COLOR_BGR2GRAY)

# Define HSV map
hsv = cv2.cvtColor(map_clr, cv2.COLOR_BGR2HSV) 

# Filter properties
lower_black = np.array([0,0,0])
upper_black = np.array([0,0,200])

# Remove all background and only have tree points
mask = cv2.inRange(hsv, lower_black, upper_black)

# Filter scan outliers of trees
mask = cv2.fastNlMeansDenoising(mask,None,20,7,21)
# res = cv2.bitwise_and(map_clr,map_clr, mask= mask)

# Edge detection of each point
map_edge = cv2.Canny(mask, 100, 100, None, 3)
# Blur edges to make it more apperent
map_edge = cv2.blur(map_edge,(10,10))

# Detect edges of lines and bundle them together
lines = cv2.HoughLinesP(map_edge, 5, (np.pi/180), 500, minLineLength=250, maxLineGap=100)
line_merger = HoughBundler()
lines = line_merger.process_lines(lines, map_edge)
lines = np.reshape(lines, (len(lines), 4))

sa = RowPlanner(start_node, lines, linear_vel, meter_per_pix, circ_radius, c_r, w_r, dt, x_offset, y_offset, stopping_iter=5000)
sa.batch_anneal()

waypoints_x, waypoints_y, theta, vel_x, vel_y, dtheta = sa.trajectory(map_origin_x, map_origin_y)
#%%
theta = theta-2*np.pi

#%%

df_xd = pd.DataFrame({'xd':waypoints_x,'yd':waypoints_y, 'thetad':theta,'dxd':vel_x, 'dyd':vel_y,'dthetad':dtheta})
df_tree_nodes = pd.DataFrame({'tree_nodes':sa.lines.reshape((1,-1))[0]})
df1 = pd.concat([df_xd, df_tree_nodes], axis=1)
df1.to_csv('../Data/sim_test_v1.csv', index=False)
# %%
# %%
# Trajectory publisher
from nav_msgs.msg import Odometry
import numpy as np
import rospy
import matplotlib.pyplot as plt
import pandas as pd
import tf


def wrapAngle(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def reference_publisher(xd, topic):
    
    reference_publisher = rospy.Publisher(topic, Odometry)
    ref = Odometry()
    ref.header.stamp = rospy.Time.now()
    ref.header.frame_id = '/map'
    ref.child_frame_id = '/base_link'

    ref.pose.pose.position.x = xd[0]
    ref.pose.pose.position.y = xd[1]
    ref.pose.pose.position.z = 0
    angle = wrapAngle(xd[2])    
    odom_quat = tf.transformations.quaternion_from_euler(0, 0, angle)
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

    reference_publisher.publish(ref)

dt = 0.01

num_points = np.shape(waypoints_x)[0] 
rospy.init_node('traj_pub', anonymous=True)
t_k = np.zeros(num_points)
t0 = rospy.Time.now().to_sec()
for k in range(num_points-1):
    t1=rospy.Time.now().to_sec()
    t_k[k+1] = t1 - t0
    xd = np.array([
        waypoints_x[k], # x_ref for circle traj
        waypoints_y[k], # y_ref for circle traj
        theta[k], # theta_ref for circle traj
        vel_x[k],
        vel_y[k],
        dtheta[k]
    ])

    reference_publisher(xd, '/reference_trajectory')
    t2 = rospy.Time.now().to_sec()
    elapsed_time = t2 - t1
    
    rate = rospy.Rate(1/(dt-elapsed_time)) 
    rate.sleep()
    t2 = rospy.Time.now().to_sec()
    elapsed_time = t2 - t1

# %%
