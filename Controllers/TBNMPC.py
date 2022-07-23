#%%
#!/usr/bin/env python
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import pandas as pd
import sys

trial_number = sys.argv[1]

def find_args(N,xcon,ucon, n, m):
    #Develops MPC Args
    
    # lower bounds - c1
    # upper bounds - c2
    n_tot=xcon.size(1)
    # Multiple Shooting Equality Constraints
    lbg = ca.DM.ones(1,(n_tot)*(N+1))*-1e-20
    ubg = ca.DM.ones(1,(n_tot)*(N+1))*1e-20

    # State Constraints
    x_lbx = ca.repmat(xcon[:,0],N+1,1)
    x_ubx = ca.repmat(xcon[:,1],N+1,1)

    # Input Constraints
    u_lbx = ca.repmat(ucon[:,0],N,1)
    u_ubx = ca.repmat(ucon[:,1],N,1)
    # Stack!
    lbx=ca.vertcat(x_lbx, u_lbx)
    ubx=ca.vertcat(x_ubx, u_ubx)

    args={
        'lbx': lbx,
        'ubx': ubx,
        'lbg': lbg,
        'ubg': ubg,
    }
    return args

def create_ocp_nom(n,m,N,dt,params):
# Determines OCP for NMPC-NMPC for nominal control for WMR.
    # MPC Params
    Q = params['Q_nom']
    R = params['R_nom']

    # States
    x = ca.SX.sym('x', n)
    state = x

    u = ca.SX.sym('u',m) 
    control = u # real input


    # Acquire total number of states/inputs for optimization purposes
    n_tot = n
    m_tot = m

    # Acquire total number of states/inputs for optimization purposes
    n_tot = n
    m_tot = m

    # Robot's Kinematics
    rhs_sys = ca.vertcat(
        x[0],
        x[1],
        x[2],
        u[0]*ca.cos(x[2]),
        u[0]*ca.sin(x[2]),
        u[1]
    )


    # Create function of system's kinematics
    sys_dy = rhs_sys  
    f_sys = ca.Function('f_sys',[state, control], [sys_dy])

    U = ca.SX.sym('U',m_tot,N); # inputs
    P = ca.SX.sym('P',n_tot+N*n) # parameter vector
    X = ca.SX.sym('X',n_tot,(N+1))

    ## Project the Problem Offline
    l = 0 # objective function
    st_ic = X[:,0] # initialize states
    g = st_ic - P[0:n_tot] # i.c. constraints (equal zero)

    for k in range(N):
        st = X[:,k]
        con = U[:,k]
        P_s = P[n*(k+1):n*(k+1)+n] # desired state trajectory
        xerr = X[:,k] - P_s # determine the error
        uerr= U [:,k]
        l = l + ca.mtimes(ca.mtimes(uerr.T,R),uerr)+ca.mtimes(ca.mtimes(xerr.T,Q),xerr) # stage cost
        st_next_actual = X[:,k+1]
        st_next_predict = ca.vertcat(
            st[:3] + f_sys(st, con)[3:]*dt, 
            f_sys(st, con)[3:]
        )
        g = ca.vertcat(g, st_next_actual-st_next_predict)
    
    # Terminal Constraints (for stability purposes)
    tferr=X[n-1,N]-P_s
    Vf = ca.mtimes(ca.mtimes(tferr.T,Q),tferr) # terminal cost
    Jn = l + Vf; # cost function

    # Establish the Optimization
    # optimization variables: U if SS X and U if MS
    opt_variables = ca.vertcat(
        X.reshape((-1, 1)), 
        U.reshape((-1, 1))
    )
    

    # construct the problem
    nlp_prob = {
        'f': Jn,
        'x': opt_variables,
        'g': g,
        'p': P
    }

    # Set Up Options
    opts = {
        'ipopt': {
            'max_iter': 100,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
        'print_time': 0
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    return solver,f_sys


def create_ocp_aux(n,m,N,dt,params):
# Determines OCP for NMPC-NMPC for nominal control for spring mass damper.
    # MPC Params
    Q = params['Q_aux']
    R = params['R_aux']

    # States
    x = ca.SX.sym('x', n)
    state = x

    u = ca.SX.sym('u',m) 
    control = u # real input


    # Acquire total number of states/inputs for optimization purposes
    n_tot = n
    m_tot = m

    # Robot's Kinematics
    rhs_sys = ca.vertcat(
        x[0],
        x[1],
        x[2],
        u[0]*ca.cos(x[2]),
        u[0]*ca.sin(x[2]),
        u[1]
    )

    # Create function of system's kinematics
    sys_dy = rhs_sys  
    f_sys = ca.Function('f_sys',[state, control], [sys_dy])


    # Decision Variables
    U = ca.SX.sym('U',m_tot,N); # inputs
    P = ca.SX.sym('P',n_tot+N*n) # parameter vector
    X = ca.SX.sym('X',n_tot,(N+1))

    ## Project the Problem Offline
    l = 0 # objective function
    st_ic = X[:,0] # initialize states
    g = st_ic-P[0:n_tot] # i.c. constraints (equal zero)
    
    for k in range(N):
        st = X[:,k]
        con = U[:,k]
        P_s = P[n*(k+1):n*(k+1)+n]
        xerr = X[:,k] - P_s 
        uerr= U [:,k]
        l = l + ca.mtimes(ca.mtimes(uerr.T,R),uerr)+ca.mtimes(ca.mtimes(xerr.T,Q),xerr) # stage cost
        st_next_actual = X[:,k+1]
        st_next_predict = ca.vertcat(
            st[:3] + f_sys(st, con)[3:]*dt, 
            f_sys(st, con)[3:]
        )
        g = ca.vertcat(g, st_next_actual-st_next_predict)

    Jn = l 

    # Establish the Optimization
    # optimization variables: U if SS X and U if MS
    opt_variables = ca.vertcat(
        X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
        U.reshape((-1, 1))
    )


    # construct the problem
    nlp_prob = {
        'f': Jn,
        'x': opt_variables,
        'g': g,
        'p': P
    }

    # Set Up Options
    opts = {
        'ipopt': {
            'max_iter': 100,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
        'print_time': 0
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    return solver,f_sys

def find_args(N,xcon,ucon, n, m):
    #Develops MPC Args
    
    # lower bounds - c1
    # upper bounds - c2
    n_tot=xcon.size(1)
    # Multiple Shooting Equality Constraints
    lbg = ca.DM.ones(1,(n_tot)*(N+1))*-1e-20
    ubg = ca.DM.ones(1,(n_tot)*(N+1))*1e-20

    # State Constraints
    x_lbx = ca.repmat(xcon[:,0],N+1,1)
    x_ubx = ca.repmat(xcon[:,1],N+1,1)

    # Input Constraints
    u_lbx = ca.repmat(ucon[:,0],N,1)
    u_ubx = ca.repmat(ucon[:,1],N,1)

    # Stack!
    lbx=ca.vertcat(x_lbx, u_lbx)
    ubx=ca.vertcat(x_ubx, u_ubx)

    args={
        'lbx': lbx,
        'ubx': ubx,
        'lbg': lbg,
        'ubg': ubg,
    }
    return args

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

df = pd.read_csv('../Data/field_test_v4.csv')
waypoints_x = df['xd'].to_numpy()
waypoints_y = df['yd'].to_numpy()
theta = df['thetad'].to_numpy()
vel_x = df['dxd'].to_numpy()
vel_y = df['dyd'].to_numpy()
dtheta = df['dthetad'].to_numpy()
num_points = np.shape(waypoints_x)[0] 

if __name__ == '__main__':
    try:
        rospy.init_node('TUBE', anonymous=True)
        velocity_publisher = rospy.Publisher('/icerobot_velocity_controller/cmd_vel', Twist, queue_size=10)
        reference_publisher = rospy.Publisher('/reference_trajectory', Odometry, queue_size=10)
        nominal_publisher = rospy.Publisher('/nominal_state', Odometry, queue_size=10)
        #  Constants
        time_threshold = 0.8

        n=6 # States
        m=2 # Controls

        N = 10
        # MPC Weight Matrices (Gains)
        Q_nom = np.diag([1, 1, 1.3, 0.1, 0.1, 0.1])
        R_nom = 0.5*np.diag([1, 1]) 

        Q_aux = np.diag([1, 1, 1.3, 0.1, 0.1, 0.1])
        R_aux = 1*np.diag([1, 1]) 

        dt = 0.2
        t_f = num_points*dt
        tf_N = (num_points+N)*dt
        t = np.arange(0, t_f, dt) 
        t_N = np.arange(0, tf_N, dt)
        time_vec = {
                't': t,
                't_N': t_N,
        }

        # Constraint Tighten for NMPC-NMPC
        # this parameter can be tuned based on repeated results
        beta=0.8
        
        u_ub = np.array([0.5, 0.3])
        u_lb = -u_ub
        x_ub = np.inf*np.ones((n,1)) 
        x_lb = -x_ub

        # X Constraints
        # For NMPC or Primary Controller
        xmax_nmpc = x_ub
        xmin_nmpc = x_lb
        # For MPSMC
        xmax = np.zeros((n,np.size(t_N))) 
        xmin = np.zeros((n,np.size(t_N))) 
        umax = np.zeros((m,np.size(t_N))) 
        umin = np.zeros((m,np.size(t_N))) 

        xmax[:,0] = x_ub.T
        xmin[:,0] = x_lb.T

        # U Constraints
        umax[:,0] = np.copy(u_ub.T)
        umin[:,0] = np.copy(u_lb.T)
        
        # Store the constraints in a structure.
        con =  {
            'xmax': xmax,
            'xmin': xmin,
            'umax': umax,
            'umin': umin,
            'xmax_nmpc' : xmax_nmpc
        }
        
        # Create Structure with Constants to pass to subfunctions.
        params={
            'Q_nom': Q_nom,
            'R_nom': R_nom,
            'Q_aux': Q_aux,
            'R_aux': R_aux,
            'beta': beta
        } 
        ####
        ## Constraint Arguments
        ####
        
        s_con_nmpc = ca.horzcat(xmin[:,0], xmax[:,0])
        u_con_nmpc = ca.horzcat(umin[:,0], umax[:,0])

        # For NMPC (Auxiliary)
        s_con_mpc = ca.horzcat(xmin_nmpc, xmax_nmpc)
        u_con_mpc = ca.horzcat(umin[:,0], umax[:,0])

        
        ## Data Collection Setup
        args = {
            'args_mpc': find_args(N,s_con_mpc,u_con_mpc, n, m),
            'args_nmpc': find_args(N,s_con_nmpc,u_con_nmpc, n, m)
        }
        x_ic = states().flatten()

        ic={
            'x_ic': x_ic,
        }

        # Initialization Vectors
        xd = np.zeros((n, num_points+N))
        # True System Vectors
        x_sys = np.zeros((n, num_points))
        angle = np.zeros((1, num_points))
        x_sys[:,0] = x_ic
        u_tot = np.zeros((m, num_points))

        ## NMPC (Primary) State Vectors
        # Nominal State
        z = np.zeros((n, num_points))
        z[:,0] = x_ic

        # Nominal Control
        v = np.zeros((m, num_points))

        # Initialization Vectors
        Z0 = ca.repmat(x_ic,1,N+1).T
        v0 = np.zeros((N, m))

        zzl = [] # open loop trajectory (MPC predictions)
        v_cl = [] # open loop input

        ## NMPC (Auxiliary) State Vectors
        # Auxiliary State
        x = np.zeros((n, num_points)) 
        x[:,0] = x_ic

        # Auxiliary Control
        u = np.zeros((m, num_points))

        t_k = np.zeros(num_points)
        # Initialization Vectors
        X0 = ca.repmat(x_ic,1,N+1).T
        u0 = np.zeros((N, m))

        xxl = [] # open loop trajectory (MPC predictions)
        u_cl = [] # open loop input
        
        ## Create Optimal Control Problem
        [mpc,f_nom] = create_ocp_nom(n,m,N,dt,params)
        [mayne,f_aux] = create_ocp_aux(n,m,N,dt,params)

        # Arguments
        args_mpc = args['args_mpc']
        args_nmpc = args['args_nmpc']

        # Augment the Nominal Constraints with U tightening
        args_mpc['lbx'][n*(N+1):] = beta*args_mpc['lbx'][n*(N+1):]
        args_mpc['ubx'][n*(N+1):] = beta*args_mpc['ubx'][n*(N+1):]

        # Initialize p vector which contains desired trajectory
        args_mpc['p'] = ca.DM.zeros(1, (n + N*n))
        args_nmpc['p'] = ca.DM.zeros(1, (n + N*n))

        t0 = rospy.Time.now().to_sec()
        waypoints_x = np.append(waypoints_x, waypoints_x[-1]*np.ones(N))
        waypoints_y = np.append(waypoints_y, waypoints_y[-1]*np.ones(N))
        theta = np.append(theta, theta[-1]*np.ones(N))
        vel_x = np.append(vel_x, np.zeros(N))
        vel_y = np.append(vel_y, np.zeros(N))
        dtheta = np.append(dtheta, np.zeros(N))
        ## Simulation

        for k in range(num_points-1):
        ## Primary Controller
            t1=rospy.Time.now().to_sec()
            t_k[k+1] = t1 - t0
            args_mpc['p'][0:n] = z[:,k] # provide initial states
            for r in range(N):
                x_ref = waypoints_x[k+r]# x_ref for circle traj
                y_ref = waypoints_y[k+r] # y_ref for circle traj
                theta_ref = theta[k+r] # theta_ref for circle traj
                dx_ref = vel_x[k+r]# x_ref for circle traj
                dy_ref = vel_y[k+r] # y_ref for circle traj
                dtheta_ref = dtheta[k+r] # theta_ref for circle traj
                xd[:,k+r] = np.vstack((x_ref, y_ref, theta_ref,  dx_ref, dy_ref, dtheta_ref)).flatten() # store ref trajectories
                args_mpc['p'][n*(r+1):n*(r+1)+n] = xd[:,k+r]  # assign ref to p
            odom_pub(xd[:,k], reference_publisher)
            # Because using MS, give initial guesses for X and U
            args_mpc['x0'] = ca.vertcat(
                ca.reshape(Z0.T, n*(N+1), 1),
                ca.reshape(v0.T, m*N, 1)
            )

            # Solve the Problem
            sol_mpc=mpc(
                x0 = args_mpc['x0'],
                lbx = args_mpc['lbx'],
                ubx = args_mpc['ubx'],
                lbg = args_mpc['lbg'],
                ubg = args_mpc['ubg'],
                p = args_mpc['p']
            )

            # Reshape states/inputs as matrices
            v_temp = ca.reshape(sol_mpc['x'][n*(N+1):n*(N+1)+m*N], m, N).T
            zzl = ca.reshape(sol_mpc['x'][0:n*(N+1)], n, N+1).T

            v_cl = ca.vertcat(v_cl, v_temp[0,:]) #store open loop trajectory
            v[:,k] = np.array(v_temp[0,:].T).flatten() #nominal control

            # Propagate the Nominal Solution
            z[3:,k+1] = ca.DM.full(f_nom(z[:,k], v[:,k])[3:]).T
            z[:3,k+1] = ca.DM.full(z[:3,k] + (dt * f_nom(z[:,k], v[:,k])[3:])).T
            odom_pub(z[:,k+1], nominal_publisher)
            # Initialize Next Optimization Variables
            ##### Add initial conditions
            #####
            Z0 = ca.reshape(sol_mpc['x'][0:n*(N+1)], n, N+1)
            Z0 = ca.horzcat(
                Z0[:, 1:],
                ca.reshape(Z0[:, -1], -1, 1)
            ).T
            v0 = ca.vertcat(
                v_temp[1:np.shape(v_temp)[0],:],
                v_temp[np.shape(v_temp)[0]-1,:]
            )

            ##### Add initial conditions
            #####
            ## Auxiliary Controller
            args_nmpc['p'][0:n] = x_sys[:,k] # initialize controller with true system values

            for j in range(N):
                args_nmpc['p'][n*(j+1):n+(j+1)*(n)] = ca.horzcat(
                    zzl[j,:]
                )
            # Because using MS, give initial guesses for X and U
            args_nmpc['x0'] = ca.vertcat(
                ca.reshape(X0.T, n*(N+1),1),
                ca.reshape(u0.T, m*N,1)
                )
            
            # Solve the Problem
            sol_mayne=mayne(
                x0=args_nmpc['x0'],
                lbx=args_nmpc['lbx'],
                ubx=args_nmpc['ubx'],
                lbg=args_nmpc['lbg'],
                ubg=args_nmpc['ubg'],
                p=args_nmpc['p']
            )

            # Reshape states/inputs as matrices
            u_temp = ca.reshape(sol_mayne['x'][n*(N+1):n*(N+1)+m*N], m, N).T
            xxl = ca.reshape(sol_mayne['x'][0:n*(N+1)], n, N+1).T

            u_cl = ca.vertcat(u_cl, u_temp[0,:]) #store open loop trajectory
            u[:,k] =  np.array(u_temp[0,:].T).flatten() 
            

            # Initialize Next Optimization Variables
            X0 = ca.reshape(sol_mayne['x'][0:n*(N+1)], n, N+1)

            X0 = ca.horzcat(
                X0[:, 1:],
                ca.reshape(X0[:, -1], -1, 1)
            ).T

            u0 = ca.vertcat(
                u_temp[1:np.shape(u_temp)[0],:],
                u_temp[np.shape(u_temp)[0]-1,:]
            )
            controller_input(u[:,k], velocity_publisher)
            ## System Propagation
            # propagate with one step euler discretization
            u_tot[:,k] = u[:,k] # control is from optimal auxiliary controller
            x_sys[:,k+1] = states().flatten()
            angle[:,k+1] = np.copy(x_sys[2,k+1])
            x_sys[2,k+1] = unwrapAngle(x_sys[2,k+1], x_sys[2,k], angle[:,k], 2*np.pi)

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
df_x_sys = pd.DataFrame({'x':x_sys[0],'y':x_sys[1], 'theta':x_sys[2], 'dx':x_sys[3],'dy':x_sys[4], 'dtheta':x_sys[5]})
df_z = pd.DataFrame({'xz':z[0],'yz':z[1], 'thetaz':z[2], 'dxz':z[3],'dyz':z[4], 'dthetaz':z[5]})
df_u = pd.DataFrame({'ux':u_tot[0],'uw':u_tot[1], 'vx':v[0],'vw':v[1]})
df1 = pd.concat([df_t, df_x_sys, df_z, df_u], axis=1)
df1.to_csv('tube_mpc_field'+ str(trial_number) + '.csv', index=False)
# %%
