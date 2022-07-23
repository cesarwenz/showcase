#%%
#!/usr/bin/env python
from time import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rospy
import pandas as pd
import sys

trial_number = sys.argv[1]

def controller_input(u, pub):
    vel_msg = Twist()
    vel_msg.linear.x = u[0]
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
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


def wrapAngle(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def find_args(N,xcon,ucon):
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
    lbx = ca.vertcat(x_lbx, u_lbx)
    ubx = ca.vertcat(x_ubx, u_ubx)

    args={
        'lbx': lbx,
        'ubx': ubx,
        'lbg': lbg,
        'ubg': ubg,
    }
    return args

def skid_create_ocp_prim(n,m,N,dt,params):
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
    U = ca.MX.sym('U',m_tot,N); # inputs
    P = ca.MX.sym('P',n_tot+N*n) # parameter vector
    X = ca.MX.sym('X',n_tot,(N+1))
    ## Project the Problem Offline
    l = 0 # objective function
    st_ic = X[:,0] # initialize states
    g = st_ic - P[0:n_tot] # i.c. constraints (equal zero)

    for k in range(N):
        st = X[:,k]
        con = U[:,k]
        P_d = P[n*(k+1):n+(k+1)*n]
        xerr = X[0:n,k] - P_d # determine the error
        uerr= U [0:m,k]
        l = l + ca.mtimes(ca.mtimes(uerr.T,R),uerr)+ca.mtimes(ca.mtimes(xerr.T,Q),xerr) # stage cost
        st_next_actual = X[:,k+1]
        st_next_predict = ca.vertcat(
            st[:3] + f_sys(st, con)[3:]*dt, 
            f_sys(st, con)[3:]
        )
        g = ca.vertcat(g, st_next_actual-st_next_predict)

    # Terminal Constraints (for stability purposes)
    tferr = X[n-1,N] - P_d
    Vf = ca.mtimes(ca.mtimes(tferr.T,Q),tferr) # terminal cost
    Jn = l + Vf # cost function

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

def skid_create_ocp_aux(n,m,N,dt,params):

    R_S = params['R_S']
    R_aux = params['R_aux']
    # States
    x = ca.SX.sym('x', n)
    state = x
    
    # Controls
    u = ca.SX.sym('u',m) 
    control = u # real input

    n_tot = n 
    m_tot = m

    xe_temp = lambda x, xd: np.array([
        np.cos(x[2])*(xd[0]-x[0]) + np.sin(x[2])*(xd[1]-x[1]), 
        -np.sin(x[2])*(xd[0]-x[0]) + np.cos(x[2])*(xd[1]-x[1]), 
        xd[2]-x[2]
    ])

    xed = lambda x, xd: np.array([
        np.sqrt(ca.fmax(1e-4, (xd[3]**2 + xd[4]**2)))*np.cos(xe_temp(x,xd)[2]) + x[5]*xe_temp(x,xd)[1] - np.sqrt((x[3])**2 + (x[4])**2),
        np.sqrt(ca.fmax(1e-4, (xd[3]**2 + xd[4]**2)))*np.sin(xe_temp(x,xd)[2]) - x[5]*xe_temp(x,xd)[0],
        xd[5]-x[5]
    ])

    xe = lambda x, xd: np.hstack([xe_temp(x, xd), xed(x, xd)])

    ss_sys = lambda x, xd: np.array([
        xe(x, xd)[0], 
        xe(x, xd)[2] + (lam[1]/2) * np.arctan(xe(x, xd)[1]) + (lam[2]/2)*ca.fabs(xe(x, xd)[2])*np.sign(xe(x, xd)[1])
    ])

    # Disturbance
    d = ca.SX.sym('d') 
  
    # Robot's kinematics
    rhs_sys = ca.vertcat(
        x[0],
        x[1],
        x[2],
        u[0]*ca.cos(x[2]),
        u[0]*ca.sin(x[2]),
        u[1]
    )

    f_sys = ca.Function('f_sys',[state, control], [rhs_sys])

    # Decision Variables
    U = ca.SX.sym('U',m_tot,N); # inputs
    P = ca.SX.sym('P',n_tot+N*(n_tot+m)) # parameter vector
    X = ca.SX.sym('X',n_tot,(N+1))

    ## Project the Problem Offline
    l = 0 # objective function
    st_ic = X[:,0] # initialize states
    g = st_ic-P[0:n_tot] # i.c. constraints (equal zero)
    
    for k in range(N):
        st=X[:,k]
        con = U[:,k]
        P_d = P[(n_tot+m)*(k+1)-m:n_tot+(n_tot+m)*(k+1)] # desired trajectory
        ss_err = ss_sys(st, P_d[:n_tot])[np.newaxis].T # determine the error between aux state and nom state
        uerr= U [0:m,k] - P_d[n_tot:]# determine the error between aux control and nom control

        l = l + ca.mtimes(ca.mtimes(uerr.T,R_aux),uerr)+ca.mtimes(ca.mtimes(ss_err.T,R_S),ss_err) # stage cost
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
        rospy.init_node('MPSMC', anonymous=True)
        
        velocity_publisher = rospy.Publisher('/icerobot_velocity_controller/cmd_vel', Twist, queue_size=10)
        reference_publisher = rospy.Publisher('/reference_trajectory', Odometry, queue_size=10)
        nominal_publisher = rospy.Publisher('/nominal_state', Odometry, queue_size=10)
        time_threshold = 0.8
        #  Constants
        n=6 # States
        m=2 # Controls
        dt = 0.2

        Q_nom = np.diag([1, 1, 1.3, 0.1, 0.1, 0.1])
        R_nom = 1*np.diag([1, 1]) 

        # Auxiliary MPC (For MPSMC) (Gains)
        R_S = 1e0*np.diag([1, 1]) # for the mpsmc surface, n_dy size
        R_aux = 1e0*np.diag([1, 1])  # for the mpsmc input, m size

        lam = 1*np.array([1, 0.4, 0.45]) 
        N = 10

        x_ic = states().flatten()
        xd_ic = states().flatten()

        num_points = np.shape(waypoints_x)[0] 
        t_f = num_points*dt
        tf_N = (num_points+N)*dt
        t = np.arange(0, t_f, dt) 
        t_N = np.arange(0, tf_N, dt)
        time_vec = {
                't': t,
                't_N': t_N,
        }

        # Add zeros to end of path for horizon
        waypoints_x = np.append(waypoints_x, waypoints_x[-1]*np.ones(N))
        waypoints_y = np.append(waypoints_y, waypoints_y[-1]*np.ones(N))
        theta = np.append(theta, theta[-1]*np.ones(N))
        vel_x = np.append(vel_x, np.zeros(N))
        vel_y = np.append(vel_y, np.zeros(N))
        dtheta = np.append(dtheta, np.zeros(N))

        #  Uncertainty (OPTIONAL)
        #  Offers parameter uncertainty based on the fraction px.
        shuff = lambda a,b: a+ (b-a)*np.random.randn(1)
        bounds = lambda x, frac: x*(1+frac)
        px = 0 # between 0 and 1

        # Maximum Disturbance Bound
        D = 0 # upper bound of expected disturbance reps 10N

        xe_temp = lambda x, xd: np.array([
            np.cos(x[2])*(xd[0]-x[0]) + np.sin(x[2])*(xd[1]-x[1]), 
            -np.sin(x[2])*(xd[0]-x[0]) + np.cos(x[2])*(xd[1]-x[1]), 
            xd[2]-x[2]
        ])

        xed = lambda x, xd: np.array([
            np.sqrt(ca.fmax(1e-4, (xd[3]**2 + xd[4]**2)))*np.cos(xe_temp(x,xd)[2]) + x[5]*xe_temp(x,xd)[1] - np.sqrt((x[3])**2 + (x[4])**2),
            np.sqrt(ca.fmax(1e-4, (xd[3]**2 + xd[4]**2)))*np.sin(xe_temp(x,xd)[2]) - x[5]*xe_temp(x,xd)[0],
            xd[5]-x[5] 
        ])

        xe = lambda x, xd: np.hstack([xe_temp(x, xd), xed(x, xd)])

        ss_sys = lambda x, xd: np.array([
            xe(x, xd)[0], 
            xe(x, xd)[2] + (lam[1]/2) * np.arctan(xe(x, xd)[1]) + (lam[2]/2)*np.abs(xe(x, xd)[2])*np.sign(xe(x, xd)[1])
        ])
        ## Constraint for changing
        ####

        # Input Constraint
        u_ub = np.array([0.4, 0.3]) 
        u_lb = -u_ub # tau
        # State Constraint
        x_ub = np.inf*np.ones((n,1)) 
        x_lb = -x_ub # position and velocity constraints
        # x,y,theta,vx,omega

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
        umax[:,0] = np.copy(u_ub)
        umin[:,0] = np.copy(u_lb)

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
            'lam' : lam,
            'Q_nom': Q_nom,
            'R_nom': R_nom,
            'R_aux': R_aux,
            'R_S': R_S
        }
        ####
        ## Constraint Arguments
        ####


        # Inequality Constraints (Linear Ax<b)
        # For NMPC (Primary)
        s_con_mpsmc = ca.horzcat(xmin[:,0], xmax[:,0])
        u_con_mpsmc = ca.horzcat(umin[:,0], umax[:,0])

        # For NMPC (Auxiliary)
        s_con_mpc = ca.horzcat(xmin_nmpc, xmax_nmpc)
        u_con_mpc = ca.horzcat(umin[:,0], umax[:,0])

        ## Data Collection Setup
        args = {
            'args_mpc': find_args(N,s_con_mpc,u_con_mpc),
            'args_mpsmc': find_args(N,s_con_mpsmc,u_con_mpsmc)
        }

        # Initialize System
        # provide initial conditions here for stationary, or in loop for
        # randomized.

        ic={
            'x_ic': x_ic,
            'xd_ic': xd_ic
        }

        # Create disturbance
        d = shuff(-D,D)*np.ones((m,np.shape(t)[0]))*0
        params['dist'] = d

        # Set up the Constraints
        xmax = con['xmax']
        xmin = con['xmin'] # state constraints: [s, x, phi, S0]
        umax = con['umax']
        umin = con['umin'] # control constraints: [u]
        xmax_pos = con['xmax_nmpc']


        time_fun = time_vec['t'] # grab simulation time
        t_N = time_vec['t_N'] # need this time if you are doing a trajectory to project


        # real number of states x, y, theta, vx, omega
        n_tot=np.shape(xmax)[0] # total number of "states" based on xmax_mpsmc
        m_tot=np.shape(umax)[0] # total number of "inputs" based on umax_mpsmc

        # Initial Conditions
        s_ic=np.zeros((m,1)) # sliding surface

        ## Desired State Vectors
        # put the traj info here needs to be qref, c_r, w_r

        ## True System Vectors
        x_sys = np.zeros((n, np.size(time_fun)))
        angle = np.zeros(np.size(time_fun))
        x_sys[:,0] = np.copy(x_ic) # includes kin and dy

        s_sys = np.zeros((m, np.size(time_fun)))
        s_sys[:,0] = s_ic.T # sliding variable

        u_tot = np.zeros((m, np.size(time_fun))) # control input
        xd = np.zeros((n, np.size(t_N)))
        xd[:,0] = np.copy(xd_ic)

        ## MPC System Vectors
        # These state and input vectors are denoted as "nominal" meaning they
        # belong to a version of the dynamics uncorrupted by noise.
        # State
        z = np.zeros((n, np.size(time_fun)))
        z[:,0] = np.copy(x_ic) # nominal state

        # Control
        v = np.zeros((m, np.size(time_fun))) # nominal input

        # Initialization Vectors, just necessary for MPC
        Z0 = ca.repmat(z[:,0],1,N+1).T
        v0 = np.zeros((N,m))

        zzl=[] #open loop trajectory (MPC predictions)
        v_cl=[] # open loop input

        ## MPSMC System Vectors
        # These state and input vectors belong to a version of the dynamics 
        # also uncorrupted by noise, but whose input will be sent to the real
        # system.
        # States

        x = np.zeros((n, np.size(time_fun)))
        x[:,0] = np.copy(x_ic) # MPSMC state

        e = np.zeros((n, np.size(time_fun)))
        xe_rel = np.zeros((n, np.size(time_fun)))

        # value of the tube as the difference between MPSMC x and MPC z.
        # Control
        u0 = np.zeros((N,m_tot))
        u = np.zeros((m, np.size(time_fun))) # control input MPSMC
        ss = np.zeros((m, np.size(time_fun)))
        t_k = np.zeros(np.size(time_fun))
        # Initialization Vectors
        X0 = ca.repmat(x[:,0],1,N+1).T
        xxl=[] # open loop trajectory (MPSMC predictions)
        u_cl=[] # open loop input

        ## Create Optimal Control Problem
        # Grabs the offline solution from the CasADi solver based on the given
        # system values.

        [mpc,f_mpc] = skid_create_ocp_prim(n,m,N,dt,params)

        [mpsmc,f_mpsmc]=skid_create_ocp_aux(n,m,N,dt,params)


        # Import Arguments
        args_mpc = args['args_mpc']
        args_mpsmc = args['args_mpsmc']

        # Initialize p vector which contains desired trajectory
        args_mpc['p'] = ca.DM.zeros(1, n+N*n)
        args_mpsmc['p'] = ca.DM.zeros(1, n_tot+N*(n_tot+m))

        t0 = rospy.Time.now().to_sec()
        
        ## Simulation
        for k in range(np.shape(time_fun)[0]-1):
            ## Primary Controller
            t1=rospy.Time.now().to_sec()
            t_k[k+1] = t1 - t0
            args_mpc['p'][0:n] = z[:,k] # provide initial states
            for j in range(N):
                xd[:,k+j] = ca.horzcat(
                    waypoints_x[k+j], # x_ref for circle traj
                    waypoints_y[k+j], # y_ref for circle traj
                    theta[k+j], # theta_ref for circle traj
                    vel_x[k+j],
                    vel_y[k+j],
                    dtheta[k+j]
                )
                args_mpc['p'][n*(j+1):n+(j+1)*n] = xd[:,k+j]
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
            z[3:,k+1] = ca.DM.full(f_mpc(z[:,k], v[:,k])[3:]).T
            z[:3,k+1] = ca.DM.full(z[:3,k] + (dt * f_mpc(z[:,k], v[:,k])[3:])).T
            odom_pub(z[:,k+1], nominal_publisher)
            # Initialize Next Optimization Variables
            Z0 = ca.reshape(sol_mpc['x'][0:n*(N+1)], n, N+1)
            Z0 = ca.horzcat(
                Z0[:, 1:],
                ca.reshape(Z0[:, -1], -1, 1)
            ).T
            v0 = ca.vertcat(
                v_temp[1:np.shape(v_temp)[0],:],
                v_temp[np.shape(v_temp)[0]-1,:]
            )

            ## Auxiliary Controller


            args_mpsmc['p'][0:n_tot] = np.copy(x_sys[:,k])

            for j in range(N):
                args_mpsmc['p'][(n_tot+m)*(j+1)-(m):n_tot+(j+1)*(n_tot+m)] = ca.horzcat(
                    zzl[j,:],
                    v_temp[j,:]
                )
            # Because using MS, give initial guesses for X and U
            args_mpsmc['x0'] = ca.vertcat(
                ca.reshape(X0.T, n_tot*(N+1),1),
                ca.reshape(u0.T, m_tot*N,1)
                )

            # Solve the Problem
            sol_mpsmc=mpsmc(
                x0=args_mpsmc['x0'],
                lbx=args_mpsmc['lbx'],
                ubx=args_mpsmc['ubx'],
                lbg=args_mpsmc['lbg'],
                ubg=args_mpsmc['ubg'],
                p=args_mpsmc['p']
            )

            # Reshape states/inputs as matrices
            u_temp = ca.reshape(sol_mpsmc['x'][n_tot*(N+1):n_tot*(N+1)+m_tot*N], m_tot, N).T
            xxl = ca.reshape(sol_mpsmc['x'][0:n_tot*(N+1)], n_tot, N+1).T

            u_cl = ca.vertcat(u_cl, u_temp[0,:]) #store open loop trajectory
            u[:,k] =  np.array(u_temp[0,:].T).flatten() 

            # derivative = ca.vertcat(
            #     f_mpsmc(s[:,k], x[:,k], z[:,k], u[:,k])[:2],
            #     f_mpsmc(s[:,k], x[:,k], z[:,k], u[:,k])[-3:]
            # )
            x_sys[:,k+1] = states().flatten()
            angle[k+1] = np.copy(x_sys[2,k+1])
            x_sys[2,k+1] = unwrapAngle(x_sys[2,k+1], x_sys[2,k], angle[k], 2*np.pi)
            
            # Initialize Next Optimization Variables
            X0 = ca.reshape(sol_mpsmc['x'][0:n_tot*(N+1)], n_tot, N+1)
            X0 = ca.horzcat(  
                X0[:, 1:],
                ca.reshape(X0[:, -1], -1, 1)
            ).T

            u0 = ca.vertcat(
                u_temp[1:np.shape(u_temp)[0],:],
                u_temp[np.shape(u_temp)[0]-1,:]
            )
            # Determine the new sliding surface values    
            xe_rel[:,k+1] = xe(x_sys[:,k+1], xd[:,k+1])
            ss[:,k+1] = ss_sys(x_sys[:,k+1], xd[:,k+1])
            u_tot[:,k] = u[:,k] # Assign the REAL control Input to the one produced 

            controller_input(u_tot[:,k], velocity_publisher)
            ## System Propagation
            # propagate with one step euler discretization

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

df_t = pd.DataFrame({'time':time_fun})
df_x_sys = pd.DataFrame({'x':x_sys[0],'y':x_sys[1], 'theta':x_sys[2],'dx':x_sys[3], 'dy':x_sys[4],'dtheta':x_sys[5]})
df_xd = pd.DataFrame({'xd':xd[0],'yd':xd[1], 'thetad':xd[2],'dxd':xd[3], 'dyd':xd[4],'dthetad':xd[5]})
df_z = pd.DataFrame({'xz':z[0],'yz':z[1], 'thetaz':z[2],'dxz':z[3], 'dyz':z[4],'dthetaz':z[5]})
df_u = pd.DataFrame({'ux':u_tot[0],'uw':u_tot[1], 'vx':v[0],'vw':v[1]})
df_e = pd.DataFrame({'xer':xe_rel[0],'yer':xe_rel[1], 'thetaer':xe_rel[2],'dxer':xe_rel[3], 'dyer':xe_rel[4],'dthetaer':xe_rel[5]})
df_ss = pd.DataFrame({'vss':ss[0],'wss':ss[1]})
df1 = pd.concat([df_t, df_x_sys, df_xd, df_z, df_u, df_e, df_ss], axis=1)
df1.to_csv('kin_mpsmc_mod2_field'+ str(trial_number) + '.csv', index=False)

# %%
