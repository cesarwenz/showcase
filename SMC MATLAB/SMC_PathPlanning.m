%====================================================================
% Author: Cesar Wen Zhu
% Date: April 13, 2021
%====================================================================
% Sliding mode control of SSMR using inter-row coverage path planner
% Note: Need around 700s to run program

clear all; close all;
global m g r mu_x mu_y N lam eta Izz c x_0 pos_x ...
    pos_y theta vel_x vel_y vel_theta acc_x acc_y acc_theta time_res

% SSMR physical parameters
a = 0.3005;
b = 0.3155;
c = 0.3385;
m = 80.6030;
g = 9.81;
r = 0.3327;
mu_x = 0.09;
mu_y = 0.35;
Izz = 18394117294.20/1e9;
x_0 = 0.1; % Must be between -a to b

% Control parameters
N = [0.1; 0.1];
lam = 5;
eta1= 5;
eta2= 5;
eta3= 1;
eta = [eta1; eta2; eta3];

% Path planner
ox= [0.0, 10.0, 10.0, 0.0, 0.0];
oy = [0.0, 0.0, 10.0, 10.0, 0.0];
resolution = 0.1;
row_width = 3;
offset = 1;
radius = 4;
speed = 2;
speed_time = resolution/speed;
turning_speed = 1;
turning_speed_time = resolution/turning_speed;
path = RowPlanning(ox, oy, resolution,...
                row_width, offset, radius);

% Trajectory construction
pos_x = path.rx;
pos_y = path.ry;
theta = path.theta;
vel_x = [0; diff(pos_x)/speed_time];
vel_y = [0; diff(pos_y)/speed_time];
vel_theta = [0; diff(theta)/turning_speed_time];
acc_x = zeros(length(pos_y),1);
acc_y = zeros(length(pos_y),1);
acc_theta = zeros(length(theta),1);

% Simulation using ode45
x0=[pos_x(1); pos_y(1); theta(1); 0; 0; 0]; 
time_res = speed_time;
tspan = [0:time_res:((length(pos_x)-1)*time_res)];
tic
[t,y] = ode45(@(t,x) skid(t,x), tspan, x0);
toc

figure(1);
subplot(3,1,1);
plot(t,pos_x,'k')
hold on
plot(t,y(:,1),'b')
hold on
plot(t,(y(:,1)-pos_x),'r')
legend({'x', 'x_d','error'},'FontSize',10)
xlabel('Time (s)')
ylabel('X position (m)')


subplot(3,1,2);
plot(t,pos_y,'k')
hold on
plot(t,y(:,2),'b')
hold on
plot(t,(y(:,2)-pos_y),'r')
legend({'y', 'y_d','error'},'FontSize',10)
xlabel('Time (s)')
ylabel('Y position (m)')

subplot(3,1,3);
plot(t,theta,'k')
hold on
plot(t,y(:,3),'b')
hold on
plot(t,(y(:,3)-theta),'r')
legend({'\theta', '\theta_d', 'error'},'FontSize',10)
xlabel('Time (s)')
ylabel('Angular position (rad)')

figure(2);
plot(y(:,1),y(:,2),'r')
hold on
plot(pos_x, pos_y,'b')
xlabel('x (m)')
ylabel('y (m)')
legend({'SSMR', 'Trajectory'},'FontSize',10)

% RMSE of position
rx = sqrt(sum((y(:,1)-pos_x).^2/length(y(:,1))))
ry = sqrt(sum((y(:,2)-pos_y).^2/length(y(:,2))))
rtheta = sqrt(sum((y(:,3)-theta).^2/length(y(:,3))))

function x_dot = skid(t,x)
    global m r N lam eta Izz c x_0 time_res pos_x ...
    pos_y theta vel_x vel_y vel_theta acc_x acc_y acc_theta
    x_dot = zeros(6,1);
    
    % SSMR Model
    M = [m 0; 0 m*x_0^2+Izz];
    B = 1/r * [1 1; -c c];
    C = m*x_0*[0 x(3); -x(3) 0];
    S = [cos(x(3)) x_0*sin(x(3)); sin(x(3)) -x_0*cos(x(3)); 0 1];
    d_S = [-sin(x(3))*x(6) x_0*cos(x(3))*x(6);...
        cos(x(3))*x(6) x_0*sin(x(3))*x(6); 0 0];
    
    % Trajectory reference
    n = floor(t/time_res);
    if n == 0
        ref_q = [0;0;0];
        d_ref_q = [0;0;0];
        dd_ref_q = [0;0;0];
    else
        ref_q = [pos_x(n); pos_y(n); theta(n)];
        d_ref_q = [vel_x(n); vel_y(n); vel_theta(n)];
        dd_ref_q = [acc_x(n); acc_y(n); acc_theta(n)];
    end
    
    % Error
    e = x(1:3) - ref_q;
    d_e = x(4:6) - d_ref_q;
    
    % SMC Parameters
    s = d_e + lam*e;
    K = pinv(S*B/M)*(d_S*N-S*C/M*N+eta);
    
    % Control structure
    x_dot(1:3) = x(4:6);
    x_dot(4:6) = - lam*d_e + dd_ref_q - S*C/M*N - S*B/M*K.*sign(s);
end

