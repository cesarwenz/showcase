%====================================================================
% Author: Cesar Wen Zhu
% Date: April 13, 2021
%====================================================================
% Sliding mode control of SSMR tracking a circle
% Note: Note need around 3.8s to run program

clear all; close all;
global m r N lam c_r eta Izz c x_icr w_r tao mu_x mu_y g a b T

% Control parameters
N = [0.01; 0.01];
lam = 0.7;
eta1= 20;
eta2= 20;
eta3= 2;
eta = [eta1; eta2; eta3];
Izz = 18394117294.20/1e9;
K_Mat = [];

mu_x = 0.09;
mu_y = 0.3;
g = 9.81;

% Trajectory parameters
w_r = 1;
c_r = 30;

% Physical parameters
a = 0.3005;
b = 0.3155;
c = 0.3385;
m = 80.6030;
r = 0.3327;
mu_x = 0.09;
mu_y = 0.35;
x_icr = 0; % Must be between -a to b

% Simulate
x0=[0; 0; 0; 0; 0; 0]; 
tspan = [0:0.1:30];
T = [];
tic
[t,y] = ode45(@(t,x) skid(t,x), tspan, x0);
toc

figure(1);
subplot(3,1,1);
plot(t,c_r*sin(w_r*t),'k')
hold on
plot(t,y(:,1),'b')
hold on
plot(t,(y(:,1)-c_r*sin(w_r*t)),'r')
legend({'x', 'x_d','error'},'FontSize',10)
xlabel('Time (s)')
ylabel('X position (m)')

subplot(3,1,2);
plot(t,c_r*cos(w_r*t),'k')
hold on
plot(t,y(:,2),'b')
hold on
plot(t,(y(:,2)-c_r*cos(w_r*t)),'r')
legend({'y', 'y_d','error'},'FontSize',10)
xlabel('Time (s)')
ylabel('Y position (m)')



subplot(3,1,3);
plot(t,w_r*t,'k')
hold on
plot(t,y(:,3),'b')
hold on
plot(t,(y(:,3)-w_r*t),'r')
legend({'\theta', '\theta_d', 'error'},'FontSize',10)
xlabel('Time (s)')
ylabel('Angular position (rad)')

figure(2);
plot(y(:,1),y(:,2),'b')
hold on
plot(c_r*sin(w_r*t), c_r*cos(w_r*t),'r')
xlabel('x (m)')
ylabel('y (m)')
legend({'SSMR', 'Trajectory'},'FontSize',10)
xlim([-40,40])
ylim([-40,40])

% RMSE of position
rx = sqrt(sum((y(:,1)-c_r*sin(w_r*t)).^2/length(y(:,1))))
ry = sqrt(sum((y(:,2)-c_r*cos(w_r*t)).^2/length(y(:,2))))
rtheta = sqrt(sum((y(:,3)-w_r*t).^2/length(y(:,3))))


function x_dot = skid(t,x)
    global m r N lam c_r eta Izz c x_icr w_r tao mu_x mu_y g a b T
    x_dot = zeros(6,1);

    % SSMR Model
    M = [m 0; 0 m*x_icr^2+Izz];
    B = 1/r * [1 1; -c c];
    C = m*x_icr*[0 x(3); -x(3) 0];
    S = [cos(x(3)) x_icr*sin(x(3)); sin(x(3)) -x_icr*cos(x(3)); 0 1];
    d_S = [-sin(x(3))*x(6) x_icr*cos(x(3))*x(6);...
        cos(x(3))*x(6) x_icr*sin(x(3))*x(6); 0 0];
    
    % Reference trajectory
    ref_q = [c_r*sin(w_r*t); c_r*cos(w_r*t); w_r*t];
    d_ref_q = [w_r*c_r*cos(w_r*t); -w_r*c_r*sin(w_r*t); w_r];
    dd_ref_q = [-w_r^2*c_r*sin(w_r*t); -w_r^2*c_r*cos(w_r*t); 0];
    
    % Error
    e = x(1:3) - ref_q;
    d_e = x(4:6) - d_ref_q;
    
    Ffx = mu_x*g*sign(x(4));
    Ffy = mu_y*g*sign(x(5));
    Frx = 4*cos(x(3))*Ffx - 4*sin(x(3))*Ffy;
    Mr = 2*(b-a)*Ffy;
    
    R = [Frx; Mr];
    
    % SMC parameters
    s = d_e + lam*e;
    K = pinv(S*B/M)*(S*C/M*N+eta);
    
    eta_contr = [x(4)/cos(x(3)); x(6)];
    
    % Control input 
    T_eq = pinv(S*B/M)*(S*C/M*eta_contr + S/M*R - d_S*eta_contr + dd_ref_q - lam*d_e);
    T_sw = pinv(S*B/M)*(S*B/M*K.*sign(s));
    T = [T T_eq + T_sw];
    
    % Control structure
    x_dot(1:3) = x(4:6);
    x_dot(4:6) = - lam*d_e + dd_ref_q - S*C/M*N - S*B/M*K.*sign(s);
    
    
end

