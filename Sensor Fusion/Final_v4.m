% Kalman Based Multisensor Fusion Algorithms
% Author: C. Wen (cwenzhu@uoguelph.ca)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Setting up workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; % Clears workspace
close all; % Closes all open figures
clc; % Clears screen

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Initializing parameters and variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tf = 500; % Final time in simulation
T = 5; % Sample rate
t = 0:T:tf; % Time vector
n = 9; % Defines number of states
s = 2; % Define amount of sensor
m_ir = 2; % Defines number of measurements for IRST
m_rr = 3; % Define number of measurement for Radar
m_fr = 5; % Define number of measurement for both Radar and IRST (centralized fusion)

% Initialize system matrix
A = [1 T T^2/2 0 0 0     0 0 0;
    0 1 T     0 0 0     0 0 0;
    0 0 1     0 0 0     0 0 0;
    0 0 0     1 T T^2/2 0 0 0;
    0 0 0     0 1 T     0 0 0;
    0 0 0     0 0 1     0 0 0;
    0 0 0     0 0 0     1 T T^2/2;
    0 0 0     0 0 0     0 1 T;
    0 0 0     0 0 0     0 0 1];

% Initialize system matrix for ukf function
A_ukf = @(X) [X(1)+X(2)*T+X(3)*T^2/2;
    X(2)+X(3)*T;
    X(3);
    X(4)+X(5)*T+X(6)*T^2/2;
    X(5)+X(6)*T;
    X(6);
    X(7)+X(8)*T+X(9)*T^2/2;
    X(8)+X(9)*T;
    X(9)];

% Initialize process noise-gain matrix
G = [T^3/6 0 0 0 0 0 0 0 0;
    T^2/2 0 0 0 0 0 0 0 0;
    T 0 0 0 0 0 0 0 0;
    0 0 0 T^3/6 0 0 0 0 0;
    0 0 0 T^2/2 0 0 0 0 0;
    0 0 0 T 0 0 0 0 0;
    0 0 0 0 0 0 T^3/6 0 0;
    0 0 0 0 0 0 T^2/2 0 0;
    0 0 0 0 0 0 T 0 0];

% Initialize non-linear measurement for IRST measurement
C_ir  = @(X) [atan(X(1)/X(4));
    atan(X(7)/sqrt(sum(X([1,4]).^2)))];

% Initialize non-linear measurement for radar measurement
C_rr = @(X) [atan(X(1)/X(4));
    atan(X(7)/sqrt(sum(X([1,4]).^2)));
    sqrt(sum(X([1, 4, 7]).^2))];

% Initialize non-linear measurement for combined measurement (centralized fusion)
C_fr  = @(X) [atan(X(1)/X(4));
    atan(X(7)/sqrt(sum(X([1,4]).^2)));
    atan(X(1)/X(4));
    atan(X(7)/sqrt(sum(X([1,4]).^2)));
    sqrt(sum(X([1, 4, 7]).^2))];

% Initialize linearized Jacobian for IRST measurement
Clin_ir = @(X) [X(4)/sum(X([1,4]).^2) 0 0 -X(1)/sum(X([1,4]).^2) 0 0 0 0 0;
    -X(1)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 -X(4)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 sqrt(sum(X([1,4]).^2))/(sum(X([1,4,7]).^2)) 0 0];

% Initialize linearized Jacobian for radar measurement
Clin_rr = @(X) [X(4)/sum(X([1,4]).^2) 0 0 -X(1)/sum(X([1,4]).^2) 0 0 0 0 0;
    -X(1)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 -X(4)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 sqrt(sum(X([1,4]).^2))/(sum(X([1,4,7]).^2)) 0 0;
    X(1)/sqrt(sum(X([1,4,7]).^2)) 0 0 X(4)/sqrt(sum(X([1,4,7]).^2)) 0 0 X(7)/sqrt(sum(X([1,4,7]).^2)) 0 0];

% Initialize linearized Jacobian for combined measurement (centralized fusion)
Clin_fr = @(X) [X(4)/sum(X([1,4]).^2) 0 0 -X(1)/sum(X([1,4]).^2) 0 0 0 0 0;
    -X(1)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 -X(4)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 sqrt(sum(X([1,4]).^2))/(sum(X([1,4,7]).^2)) 0 0;
    X(4)/sum(X([1,4]).^2) 0 0 -X(1)/sum(X([1,4]).^2) 0 0 0 0 0;
    -X(1)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 -X(4)*X(7)/(sum(X([1,4,7]).^2)*sqrt(sum(X([1,4]).^2))) 0 0 sqrt(sum(X([1,4]).^2))/(sum(X([1,4,7]).^2)) 0 0;
    X(1)/sqrt(sum(X([1,4,7]).^2)) 0 0 X(4)/sqrt(sum(X([1,4,7]).^2)) 0 0 X(7)/sqrt(sum(X([1,4,7]).^2)) 0 0];

% Initializes states to zero
x = zeros(n, length(t));

% Initialize initial conditions
x(:,1) = [10,2,0.5,10,5,0.3,10,1,0.01]; 

% Initialize centralized EKF states
x_ir_c_ekf = x; x_ir_d_ekf = x; x_ir_c_ukf = x; x_ir_d_ukf = x; x_ir_gf = x; x_ir_hi = x;
x_rr_c_ekf = x; x_rr_d_ekf = x; x_rr_c_ukf = x; x_rr_d_ukf = x; x_rr_gf = x; x_rr_hi = x;
x_fr_c_ekf = x; x_fr_d_ekf = x; x_fr_c_ukf = x; x_fr_d_ukf = x; x_fr_gf = x; x_fr_hi = x;

% Initializes measurements for algorithms using EKF
z_ir_ekf = zeros(m_ir, length(t)); 
z_rr_ekf = zeros(m_rr, length(t));
z_fr_ekf = zeros(m_fr, length(t));

% Initializes measurements for algorithms
z_c_ekf = zeros(m_rr, length(t)); 
z_d_ekf = zeros(m_rr, length(t)); 
z_c_ukf = zeros(m_rr, length(t)); 
z_d_ukf = zeros(m_rr, length(t)); 
z_gf = zeros(m_rr, length(t)); 
z_hi = zeros(m_rr, length(t)); 

% Initialize raw measurments
z_ir = zeros(m_ir, length(t)); 
z_rr = zeros(m_rr, length(t)); 
z_fr = zeros(m_fr, length(t)); 
z_t = zeros(m_rr, length(t));

% IRST sensor accuracies
irazivar = 10^-3; % azimuth accuracy in rad (0.06 degs)
irelevar = 10^-3; % elevation accuracy in rad (0.06 degs)

% Radar accuracies
rrazivar = 10^-2; % azimuth resolution in rad (0.6 degs)
rrelevar = 10^-2; % elevation resolution in rad (0.6 degs)
rrranvar = 30; % range resolution (30 m)

q = 10^-6; % process noise variance
Q = eye(9,9)*q; % process noise covaraince matrix

% measurement noise covaraince matrix
R_ir = diag([irazivar irelevar]); 
R_rr = diag([rrazivar rrelevar rrranvar]); 
R_fr = diag([irazivar irelevar rrazivar rrelevar rrranvar]);

% GF design parameter
% Sum of (1/gamme_i) has to be 1
gamma_GF_ir = 2; 
gamma_GF_rr = 2;

% HI design parameter
gamma_HI_ir = 1.02; % design gamma value for IRST (usually around 1)
gamma_HI_ir_s = gamma_HI_ir^2; % squared gamma for IRST
gamma_HI_rr = 1.02; % design gamma value for radar (usually around 1)
gamma_HI_rr_s = gamma_HI_rr^2; % squared gamma for radar

% Generates the H-infinity measurement noise
R_HI_ir = [eye(8,n); zeros(1,8) -gamma_HI_ir_s];
R_HI_rr = [eye(8,n); zeros(1,8) -gamma_HI_rr_s];

% Initialize H-infinity coefficient matrix (generally defined by unit
% matrix)
L_ir = eye(9-m_ir, n); 
L_rr = eye(9-m_rr, n); 

% Generates the system noise
w = mvnrnd(zeros(length(t),n),Q)';

% Generates the measurement noise
v_ir = mvnrnd(zeros(length(t),m_ir),R_ir)';
v_rr = mvnrnd(zeros(length(t),m_rr),R_rr)';
v_fr = mvnrnd(zeros(length(t),m_fr),R_fr)';

% Generates state error covariance matrices
P_ir_c_ekf = Q; P_ir_d_ekf = Q; P_ir_c_ukf = Q; P_ir_d_ukf = Q; P_ir_gf = Q; P_ir_hi = Q; 
P_rr_c_ekf = Q; P_rr_d_ekf = Q; P_rr_c_ukf = Q; P_rr_d_ukf = Q; P_rr_gf = Q; P_rr_hi = Q; 
P_fr_c_ekf = Q; P_fr_d_ekf = Q; P_fr_c_ukf = Q; P_fr_d_ukf = Q; P_fr_gf = Q; P_fr_hi = Q;

% Initialize squared error of fused states and measurements
SE_c_ekf = zeros(n,1); SE_c_ekf_m = zeros(m_rr,1);
SE_d_ekf = zeros(n,1); SE_d_ekf_m = zeros(m_rr,1);
SE_c_ukf = zeros(n,1); SE_c_ukf_m = zeros(m_rr,1);
SE_d_ukf = zeros(n,1); SE_d_ukf_m = zeros(m_rr,1);
SE_gf = zeros(n,1); SE_gf_m = zeros(m_rr,1);
SE_hi = zeros(n,1); SE_hi_m = zeros(m_rr,1);

% Initialize root mean square error of fused states, meseasurements, and
% kinematics, respectively
RMSE_c_ekf = zeros(n,1); RMSE_c_ekf_m = zeros(m_rr,1); RMSE_c_ekf_k = zeros(m_ir,1);
RMSE_d_ekf = zeros(n,1); RMSE_d_ekf_m = zeros(m_rr,1); RMSE_d_ekf_k = zeros(m_ir,1);
RMSE_c_ukf = zeros(n,1); RMSE_c_ukf_m = zeros(m_rr,1); RMSE_c_ukf_k = zeros(m_ir,1);
RMSE_d_ukf = zeros(n,1); RMSE_d_ukf_m = zeros(m_rr,1); RMSE_d_ukf_k = zeros(m_ir,1);
RMSE_gf = zeros(n,1); RMSE_gf_m = zeros(m_rr,1); RMSE_gf_k = zeros(m_ir,1);
RMSE_hi = zeros(n,1); RMSE_hi_m = zeros(m_rr,1); RMSE_hi_k = zeros(m_ir,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Simulate dynamics and fusion algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:length(t)-1 % For loop that simulates dynamics
    % Target tracking equation
    x(:,k+1) = A*x(:,k) + G*w(:,k); 
    
    % Raw Measurement 
    z_ir(:,k+1) = C_ir(x(:,k+1))+ v_ir(:,k+1); 
    z_rr(:,k+1) = C_rr(x(:,k+1))+ v_rr(:,k+1);
    z_fr(:,k+1) = C_fr(x(:,k+1))+ v_fr(:,k+1);
    
    % True trajectory 
    z_t(:,k+1) = C_rr(x(:,k));
 
    % Linearized measurement equations
    z_ir_ekf(:,k+1) = Clin_ir(x(:,k))*x(:,k+1)+ v_ir(:,k+1); 
    z_rr_ekf(:,k+1) = Clin_rr(x(:,k))*x(:,k+1)+ v_rr(:,k+1);
    z_fr_ekf(:,k+1) = Clin_fr(x(:,k))*x(:,k+1)+ v_fr(:,k+1);
    
    % Simulate sensor fusion algorithms
    % CEKF
    [x_fr_c_ekf(:,k+1), x_ir_c_ekf(:,k+1), x_rr_c_ekf(:,k+1), P_fr_c_ekf(:,:,k+1), P_ir_c_ekf(:,:,k+1), P_rr_c_ekf(:,:,k+1)] = ekf_cen(x_fr_c_ekf(:,k), x_ir_c_ekf(:,k), x_rr_c_ekf(:,k), z_fr_ekf(:,k+1), z_ir_ekf(:,k+1), z_rr_ekf(:,k+1),...
        P_fr_c_ekf(:,:,k), P_ir_c_ekf(:,:,k), P_rr_c_ekf(:,:,k), Clin_fr, Clin_ir, Clin_rr, A, Q, R_fr, R_ir, R_rr, G);
    % DEKF
    [x_fr_d_ekf(:,k+1), x_ir_d_ekf(:,k+1), x_rr_d_ekf(:,k+1), P_ir_d_ekf(:,:,k+1), P_rr_d_ekf(:,:,k+1)] = ekf_dec(x_ir_d_ekf(:,k), x_rr_d_ekf(:,k), z_ir_ekf(:,k+1), z_rr_ekf(:,k+1), P_ir_d_ekf(:,:,k), P_rr_d_ekf(:,:,k), Clin_ir, Clin_rr, A, Q, R_ir, R_rr, G);
    % CUKF
    [x_fr_c_ukf(:,k+1), x_ir_c_ukf(:,k+1), x_rr_c_ukf(:,k+1), P_fr_c_ukf(:,:,k+1), P_ir_c_ukf(:,:,k+1), P_rr_c_ukf(:,:,k+1)] = ukf_cen(x_fr_c_ukf(:,k), ...
        x_ir_c_ukf(:,k), x_rr_c_ukf(:,k), z_fr(:,k+1), z_ir(:,k+1), z_rr(:,k+1), ...
        P_fr_c_ukf(:,:,k), P_ir_c_ukf(:,:,k), P_rr_c_ukf(:,:,k), C_fr, C_ir, C_rr, A_ukf, Q, R_fr, R_ir, R_rr);
    % DUKF
    [x_fr_d_ukf(:,k+1), x_ir_d_ukf(:,k+1), x_rr_d_ukf(:,k+1), P_ir_d_ukf(:,:,k+1), P_rr_d_ukf(:,:,k+1)] = ukf_dec(x_ir_d_ukf(:,k), x_rr_d_ukf(:,k), z_ir(:,k+1), z_rr(:,k+1), ...
        P_ir_d_ukf(:,:,k), P_rr_d_ukf(:,:,k), C_ir, C_rr, A_ukf, Q, R_ir, R_rr);
    % GF
    [x_fr_gf(:,k+1), x_ir_gf(:,k+1), x_rr_gf(:,k+1), P_fr_gf(:,:,k+1)] = gainfusion(x_fr_gf(:,k), z_ir_ekf(:,k+1), z_rr_ekf(:,k+1), P_fr_gf(:,:,k), A, Clin_ir, Clin_rr, G, gamma_GF_ir, gamma_GF_rr, Q, R_ir, R_rr, s);
    % HI
    [x_fr_hi(:,k+1), x_ir_hi(:,k+1), x_rr_hi(:,k+1), P_fr_hi(:,:,k+1), P_ir_hi(:,:,k+1), P_rr_hi(:,:,k+1)] = HI(x_fr_hi(:,k), x_ir_hi(:,k), x_rr_hi(:,k), z_ir_ekf(:,k+1), z_rr_ekf(:,k+1), P_ir_hi(:,:,k), P_rr_hi(:,:,k), A, Clin_ir, Clin_rr, G, L_ir, L_rr, Q, R_HI_ir, R_HI_rr);
    
    % Convert states to measurements
    z_c_ekf(:,k+1) = C_rr(x_fr_c_ekf(:,k+1));
    z_d_ekf(:,k+1) = C_rr(x_fr_d_ekf(:,k+1));
    z_c_ukf(:,k+1) = C_rr(x_fr_c_ukf(:,k+1));
    z_d_ukf(:,k+1) = C_rr(x_fr_d_ukf(:,k+1));
    z_gf(:,k+1) = C_rr(x_fr_gf(:,k+1));
    z_hi(:,k+1) = C_rr(x_fr_hi(:,k+1));
    
    % Compute squared error for fused states
    SE_c_ekf(:,k+1) = (x(:,k+1) - x_fr_c_ekf(:,k+1)).^2;
    SE_d_ekf(:,k+1) = (x(:,k+1) - x_fr_d_ekf(:,k+1)).^2;
    SE_c_ukf(:,k+1) = (x(:,k+1) - x_fr_c_ukf(:,k+1)).^2;
    SE_d_ukf(:,k+1) = (x(:,k+1) - x_fr_d_ukf(:,k+1)).^2;
    SE_gf(:,k+1) = (x(:,k+1) - x_fr_gf(:,k+1)).^2;
    SE_hi(:,k+1) = (x(:,k+1) - x_fr_hi(:,k+1)).^2; 
    
    % Compute squared error for fused measurements
    SE_c_ekf_m(:,k+1) = (z_t(:,k+1) - z_c_ekf(:,k+1)).^2;
    SE_d_ekf_m(:,k+1) = (z_t(:,k+1) - z_d_ekf(:,k+1)).^2;
    SE_c_ukf_m(:,k+1) = (z_t(:,k+1) - z_c_ukf(:,k+1)).^2;
    SE_d_ukf_m(:,k+1) = (z_t(:,k+1) - z_d_ukf(:,k+1)).^2;
    SE_gf_m(:,k+1) = (z_t(:,k+1) - z_gf(:,k+1)).^2;
    SE_hi_m(:,k+1) = (z_t(:,k+1) - z_hi(:,k+1)).^2; 
end

% RMSE calculation for states
for k = 1:n
    
    RMSE_c_ekf(k) = (sum(SE_c_ekf(k,:))/length(t))^0.5;
    RMSE_d_ekf(k) = (sum(SE_d_ekf(k,:))/length(t))^0.5;
    RMSE_c_ukf(k) = (sum(SE_c_ukf(k,:))/length(t))^0.5;
    RMSE_d_ukf(k) = (sum(SE_d_ukf(k,:))/length(t))^0.5;
    RMSE_gf(k) = (sum(SE_gf(k,:))/length(t))^0.5;
    RMSE_hi(k) = (sum(SE_hi(k,:))/length(t))^0.5;
end

% RMSE calculation for measurements
for k = 1:m_rr
    RMSE_c_ekf_m(k) = (sum(SE_c_ekf_m(k,:))/length(t))^0.5;
    RMSE_d_ekf_m(k) = (sum(SE_d_ekf_m(k,:))/length(t))^0.5;
    RMSE_c_ukf_m(k) = (sum(SE_c_ukf_m(k,:))/length(t))^0.5;
    RMSE_d_ukf_m(k) = (sum(SE_d_ukf_m(k,:))/length(t))^0.5;
    RMSE_gf_m(k) = (sum(SE_gf_m(k,:))/length(t))^0.5;
    RMSE_hi_m(k) = (sum(SE_hi_m(k,:))/length(t))^0.5;
end
% RMSE calculation for kinematics
for k=0:2
    RMSE_c_ekf_k(k+1,:) = sqrt(sum(RMSE_c_ekf([1+k, 4+k, 7+k],:).^2));
    RMSE_d_ekf_k(k+1,:) = sqrt(sum(RMSE_d_ekf([1+k, 4+k, 7+k],:).^2));
    RMSE_c_ukf_k(k+1,:) = sqrt(sum(RMSE_c_ukf([1+k, 4+k, 7+k],:).^2));
    RMSE_d_ukf_k(k+1,:) = sqrt(sum(RMSE_d_ukf([1+k, 4+k, 7+k],:).^2));
    RMSE_gf_k(k+1,:) = sqrt(sum(RMSE_gf([1+k, 4+k, 7+k],:).^2));
    RMSE_hi_k(k+1,:) = sqrt(sum(RMSE_hi([1+k, 4+k, 7+k],:).^2));
end

% Make RMSE for states
State = ['1'; '2'; '3'; '4'; '5'; '6'; '7'; '8'; '9']; % Creates label for table of RMSE results
Results_RMSE = table(State,RMSE_c_ekf,RMSE_d_ekf,RMSE_c_ukf,RMSE_d_ukf,RMSE_gf,RMSE_hi); % Presents the RMSE results in the form of a table
Results_RMSE.Properties.VariableNames = {'State','CEKF','DEKF','CUKF','DUKF','GF','HI'}; % Labels the columns
fprintf('State RMSE: \n\n' ); % Displays title for results
disp(Results_RMSE); % Displays RMSE results table

% Make RMSE for measurements
State = ['1'; '2'; '3']; % Creates label for table of RMSE results
Results_RMSE = table(State,RMSE_c_ekf_m,RMSE_d_ekf_m,RMSE_c_ukf_m,RMSE_d_ukf_m,RMSE_gf_m,RMSE_hi_m); % Presents the RMSE results in the form of a table
Results_RMSE.Properties.VariableNames = {'State','CEKF','DEKF','CUKF','DUKF','GF','HI'}; % Labels the columns
fprintf('Measurement RMSE: \n\n' ); % Displays title for results
disp(Results_RMSE); % Displays RMSE results table


% RMSE for kinematics
State = ['1';'2';'3']; % Creates label for table of RMSE results
Results_RMSE = table(State,RMSE_c_ekf_k,RMSE_d_ekf_k,RMSE_c_ukf_k,RMSE_d_ukf_k,RMSE_gf_k,RMSE_hi_k); % Presents the RMSE results in the form of a table
Results_RMSE.Properties.VariableNames = {'State','CEKF','DEKF','CUKF','DUKF','GF','HI'}; % Labels the columns
fprintf('Kinematic RMSE: \n\n' ); % Displays title for results
disp(Results_RMSE); % Displays RMSE results table

% Plot kinematics
% figure;
% subplot(311);
% plot(t, x(1,:)); hold all; plot(t, x_fr_c_ekf(1,:), 'r'); hold on; plot(t, x_fr_d_ekf(1,:)); hold on; plot(t, x_fr_c_ukf(1,:)); hold on; plot(t, x_fr_d_ukf(1,:)); hold on; plot(t, x_fr_gf(1,:)); hold on; plot(t, x_fr_hi(1,:));
% subplot(312);
% plot(t, x(4,:)); hold all; plot(t, x_fr_c_ekf(4,:)); hold on; plot(t, x_fr_d_ekf(4,:)); hold on; plot(t, x_fr_c_ukf(4,:)); hold on; plot(t, x_fr_d_ukf(4,:)); hold on; plot(t, x_fr_gf(4,:)); hold on; plot(t, x_fr_hi(4,:));
% subplot(313);
% plot(t, x(7,:)); hold all; plot(t, x_fr_c_ekf(7,:)); hold on; plot(t, x_fr_d_ekf(7,:)); hold on; plot(t, x_fr_c_ukf(7,:)); hold on; plot(t, x_fr_d_ukf(7,:)); hold on; plot(t, x_fr_gf(7,:)); hold on; plot(t, x_fr_hi(7,:));
% xlabel('Time (sec)'); ylabel('Z'); legend('True','CEKF', 'DEKF', 'CUKF', 'DUKF', 'GF', 'HI'); 
% figure;
% subplot(311);
% plot(t, x(2,:)); hold all; plot(t, x_fr_c_ekf(2,:)); hold on; plot(t, x_fr_d_ekf(2,:)); hold on; plot(t, x_fr_c_ukf(2,:)); hold on; plot(t, x_fr_d_ukf(2,:)); hold on; plot(t, x_fr_gf(2,:)); hold on; plot(t, x_fr_hi(2,:));
% subplot(312);
% plot(t, x(5,:)); hold all; plot(t, x_fr_c_ekf(5,:)); hold on; plot(t, x_fr_d_ekf(5,:)); hold on; plot(t, x_fr_c_ukf(5,:)); hold on; plot(t, x_fr_d_ukf(5,:)); hold on; plot(t, x_fr_gf(5,:)); hold on; plot(t, x_fr_hi(5,:));
% subplot(313);
% plot(t, x(8,:)); hold all; plot(t, x_fr_c_ekf(8,:)); hold on; plot(t, x_fr_d_ekf(8,:)); hold on; plot(t, x_fr_c_ukf(8,:)); hold on; plot(t, x_fr_d_ukf(8,:)); hold on; plot(t, x_fr_gf(8,:)); hold on; plot(t, x_fr_hi(8,:));
% xlabel('Time (sec)'); ylabel('Z'); legend('True','CEKF', 'DEKF', 'CUKF', 'DUKF', 'GF', 'HI'); 
% figure;
% subplot(311);
% plot(t, x(3,:)); hold all; plot(t, x_fr_c_ekf(3,:)); hold on; plot(t, x_fr_d_ekf(3,:)); hold on; hold on; plot(t, x_fr_c_ukf(3,:)); hold on; plot(t, x_fr_d_ukf(3,:)); plot(t, x_fr_gf(3,:)); hold on; plot(t, x_fr_hi(3,:));
% subplot(312);
% plot(t, x(6,:)); hold all; plot(t, x_fr_c_ekf(6,:)); hold on; plot(t, x_fr_d_ekf(6,:)); hold on; hold on; plot(t, x_fr_c_ukf(6,:)); hold on; plot(t, x_fr_d_ukf(6,:)); plot(t, x_fr_gf(6,:)); hold on; plot(t, x_fr_hi(6,:));
% subplot(313);
% plot(t, x(9,:)); hold all; plot(t, x_fr_c_ekf(9,:)); hold on; plot(t, x_fr_d_ekf(9,:)); hold on; hold on; plot(t, x_fr_c_ukf(9,:)); hold on; plot(t, x_fr_d_ukf(9,:)); plot(t, x_fr_gf(9,:)); hold on; plot(t, x_fr_hi(9,:));
% 
% xlabel('Time (sec)'); ylabel('Z'); legend('True','CEKF', 'DEKF', 'CUKF', 'DUKF', 'GF', 'HI'); % Plots differential pressure with time

% Plot individual measurements with fusion 
figure;
plot(t, z_ir(1,:), '-b', 'LineWidth', 1); hold all; plot(t, z_rr(1,:), ':r', 'LineWidth', 1.5); hold on; plot(t, z_t(1,:), 'k','LineWidth', 1); 
% hold on; plot(t, z_c_ekf(1,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_d_ekf(1,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_c_ukf(1,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_d_ukf(1,:), 'g','LineWidth', 1); 
hold on; plot(t, z_gf(1,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_hi(1,:), 'g','LineWidth', 1); 
xlabel ('Time (s)');
ylabel ('Azimuth Angle (rads)');
% legend('IRST','Radar', 'True', 'CEKF');
% legend('IRST','Radar', 'True', 'DEKF');
% legend('IRST','Radar', 'True', 'CUKF');
% legend('IRST','Radar', 'True', 'DUKF');
legend('IRST','Radar', 'True', 'GF');
% legend('IRST','Radar', 'True', 'HI');
figure;
plot(t, z_ir(2,:), '-b', 'LineWidth', 1); hold all; plot(t, z_rr(2,:), ':r', 'LineWidth', 1.5); hold on; plot(t, z_t(2,:), 'k','LineWidth', 1);
% hold on; plot(t, z_c_ekf(2,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_d_ekf(2,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_c_ukf(2,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_d_ukf(2,:), 'g','LineWidth', 1); 
hold on; plot(t, z_gf(2,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_hi(2,:), 'g','LineWidth', 1); 
xlabel ('Time (s)');
ylabel ('Elevation Angle (rads)');
% legend('IRST','Radar', 'True', 'CEKF');
% legend('IRST','Radar', 'True', 'DEKF');
% legend('IRST','Radar', 'True', 'CUKF');
% legend('IRST','Radar', 'True', 'DUKF');
legend('IRST','Radar', 'True', 'GF');
% legend('IRST','Radar', 'True', 'HI');
figure;
plot(t, zeros(1, length(t)), '-b', 'LineWidth', 1); hold all; plot(t, z_rr(3,:), ':r', 'LineWidth', 2); hold on; plot(t, z_t(3,:), 'k','LineWidth', 1);

% hold on; plot(t, z_c_ekf(3,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_d_ekf(3,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_c_ukf(3,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_d_ukf(3,:), 'g','LineWidth', 1); 
hold on; plot(t, z_gf(3,:), 'g','LineWidth', 1); 
% hold on; plot(t, z_hi(3,:), 'g','LineWidth', 1); 
xlabel ('Time (s)');
ylabel ('Range (m)');
% legend('IRST','Radar', 'True', 'CEKF');
% legend('IRST','Radar', 'True', 'DEKF');
% legend('IRST','Radar', 'True', 'CUKF');
% legend('IRST','Radar', 'True', 'DUKF');
legend('IRST','Radar', 'True', 'GF');
% legend('IRST','Radar', 'True', 'HI');
 

