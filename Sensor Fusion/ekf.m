%% EKF function
function [x, P] = ekf(x, z, P, A, C, Q, R, G)
n = size(x,1); % Defines number of states
% Prediction stage
P = A*P*A' + G*Q*G'; % Predicts state error covariance
% Update stage
K = P*C'*pinv(C*P*C' + R); % Calculates Kalman gain
x = x + K*(z - C*x); % Updates state estimates
P = (eye(n) - K*C)*P*(eye(n) - K*C)' + K*R*K'; % Updates state error covariance
end