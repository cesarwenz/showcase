%% UKF function (see slide #15 on aL08 for summary equations)
function [x, P] = ukf(x, z, P, A, Cnon, Q, R)
% a) Initialize Stage
n = size(x,1); % Defines number of states
m = size(z,1); % Defines number of measurements
kappa = 1e-3; % Defines kappa value (user defined)
sqrtnkp = sqrtm((n+kappa)*P); % Calculates the square root of (n+kappa)*P
X = zeros(n,2*n+1); % Initializes sigma points to zero
W = zeros(1,2*n+1); % Initializes weights to zero for each sigma point
% Sigma point #1
X(:,1) = x; % Defines first sigma point
W(1) = kappa/(n+kappa); % Defines weight for first sigma point
% Sigma points 2 to 2*n+1
for i = 1:n
    X(:,i+1) = x + sqrtnkp(:,i); % Defines 2 to n+1 sigma points
    W(i+1) = 1/(2*(n+kappa)); % Defines corresponding weights
    X(:,i+n+1) = x - sqrtnkp(:,i); % Defines n+2 to 2*n+1 sigma points
    W(i+n+1) = 1/(2*(n+kappa)); % Defines corresponding weights
end
% b) Prediction Stage
for i = 1:2*n+1
    X(:,i) = A(X(:,i)); % Calculates predicted sigma points
    Z(:,i) = Cnon(X(:,i));
end
x = X*W'; % Predicted state estimates
P = Q; % Starts predicted state error covariance calculation
for i = 1:2*n+1
    P = P + W(i)*(X(:,i)-x)*transpose(X(:,i)-x); % Calculates predicted state error covariance
end
% Z = C*X; % Calculates measurements based on sigma points
zhat = Z*W'; % Calculates predicted measurements
% c) Update Stage
Pzz = R; % Starts innovation covariance calculation
for i = 1:2*n+1
    Pzz = Pzz + W(i)*(Z(:,i)-zhat)*transpose(Z(:,i)-zhat); % Calculates innovation covariance
end
Pxz = zeros(n,m); % Starts predicted cross-covariance (needed for gain calculation)
for i = 1:2*n+1
    Pxz = Pxz + W(i)*(X(:,i)-x)*transpose(Z(:,i)-zhat); % Calculates cross-covariance
end
K = Pxz*pinv(Pzz); % Calculates UKF gain
x = x + K*(z - zhat); % Updates state estimates
P = P - K*Pzz*transpose(K); % Updates state error covariance
end