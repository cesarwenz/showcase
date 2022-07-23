%% EKF function
function [x, x1, x2, P1, P2] = ekf_dec(x1, x2, z1, z2, P1, P2, C1, C2, A, Q, R1, R2, G)
    [x1, P1] = ekf(x1, z1, P1, A, C1(x1), Q, R1, G); % Calls EKF function
    [x2, P2] = ekf(x2, z2, P2, A, C2(x2), Q, R2, G); 
    Pinv = inv(P1+P2);
    x = x1 + P1*Pinv*(x2-x1);
    P = P1-P1*Pinv*P1';
end