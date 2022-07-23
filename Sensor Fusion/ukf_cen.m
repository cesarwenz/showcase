function [x, x1, x2, P, P1, P2] = ukf_cen(x, x1, x2, z, z1, z2, P, P1, P2, C, C1, C2, A, Q, R, R1, R2)
    [x1, P1] = ukf(x1, z1, P1, A, C1, Q, R1); % Calls EKF function
    [x2, P2] = ukf(x2, z2, P2, A, C2, Q, R2); 
    [x, P] = ukf(x, z, P, A, C, Q, R); 
end