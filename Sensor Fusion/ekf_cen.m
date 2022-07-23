function [x, x1, x2, P, P1, P2] = ekf_cen(x, x1, x2, z, z1, z2, P, P1, P2, C, C1, C2, A, Q, R, R1, R2, G)
    [x1, P1] = ekf(x1, z1, P1, A, C1(x1), Q, R1, G);
    [x2, P2] = ekf(x2, z2, P2, A, C2(x2), Q, R2, G); 
    [x, P] = ekf(x, z, P, A, C(x), Q, R, G); 
end