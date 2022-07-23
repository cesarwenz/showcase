%% EKF function
function [x, x_ir, x_rr, P] = gainfusion(x, z1, z2, P, A, C1, C2, G, gamma1, gamma2, Q, R1, R2, s)
    n = size(x,1); % Defines number of states
    % Global filter prediction
    x = A*x;
    P = A*P*A' + G*Q*G';
    
    % Local filter reset
    x_ir = x;
    x_rr = x;
    
    % Local filter update
    K1 = (1/gamma1)*P*C1(x)'*inv(C1(x)*P*C1(x)'+(1/gamma1)*R1);
    K2 = (1/gamma2)*P*C2(x)'*inv(C2(x)*P*C2(x)'+(1/gamma2)*R2);
    
    x_ir = x + K1*(z1-C1(x)*x);
    x_rr = x + K2*(z2-C2(x)*x);
    
    x = x_ir + x_rr - (s-1)*x;
    P = (eye(n) - (K1*C1(x)+K2*C2(x)))*P*(eye(n) - (K1*C1(x)+K2*C2(x)))' + K1*R1*K1'+ K2*R2*K2';
    
end