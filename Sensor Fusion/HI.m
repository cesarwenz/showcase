%% EKF function
function [x, x1, x2, P, P1, P2] = HI(x, x1, x2, z1, z2, P1, P2, A, C1, C2, G, L1, L2, Q, R1, R2)
    
    % Prediction stage
    R1e = R1 + [C1(x1);L1]*P1*[C1(x1)' L1'];
    R2e = R2 + [C2(x2);L2]*P2*[C2(x2)' L2'];
    
    P1 = A*P1*A'+G*Q*G' - A*P1*[C1(x1)' L1']*inv(R1e)*[C1(x1);L1]*P1*A';
    P2 = A*P2*A'+G*Q*G' - A*P2*[C2(x2)' L2']*inv(R2e)*[C2(x2);L2]*P2*A';
    
    % Local filter Update
    K1 = P1*C1(x1)'*inv(eye(2)+C1(x1)*P1*C1(x1)');
    K2 = P2*C2(x2)'*inv(eye(3)+C2(x2)*P2*C2(x2)');
    
    x1 = A*x1 + K1*(z1-C1(x1)*A*x1);
    x2 = A*x2 + K2*(z2-C2(x2)*A*x2);
    
    % Global fusion Update
    Pinv = inv(P1+P2);
    x = x1 + P1*Pinv*(x2-x1);
    P = P1-P1*Pinv*P1';
end