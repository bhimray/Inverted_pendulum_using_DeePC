clear; clc; close all;

%% =========================================================
%  CART-POLE MODEL (CONTINUOUS)
%% =========================================================
M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

p = I*(M+m)+M*m*l^2;

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];

B = [0;
     (I+m*l^2)/p;
     0;
     m*l/p];

C = [1 0 0 0;
     0 0 1 0];

D = [0;0];

%% =========================================================
%  DISCRETIZATION (IMPORTANT FOR MPC COMPARISON)
%% =========================================================
Ts = 0.01;
sys_c = ss(A,B,C,D);
sys_d = c2d(sys_c, Ts);

Ad = sys_d.A;
Bd = sys_d.B;
Cd = sys_d.C;
Dd = sys_d.D;

nx = size(Ad,1);
nu = size(Bd,2);
ny = size(Cd,1);

%% =========================================================
%  LQR DESIGN (DISCRETE)
%% =========================================================
Q = Cd'*Cd;
Q(1,1) = 5000;   % cart position weight
Q(3,3) = 100;    % angle weight

R = 1;

K = dlqr(Ad, Bd, Q, R);

fprintf('\nLQR gain K:\n');
disp(K);

%% =========================================================
%  REFERENCE TRACKING (CRITICAL FIX)
%% =========================================================
% We track cart position only
% reference: x -> r, phi -> 0

Cref = [1 0 0 0];   % output we track

% Feedforward gain for zero steady-state error
Nbar = inv(Cref * inv(eye(nx) - (Ad - Bd*K)) * Bd);

fprintf('\nFeedforward gain Nbar:\n');
disp(Nbar);

%% =========================================================
%  SIMULATION SETTINGS
%% =========================================================
Tsim = 5;                 % seconds
Nsim = Tsim / Ts;

x = zeros(nx, Nsim+1);
y = zeros(ny, Nsim);
u = zeros(nu, Nsim);

% Initial condition
x(:,1) = [0.0; 0; 0.3; 0];

% Reference
r = 0.2;   % cart position reference in meters

%% =========================================================
%  CLOSED-LOOP SIMULATION
%% =========================================================
lqr_total_timer = tic;
lqr_step_time_hist = zeros(1, Nsim);

for k = 1:Nsim
    lqr_step_timer = tic;
    
    % control law: u = -Kx + Nbar*r
    u(:,k) = -K*x(:,k) + Nbar*r;
    
    % propagate system
    x(:,k+1) = Ad*x(:,k) + Bd*u(:,k);
    
    % output
    y(:,k) = Cd*x(:,k);

    lqr_step_time_hist(k) = toc(lqr_step_timer);
end

lqr_total_time = toc(lqr_total_timer);
lqr_avg_step_time = mean(lqr_step_time_hist);
lqr_max_step_time = max(lqr_step_time_hist);

fprintf('\n=== LQR computation time ===\n');
fprintf('Sampling time Ts            = %.6f s (%.2f ms)\n', Ts, Ts*1000);
fprintf('Total computation time      = %.6f s (%.2f ms)\n', lqr_total_time, lqr_total_time*1000);
fprintf('Average step time           = %.6f s (%.4f ms)\n', lqr_avg_step_time, lqr_avg_step_time*1000);
fprintf('Worst-case step time        = %.6f s (%.4f ms)\n', lqr_max_step_time, lqr_max_step_time*1000);

t = (0:Nsim-1)*Ts;

%% =========================================================
%  COST (for fair comparison with DeePC)
%% =========================================================
Qy = diag([500, 120]);
Ru = 1;

J = 0;
for k = 1:Nsim
    e = y(:,k) - [r; 0];
    J = J + e'*Qy*e + u(:,k)'*Ru*u(:,k);
end

fprintf('\nLQR cumulative cost: %.6f\n', J);

%% =========================================================
%  PLOTS
%% =========================================================
t_desired = 1.5;          % presentation marker: desired value reached

figure('Name','Correct LQR Benchmark','Color','w');

subplot(3,1,1)
plot(t, y(1,:), 'LineWidth',2)
hold on
xline(t_desired, '--r', 'Target reached', ...
    'LineWidth', 1.5, 'LabelOrientation', 'horizontal', ...
    'LabelVerticalAlignment', 'bottom');
ylabel('cart position (m)')
title('LQR (correct tracking, discrete-time)')
grid on

subplot(3,1,2)
plot(t, y(2,:), 'LineWidth',2)
hold on
xline(t_desired, '--r', 'LineWidth', 1.5);
ylabel('pendulum angle (rad)')
grid on

subplot(3,1,3)
plot(t, u, 'LineWidth',2)
hold on
xline(t_desired, '--r', 'LineWidth', 1.5);
xlabel('Time (s)')
ylabel('control input u')
grid on

%% =========================================================
%  SAVE FOR COMPARISON
%% =========================================================
xLqr = x;
yLqr = y;
uLqr = u;
JLqr = J;

% Aliases used by animation.m when this LQR result file is selected.
xAnim = xLqr;
uAnim = uLqr;
controller_name = 'LQR';

save('lqr_baseline.mat', ...
    'x','y','u','t','K','Nbar','Ad','Bd','Cd','Dd','Ts','J', ...
    'lqr_total_time','lqr_avg_step_time','lqr_max_step_time','lqr_step_time_hist', ...
    'xLqr','yLqr','uLqr','JLqr', ...
    'xAnim','uAnim','controller_name');
