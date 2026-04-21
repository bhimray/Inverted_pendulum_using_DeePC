clear
clc
close all

%% Physical parameters
M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.81;
l = 0.3;

Ts = 0.01;

%% Linearized system
q = (M+m)*(I+m*l^2)-(m*l)^2;

A = [0 1 0 0;
     0 -(I+m*l^2)*b/q (m^2*g*l^2)/q 0;
     0 0 0 1;
     0 -(m*l*b)/q m*g*l*(M+m)/q 0];

B = [0;
     (I+m*l^2)/q;
     0;
     m*l/q];

C = [1 0 0 0;
     0 0 1 0];

D = zeros(2,1);

sysd = c2d(ss(A,B,C,D),Ts);
% step(ss(A,B,C,D,Ts))

Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;

nx = size(Ad,1);
nu = size(Bd,2);
ny = size(Cd,1);

%% MPC parameters
N = 30; % getting infeasiblity for N < 30 because of terminal constraint

Q = diag([200 200]);
R = 0.01;
r = [0.2; 0];   % desired output

umin = -10;
umax = 10;

%% Simulation
Tsim = 500;
t = (0:Tsim-1)*Ts;

x = [0.0;0.0;0.0;0.0];

x_hist = zeros(nx,Tsim);
u_hist = zeros(nu,Tsim);
y_hist = zeros(ny,Tsim);
solve_time_hist = zeros(1,Tsim);
step_time_hist = zeros(1,Tsim);

ops = sdpsettings('solver','OSQP','verbose',1);

for k = 1:Tsim

    step_timer = tic;

    %% Decision variables
    x_var = sdpvar(nx,N+1);   % predicted states
    u_var = sdpvar(nu,N);     % predicted inputs

    con = [];
    obj = 0;

    %% Initial condition
    con = [con, x_var(:,1) == x];

    %% Build dynamics recursively
    for i = 1:N

        % dynamics
        con = [con, x_var(:,i+1) == Ad*x_var(:,i) + Bd*u_var(:,i)];

        %input constraints
        con = [con, umin <= u_var(:,i) <= umax];

        % output
        y_i = Cd*x_var(:,i);

        % cost
        obj = obj + ((y_i - r)'*Q*(y_i - r)) + (u_var(:,i)'*R*u_var(:,i));

    end
    x_ref = [0.2; 0; 0; 0];
    con = [con, x_var(:,N+1) == x_ref];
    %% Solve
    solve_timer = tic;
    sol = optimize(con,obj,ops);
    solve_time_hist(k) = toc(solve_timer);

    if sol.problem ~= 0
        disp('Solver failed');
    end

    %% Apply first control
    u_opt = value(u_var);
    u_apply = u_opt(:,1);

    %% System update
    x = Ad*x + Bd*u_apply;
    y = Cd*x;

    %% Store
    x_hist(:,k) = x;
    u_hist(:,k) = u_apply;
    y_hist(:,k) = y;
    step_time_hist(k) = toc(step_timer);

end

%% Real-time computation check
max_solve_time = max(solve_time_hist);
avg_solve_time = mean(solve_time_hist);
max_step_time = max(step_time_hist);
avg_step_time = mean(step_time_hist);

fprintf('\nRecursive MPC real-time check:\n');
fprintf('Sampling time Ts              = %.6f s (%.2f ms)\n', Ts, Ts*1000);
fprintf('Average optimization time     = %.6f s (%.2f ms)\n', avg_solve_time, avg_solve_time*1000);
fprintf('Worst-case optimization time  = %.6f s (%.2f ms)\n', max_solve_time, max_solve_time*1000);
fprintf('Average control-step time     = %.6f s (%.2f ms)\n', avg_step_time, avg_step_time*1000);
fprintf('Worst-case control-step time  = %.6f s (%.2f ms)\n', max_step_time, max_step_time*1000);

if max_step_time < Ts
    fprintf('Result: feasible for real-time computation based on total control-step time.\n\n');
else
    fprintf('Result: not feasible for real-time computation based on total control-step time.\n\n');
end

%% Plots
figure
subplot(3,1,1)
plot(t,y_hist(1,:),'LineWidth',2)
title('Cart Position (Recursive MPC)')
grid on

subplot(3,1,2)
plot(t,y_hist(2,:),'LineWidth',2)
title('Pendulum Angle (Recursive MPC)')
grid on

subplot(3,1,3)
plot(t,u_hist,'LineWidth',2)
title('Control Force')
grid on
