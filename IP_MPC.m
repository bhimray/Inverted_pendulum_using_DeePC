clear
clc
close all

%% Physical parameters (cart-pole)
M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.81;
l = 0.3;

Ts = 0.01; % sampling time for MPC

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

Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;

nx = size(Ad,1); % states = no. of rows of A
nu = size(Bd,2); % inputs = no. of columns of B
ny = size(Cd,1); % outputs = no. of rows of C

%% MPC parameters

N = 5; % prediction horizon

Q = diag([50 50]);
R = 0.01;

Q = kron(eye(N),Q);
R = kron(eye(N),R);

umin = -10;
umax = 10;

%% Build prediction matrices

Gamma = zeros(ny*N,nx);
Theta = zeros(ny*N,nu*N);

for i=1:N

    Gamma((i-1)*ny+1:i*ny,:) = Cd*(Ad^i);

    for j=1:i

        Theta((i-1)*ny+1:i*ny,(j-1)*nu+1:j*nu) = Cd*(Ad^(i-j))*Bd;

    end

end

%% Simulation

Tsim = 500; % number of simulation steps
t = (0:Tsim-1) * Ts;

x = [0.05;0;0.15;0];

x_hist = zeros(nx,Tsim);
u_hist = zeros(nu,Tsim);
y_hist = zeros(ny,Tsim);
solve_time_hist = zeros(1,Tsim);
step_time_hist = zeros(1,Tsim);

ref = zeros(ny*N,1);

ops = sdpsettings('solver','quadprog','verbose',1);

for k=1:Tsim

    step_timer = tic;

    u = sdpvar(nu*N,1); % solve for this

    y = Gamma*x + Theta*u; % start from current state and predict future outputs

    con = [umin <= u <= umax]; % input constraints

    obj = (y-ref)'*Q*(y-ref) + u'*R*u; % cost

    solve_timer = tic;
    sol = optimize(con,obj, ops); % solve optimization problem
    solve_time_hist(k) = toc(solve_timer);

    if sol.problem ~= 0
        disp('Solver failed!');
    end

    u_opt = value(u); % optimal control sequence

    u_apply = u_opt(1); % apply only the first control input

    x = Ad*x + Bd*u_apply; % update state based on applied control

    y = Cd*x; 

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

fprintf('\nCondensed MPC real-time check:\n');
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
title('Cart Position (MPC)')
xlabel('Time (s)')
ylabel('x (m)')
grid on

subplot(3,1,2)
plot(t,y_hist(2,:),'LineWidth',2)
title('Pendulum Angle (MPC)')
xlabel('Time (s)')
ylabel('\phi (rad)')
grid on

subplot(3,1,3)
plot(t,u_hist,'LineWidth',2)
title('Control Force')
xlabel('Time (s)')
ylabel('u (N)')
grid on
