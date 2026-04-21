clear; clc; close all;

%% =========================================================
%  REGULARIZED DEEPC (YALMIP, RECEDING HORIZON) FOR CART-POLE
%
%  Requirements:
%    - YALMIP installed
%    - A QP solver available through YALMIP
%      (quadprog / gurobi / mosek / osqp, etc.)
%
%  Offline data file required:
%    - deepc_cartpole_dataset.mat
%
%  Main formulation:
%
%    min_{g,u,x,sigma}
%        sum_{k=0}^{N-1} (x_k-r_k)'Qx(x_k-r_k) + u_k'Ru u_k
%        + lambda_g * ||g||_1
%        + lambda_sigma * ||sigma||_2^2
%
%    s.t.
%        Up*g        = u_ini
%        Xp*g        = x_ini              (or x_ini + sigma if enabled)
%        Uf*g        = u
%        Xf*g        = x
%        x_0         = current measured state
%        input/state constraints
%
%  Receding horizon:
%    - Solve DeePC QP
%    - Apply first input only
%    - Shift past window
%    - Repeat
%
%% =========================================================

%% ---------------------------------------------------------
% 1) Load offline dataset
%% ---------------------------------------------------------
load('deepc_cartpole_dataset.mat');

% Expecting at least:
% Ad, Bd, Cd, Dd
% Up, Uf, Xp, Xf
% Tini, Ts
% Npred or equivalent horizon

N = Npred; % from dataset
nx = size(Ad,1);
nu = size(Bd,2);
ny = size(Cd,1);

ncol = size(Up,2);      % number of columns in Hankel matrices

if ~exist('Xp','var') || ~exist('Xf','var')
    error('Dataset must contain state Hankel matrices Xp and Xf. Rerun deepc_data_collection.m.');
end

if size(Xp,2) ~= ncol || size(Xf,2) ~= ncol
    error('State Hankel matrices do not match input Hankel column count.');
end

nsigma = size(Xp,1);    % nx*Tini

fprintf('\n=== Loaded offline DeePC dataset ===\n');
fprintf('nx = %d, nu = %d, ny = %d\n', nx, nu, ny);
fprintf('Tini = %d, N = %d\n', Tini, N);
fprintf('Hankel columns = %d\n', ncol);

%% ---------------------------------------------------------
% 2) Online control design settings
%% ---------------------------------------------------------
% Stage cost on full state x = [cart position; cart velocity; pole angle; pole angular velocity]
Qx = diag([200, 10, 1000, 100]);
Ru = 0.05;

% Regularization
lambda_g = 10;
lambda_sigma = 1e8;
use_state_slack = true;

% Online input and state constraints
u_min = -9.5;
u_max =  9.5;

x_min = -0.40;
x_max =  0.40;

phi_min = -0.15;
phi_max =  0.15;

% Terminal pole-angle constraint on the last predicted state.
enforce_terminal_phi = true;
phi_terminal_ref = 0;

% Number of online simulation steps after warm start
Nsim = 300;

% Initial state
x0 = [0.0; 0.00; -0.1; 0.00];

% Constant state reference
r_stage = [0.0; 0.0; 0.0; 0.0];

fprintf('\n=== Online DeePC design ===\n');
fprintf('Qx = diag([%.2f, %.2f, %.2f, %.2f])\n', Qx(1,1), Qx(2,2), Qx(3,3), Qx(4,4));
fprintf('Ru = %.2f\n', Ru);
fprintf('lambda_g = %.2e\n', lambda_g);
fprintf('lambda_sigma = %.2e\n', lambda_sigma);
fprintf('use_state_slack = %d\n', use_state_slack);
fprintf('u in [%.2f, %.2f]\n', u_min, u_max);
fprintf('x in [%.2f, %.2f]\n', x_min, x_max);
fprintf('phi in [%.2f, %.2f]\n', phi_min, phi_max);
fprintf('terminal phi constraint enabled = %d\n', enforce_terminal_phi);

%% ---------------------------------------------------------
% 3) Build stacked cost matrices and reference
%% ---------------------------------------------------------
Qbar = kron(eye(N), Qx);
Rbar = kron(eye(N), Ru);

r_stack = repmat(r_stage, N, 1);

%% ---------------------------------------------------------
% 4) Define YALMIP decision variables
%% ---------------------------------------------------------
g = sdpvar(ncol,1);
sigma = sdpvar(nsigma,1);

u = sdpvar(nu*N,1);
x = sdpvar(nx*N,1);

% Parameters that change online
u_ini_par = sdpvar(nu*Tini,1);
x_ini_par = sdpvar(nx*Tini,1);
x_now_par = sdpvar(nx,1);
r_par     = sdpvar(nx*N,1);

%% ---------------------------------------------------------
% 5) DeePC constraints
%% ---------------------------------------------------------
constraints = [];

% Core DeePC behavioral constraints
constraints = [constraints, Up*g == u_ini_par];
if use_state_slack
    constraints = [constraints, Xp*g == x_ini_par + sigma];
else
    constraints = [constraints, Xp*g == x_ini_par];
    constraints = [constraints, sigma == zeros(nsigma,1)];
end
constraints = [constraints, Uf*g == u];
constraints = [constraints, Xf*g == x];
constraints = [constraints, x(1:nx) == x_now_par];

% Input bounds
constraints = [constraints, u_min <= u <= u_max];

% State bounds:
% x is stacked as [cart;cart_dot;phi;phi_dot] at each prediction step.
x_idx   = 1:nx:(nx*N); % indices of cart position in stacked x
phi_idx = 3:nx:(nx*N); % indices of pole angle in stacked x

constraints = [constraints, x_min   <= x(x_idx)   <= x_max];
constraints = [constraints, phi_min <= x(phi_idx) <= phi_max];

% Terminal constraint on pole angle at last predicted state
phi_terminal_idx = 3 + nx*(N-1);
constraints = [constraints, x(phi_terminal_idx) == phi_terminal_ref];

%% ---------------------------------------------------------
% 6) Objective
%% ---------------------------------------------------------
objective = (x - r_par)'*Qbar*(x - r_par) ...
          + u'*Rbar*u ...
          + lambda_g*norm(g,1) ...
          + lambda_sigma*(sigma'*sigma);

%% ---------------------------------------------------------
% 7) Solver settings
%% ---------------------------------------------------------
solver_name = 'osqp';

ops = sdpsettings( ...
    'solver', solver_name, ...
    'verbose', 0, ...
    'debug', 0);

% Build optimizer object for repeated receding-horizon solves
controller = optimizer(constraints, objective, ops, ...
    {u_ini_par, x_ini_par, x_now_par, r_par}, ...
    {u, x, g, sigma});

fprintf('\nYALMIP optimizer built with solver: %s\n', solver_name);

%% ---------------------------------------------------------
% 8) Warm start to initialize past window
%% ---------------------------------------------------------
x_warm = x0;

u_ini_hist = zeros(nu, Tini);
x_ini_hist = zeros(nx, Tini);

for k = 1:Tini
    u_w = 0;

    u_ini_hist(:,k) = u_w;
    x_ini_hist(:,k) = x_warm;

    x_warm = Ad*x_warm + Bd*u_w;
end

%% //TODO: CHECK IF THIS WARM START IS GOOD OR CAN BE IMPROVED.
u_ini = reshape(u_ini_hist, [], 1);
x_ini = reshape(x_ini_hist, [], 1);

%% ------------------------ ---------------------------------
% 9) Recursive DeePC closed-loop simulation
%% ------------------------ ---------------------------------
xDeepc = zeros(nx, Nsim+1);
yDeepc = zeros(ny, Nsim);
uDeepc = zeros(nu, Nsim);

gNorm = zeros(1, Nsim);
sigmaNorm = zeros(1, Nsim);
solver_status = zeros(1, Nsim);
solve_time_hist = zeros(1, Nsim);
step_time_hist = zeros(1, Nsim);

xDeepc(:,1) = x_warm;

for t = 1:Nsim

    step_timer = tic;

    % Solve DeePC with latest past window
    solve_timer = tic;
    sol = controller{{u_ini, x_ini, xDeepc(:,t), r_stack}};
    solve_time_hist(t) = toc(solve_timer);
    %% //TODO: ADD TERMINAL CONSTRAINTS AND FEASIBILITY CHECKS

    if isa(sol, 'cell') && numel(sol) == 4
        u_star = sol{1};
        x_star = sol{2};
        g_star = sol{3};
        sigma_star = sol{4};
        status_ok = true;
    else
        status_ok = false;
    end

    if ~status_ok || any(isnan(u_star)) || any(isnan(x_star))
        warning('DeePC optimization failed at step %d. Applying zero input.', t);

        u_apply = zeros(nu,1);
        u_apply = min(max(u_apply, u_min), u_max);

        g_star = zeros(ncol,1);
        sigma_star = zeros(nsigma,1);
        solver_status(t) = 0;
    else
        % Receding horizon: apply only the first input
        u_apply = u_star(1:nu);
        solver_status(t) = 1;
    end

    % Plant output
    y_now = Cd*xDeepc(:,t) + Dd*u_apply;

    % Log
    uDeepc(:,t) = u_apply;
    yDeepc(:,t) = y_now;
    gNorm(t) = norm(g_star,2);
    sigmaNorm(t) = norm(sigma_star,2);

    % Propagate plant
    xDeepc(:,t+1) = Ad*xDeepc(:,t) + Bd*u_apply;

    % Shift past window
    u_ini = [u_ini(nu+1:end); u_apply];
    x_ini = [x_ini(nx+1:end); xDeepc(:,t)];

    step_time_hist(t) = toc(step_timer);
end

%% ---------------------------------------------------------
% 10) Diagnostics
%% ---------------------------------------------------------
JDeepc = 0;

for k = 1:Nsim
    eD = xDeepc(:,k) - r_stage;

    JDeepc = JDeepc + eD'*Qx*eD + uDeepc(:,k)'*Ru*uDeepc(:,k);
end

deepc_input_viol = nnz(uDeepc < u_min | uDeepc > u_max);
deepc_x_viol     = nnz(yDeepc(1,:) < x_min | yDeepc(1,:) > x_max);
deepc_phi_viol   = nnz(yDeepc(2,:) < phi_min | yDeepc(2,:) > phi_max);

max_solve_time = max(solve_time_hist);
avg_solve_time = mean(solve_time_hist);
max_step_time = max(step_time_hist);
avg_step_time = mean(step_time_hist);
[~, worst_solve_step] = max(solve_time_hist);
[~, worst_step] = max(step_time_hist);

fprintf('\n=== Closed-loop DeePC diagnostics ===\n');
fprintf('Successful solves           = %d / %d\n', nnz(solver_status==1), Nsim);
fprintf('Input violations            = %d\n', deepc_input_viol);
fprintf('x violations                = %d\n', deepc_x_viol);
fprintf('phi violations              = %d\n', deepc_phi_viol);
fprintf('Average ||g||_2             = %.4e\n', mean(gNorm));
fprintf('Average ||sigma||_2         = %.4e\n', mean(sigmaNorm));
fprintf('DeePC cumulative stage cost = %.6f\n', JDeepc);

fprintf('\n=== DeePC real-time computation check ===\n');
fprintf('Sampling time Ts              = %.6f s (%.2f ms)\n', Ts, Ts*1000);
fprintf('Average optimization time     = %.6f s (%.2f ms)\n', avg_solve_time, avg_solve_time*1000);
fprintf('Worst-case optimization time  = %.6f s (%.2f ms)\n', max_solve_time, max_solve_time*1000);
fprintf('Worst optimization step       = %d\n', worst_solve_step);
fprintf('Average control-step time     = %.6f s (%.2f ms)\n', avg_step_time, avg_step_time*1000);
fprintf('Worst-case control-step time  = %.6f s (%.2f ms)\n', max_step_time, max_step_time*1000);
fprintf('Worst control-step index      = %d\n', worst_step);

if max_step_time < Ts
    fprintf('Result: feasible for real-time computation based on total control-step time.\n');
else
    fprintf('Result: not feasible for real-time computation based on total control-step time.\n');
end

%% ---------------------------------------------------------
% 11) Save results
%% ---------------------------------------------------------
results_file = 'deepc_yalmip_results_versionB.mat';

save(results_file, ...
    'xDeepc','yDeepc','uDeepc', ...
    'gNorm','sigmaNorm','solver_status', ...
    'solve_time_hist','step_time_hist', ...
    'max_solve_time','avg_solve_time', ...
    'max_step_time','avg_step_time', ...
    'worst_solve_step','worst_step', ...
    'Qx','Ru','lambda_g','lambda_sigma', ...
    'use_state_slack', ...
    'enforce_terminal_phi','phi_terminal_ref', ...
    'u_min','u_max','x_min','x_max','phi_min','phi_max', ...
    'Nsim','Ts','r_stage');

fprintf('\nResults saved to %s\n', results_file);

%% ---------------------------------------------------------
% 12) Plots
%% ---------------------------------------------------------
t = (0:Nsim-1)*Ts;
t_state = (0:Nsim)*Ts;

figure('Name','DeePC closed-loop response','Color','w');

subplot(3,1,1);
plot(t, yDeepc(1,:), 'LineWidth', 1.6); hold on;
yline(x_max, ':r'); yline(x_min, ':r');
grid on;
ylabel('x (m)');
title('Cart position');
legend('DeePC','Location','best');

subplot(3,1,2);
plot(t, yDeepc(2,:), 'LineWidth', 1.6); hold on;
yline(phi_max, ':r'); yline(phi_min, ':r');
grid on;
ylabel('\phi (rad)');
title('Pole angle');

subplot(3,1,3);
plot(t, uDeepc, 'LineWidth', 1.6); hold on;
yline(u_max, ':r'); yline(u_min, ':r');
grid on;
ylabel('u');
xlabel('Time (s)');
title('Control input');

figure('Name','DeePC internal variables','Color','w');

subplot(2,1,1);
plot(t, gNorm, 'LineWidth', 1.4);
grid on;
ylabel('||g||_2');
title('Coefficient norm');

subplot(2,1,2);
plot(t, sigmaNorm, 'LineWidth', 1.4);
grid on;
ylabel('||\sigma||_2');
xlabel('Time (s)');
title('Past-state slack norm');

figure('Name','DeePC computation time','Color','w');
plot(1:Nsim, step_time_hist*1000, 'LineWidth', 1.5); hold on;
plot(1:Nsim, solve_time_hist*1000, '--', 'LineWidth', 1.2);
yline(Ts*1000, ':r', 'LineWidth', 1.4);
grid on;
xlabel('Simulation step');
ylabel('Computation time (ms)');
title('DeePC computation time per control step');
legend('Total control-step time','Optimization time','Sampling deadline','Location','best');

figure('Name','DeePC state trajectory','Color','w');
plot(t_state, xDeepc', 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('states');
legend('x','x\_dot','phi','phi\_dot','Location','best');
title('DeePC state trajectory');
