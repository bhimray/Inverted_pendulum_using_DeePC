clear; clc; close all;

%% =========================================================
%  OUTPUT-BASED REGULARIZED DEEPC FOR CART-POLE
%
%  This script implements the DeePC formulation from input/output data:
%
%    min_{g,u,y,sigma_y}
%        sum_{k=0}^{N-1} (y_k-r_k)'Qy(y_k-r_k) + u_k'Ru u_k
%        + lambda_g * ||g||_1
%        + lambda_y * ||sigma_y||_2^2
%
%    s.t.
%        Up*g        = u_ini
%        Yp*g        = y_ini + sigma_y
%        Uf*g        = u
%        Yf*g        = y
%        input/output constraints
%
%  Important:
%    - The online DeePC optimizer below does not use Ad, Bd, Cd, or Dd.
%    - Ad and Bd are used only at the end as a stand-in plant simulator.
%      On hardware, that simulation line is replaced by the real plant and
%      sensor measurements.
%% =========================================================

%% ---------------------------------------------------------
% 1) Load offline input/output Hankel dataset
%% ---------------------------------------------------------
data = load('deepc_cartpole_dataset.mat');

required = {'Up','Uf','Yp','Yf','U','Y','Tini','Npred','Ts'};
for i = 1:numel(required)
    if ~isfield(data, required{i})
        error('Dataset is missing %s. Rerun deepc_data_collection.m.', required{i});
    end
end

Up = data.Up;
Uf = data.Uf;
Yp = data.Yp;
Yf = data.Yf;
U_data = data.U;
Y_data = data.Y;
Tini = data.Tini;
N = data.Npred;
Ts = data.Ts;

nu = size(Up,1) / Tini;
ny = size(Yp,1) / Tini;
ncol = size(Up,2);

if abs(nu - round(nu)) > eps || abs(ny - round(ny)) > eps
    error('Hankel dimensions are inconsistent with Tini.');
end
nu = round(nu);
ny = round(ny);

if size(Uf,1) ~= nu*N || size(Yf,1) ~= ny*N
    error('Future Hankel dimensions are inconsistent with Npred.');
end

if size(Yp,2) ~= ncol || size(Uf,2) ~= ncol || size(Yf,2) ~= ncol
    error('Input and output Hankel matrices must have the same number of columns.');
end

nsigma_y = ny*Tini;

fprintf('\n=== Loaded output DeePC dataset ===\n');
fprintf('nu = %d, ny = %d\n', nu, ny);
fprintf('Tini = %d, N = %d\n', Tini, N);
fprintf('Hankel columns = %d\n', ncol);

%% ---------------------------------------------------------
% 2) Online DeePC design settings
%% ---------------------------------------------------------
% Outputs are y = [cart position; pole angle].
Qy = diag([200, 1000]);
Ru = 0.05;

lambda_g = 10;
lambda_y = 1e8;
use_output_slack = true;

u_min = -9.5;
u_max =  9.5;

x_min = -0.40;
x_max =  0.40;

phi_min = -0.35;
phi_max =  0.35;

enforce_terminal_phi = false;
phi_terminal_ref = 0;

Nsim = 400;

% Output reference: [cart position; pole angle].
r_stage = [0.0; 0.0];
r_stack = repmat(r_stage, N, 1);

% Desired closed-loop initial condition for simulation:
% state = [cart position; cart velocity; pole angle; pole angular velocity].
x0_sim = [0.0; 0.0; 0.3; 0.0];
y0_sim = [x0_sim(1); x0_sim(3)];

fprintf('\n=== Online output DeePC design ===\n');
fprintf('Qy = diag([%.2f, %.2f])\n', Qy(1,1), Qy(2,2));
fprintf('Ru = %.2f\n', Ru);
fprintf('lambda_g = %.2e\n', lambda_g);
fprintf('lambda_y = %.2e\n', lambda_y);
fprintf('use_output_slack = %d\n', use_output_slack);
fprintf('u in [%.2f, %.2f]\n', u_min, u_max);
fprintf('cart output in [%.2f, %.2f]\n', x_min, x_max);
fprintf('pole angle output in [%.2f, %.2f]\n', phi_min, phi_max);
fprintf('terminal pole-angle constraint enabled = %d\n', enforce_terminal_phi);

%% ---------------------------------------------------------
% 3) Build stacked cost matrices
%% ---------------------------------------------------------
Qbar = kron(eye(N), Qy);
Rbar = kron(eye(N), Ru);

%% ---------------------------------------------------------
% 4) Define YALMIP decision variables
%% ---------------------------------------------------------
g = sdpvar(ncol,1);
sigma_y = sdpvar(nsigma_y,1);

u = sdpvar(nu*N,1);
y = sdpvar(ny*N,1);

u_ini_par = sdpvar(nu*Tini,1);
y_ini_par = sdpvar(ny*Tini,1);
r_par = sdpvar(ny*N,1);

%% ---------------------------------------------------------
% 5) Output-based DeePC constraints
%% ---------------------------------------------------------
constraints = [];

constraints = [constraints, Up*g == u_ini_par];
if use_output_slack
    constraints = [constraints, Yp*g == y_ini_par + sigma_y];
else
    constraints = [constraints, Yp*g == y_ini_par];
    constraints = [constraints, sigma_y == zeros(nsigma_y,1)];
end
constraints = [constraints, Uf*g == u];
constraints = [constraints, Yf*g == y];

constraints = [constraints, u_min <= u <= u_max];

cart_idx = 1:ny:(ny*N);
phi_idx = 2:ny:(ny*N);

constraints = [constraints, x_min <= y(cart_idx) <= x_max];
constraints = [constraints, phi_min <= y(phi_idx) <= phi_max];

if enforce_terminal_phi
    phi_terminal_idx = 2 + ny*(N-1);
    constraints = [constraints, y(phi_terminal_idx) == phi_terminal_ref];
end

%% ---------------------------------------------------------
% 6) Objective
%% ---------------------------------------------------------
objective = (y - r_par)'*Qbar*(y - r_par) ...
          + u'*Rbar*u ...
          + lambda_g*norm(g,1) ...
          + lambda_y*(sigma_y'*sigma_y);

%% ---------------------------------------------------------
% 7) Solver settings
%% ---------------------------------------------------------
solver_name = 'osqp';

ops = sdpsettings( ...
    'solver', solver_name, ...
    'verbose', 0, ...
    'debug', 0);

controller = optimizer(constraints, objective, ops, ...
    {u_ini_par, y_ini_par, r_par}, ...
    {u, y, g, sigma_y});

fprintf('\nYALMIP optimizer built with solver: %s\n', solver_name);
fprintf('Controller formulation uses only Up, Uf, Yp, and Yf.\n');

%% ---------------------------------------------------------
% 8) Closed-loop test setup
%% ---------------------------------------------------------
% The controller is data driven. This block is only the simulation
% environment used to test the computed input sequence.
if ~isfield(data, 'Ad') || ~isfield(data, 'Bd') || ~isfield(data, 'X')
    error(['Closed-loop simulation needs a plant or measured online data. ', ...
           'The DeePC controller itself does not need Ad/Bd.']);
end

Ad_sim = data.Ad;
Bd_sim = data.Bd;
nx_sim = size(Ad_sim,1);
output_state_idx = [1 3];   % measured outputs: cart position and pole angle

%% ---------------------------------------------------------
% 9) Initialize past window for the desired measured condition
%% ---------------------------------------------------------
% In hardware, u_ini and y_ini are the last Tini measured input/output
% samples. For simulation, build a dynamically consistent zero-input
% prehistory that ends exactly at the requested x0_sim.
u_ini_hist = zeros(nu, Tini);
y_ini_hist = zeros(ny, Tini);
x_hist = zeros(nx_sim, Tini);

x_hist(:,Tini) = x0_sim;
for k = Tini-1:-1:1
    x_hist(:,k) = Ad_sim \ x_hist(:,k+1);
end

for k = 1:Tini
    y_ini_hist(:,k) = x_hist(output_state_idx,k);
end

u_ini = reshape(u_ini_hist, [], 1);
y_ini = reshape(y_ini_hist, [], 1);

xDeepc = zeros(nx_sim, Nsim+1);
yDeepc = zeros(ny, Nsim);
uDeepc = zeros(nu, Nsim);

yPred = zeros(ny*N, Nsim);
gNorm = zeros(1, Nsim);
sigmaNorm = zeros(1, Nsim);
solver_status = zeros(1, Nsim);
solve_time_hist = zeros(1, Nsim);
step_time_hist = zeros(1, Nsim);

xDeepc(:,1) = x0_sim;

for t = 1:Nsim
    step_timer = tic;

    solve_timer = tic;
    sol = controller{{u_ini, y_ini, r_stack}};
    solve_time_hist(t) = toc(solve_timer);

    if isa(sol, 'cell') && numel(sol) == 4
        u_star = sol{1};
        y_star = sol{2};
        g_star = sol{3};
        sigma_star = sol{4};
        status_ok = true;
    else
        status_ok = false;
    end

    if ~status_ok || any(isnan(u_star)) || any(isnan(y_star))
        warning('DeePC optimization failed at step %d. Applying zero input.', t);

        u_apply = zeros(nu,1);
        y_star = nan(ny*N,1);
        g_star = zeros(ncol,1);
        sigma_star = zeros(nsigma_y,1);
        solver_status(t) = 0;
    else
        u_apply = u_star(1:nu);
        solver_status(t) = 1;
    end

    u_apply = min(max(u_apply, u_min), u_max);

    % Plant simulation only. Replace this with the real plant on hardware.
    y_meas = xDeepc(output_state_idx,t);
    x_next = Ad_sim*xDeepc(:,t) + Bd_sim*u_apply;
    y_next = x_next(output_state_idx);
    xDeepc(:,t+1) = x_next;

    uDeepc(:,t) = u_apply;
    yDeepc(:,t) = y_meas;
    yPred(:,t) = y_star;
    gNorm(t) = norm(g_star,2);
    sigmaNorm(t) = norm(sigma_star,2);

    u_ini = [u_ini(nu+1:end); u_apply];
    y_ini = [y_ini(ny+1:end); y_next];

    step_time_hist(t) = toc(step_timer);
end

%% ---------------------------------------------------------
% 10) Diagnostics
%% ---------------------------------------------------------
JDeepc = 0;

for k = 1:Nsim
    eD = yDeepc(:,k) - r_stage;
    JDeepc = JDeepc + eD'*Qy*eD + uDeepc(:,k)'*Ru*uDeepc(:,k);
end

deepc_input_viol = nnz(uDeepc < u_min | uDeepc > u_max);
deepc_x_viol = nnz(yDeepc(1,:) < x_min | yDeepc(1,:) > x_max);
deepc_phi_viol = nnz(yDeepc(2,:) < phi_min | yDeepc(2,:) > phi_max);

max_solve_time = max(solve_time_hist);
avg_solve_time = mean(solve_time_hist);
max_step_time = max(step_time_hist);
avg_step_time = mean(step_time_hist);
[~, worst_solve_step] = max(solve_time_hist);
[~, worst_step] = max(step_time_hist);

fprintf('\n=== Closed-loop output DeePC diagnostics ===\n');
fprintf('Successful solves           = %d / %d\n', nnz(solver_status==1), Nsim);
fprintf('Input violations            = %d\n', deepc_input_viol);
fprintf('cart output violations      = %d\n', deepc_x_viol);
fprintf('pole angle violations       = %d\n', deepc_phi_viol);
fprintf('Average ||g||_2             = %.4e\n', mean(gNorm));
fprintf('Average ||sigma_y||_2       = %.4e\n', mean(sigmaNorm));
fprintf('DeePC cumulative output cost = %.6f\n', JDeepc);

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
    'xDeepc','yDeepc','uDeepc','yPred', ...
    'gNorm','sigmaNorm','solver_status', ...
    'solve_time_hist','step_time_hist', ...
    'max_solve_time','avg_solve_time', ...
    'max_step_time','avg_step_time', ...
    'worst_solve_step','worst_step', ...
    'Qy','Ru','lambda_g','lambda_y', ...
    'use_output_slack', ...
    'enforce_terminal_phi','phi_terminal_ref', ...
    'u_min','u_max','x_min','x_max','phi_min','phi_max', ...
    'Nsim','Ts','r_stage');

fprintf('\nResults saved to %s\n', results_file);

%% ---------------------------------------------------------
% 12) Plots
%% ---------------------------------------------------------
t = (0:Nsim-1)*Ts;
t_state = (0:Nsim)*Ts;

plot_dir = 'deepc_plots';
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

fig_response = figure('Name','Output DeePC closed-loop response','Color','w');

subplot(3,1,1);
plot(t, yDeepc(1,:), 'LineWidth', 1.6); hold on;
yline(x_max, ':r'); yline(x_min, ':r');
grid on;
ylabel('x (m)');
title('Cart position output');
legend('DeePC','Location','best');

subplot(3,1,2);
plot(t, yDeepc(2,:), 'LineWidth', 1.6); hold on;
yline(phi_max, ':r'); yline(phi_min, ':r');
grid on;
ylabel('\phi (rad)');
title('Pole angle output');

subplot(3,1,3);
plot(t, uDeepc, 'LineWidth', 1.6); hold on;
yline(u_max, ':r'); yline(u_min, ':r');
grid on;
ylabel('u');
xlabel('Time (s)');
title('Control input');

fig_internal = figure('Name','Output DeePC internal variables','Color','w');

subplot(2,1,1);
plot(t, gNorm, 'LineWidth', 1.4);
grid on;
ylabel('||g||_2');
title('Coefficient norm');

subplot(2,1,2);
plot(t, sigmaNorm, 'LineWidth', 1.4);
grid on;
ylabel('||\sigma_y||_2');
xlabel('Time (s)');
title('Past-output slack norm');

fig_timing = figure('Name','Output DeePC computation time','Color','w');
plot(1:Nsim, step_time_hist*1000, 'LineWidth', 1.5); hold on;
plot(1:Nsim, solve_time_hist*1000, '--', 'LineWidth', 1.2);
yline(Ts*1000, ':r', 'LineWidth', 1.4);
grid on;
xlabel('Simulation step');
ylabel('Computation time (ms)');
title('DeePC computation time per control step');
legend('Total control-step time','Optimization time','Sampling deadline','Location','best');

fig_state = figure('Name','Plant state trajectory used for simulation only','Color','w');
plot(t_state, xDeepc', 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('states');
legend('x','x\_dot','phi','phi\_dot','Location','best');
title('Simulated plant state trajectory');

plot_files = {
    fig_response, 'deepc_closed_loop_response';
    fig_internal, 'deepc_internal_variables';
    fig_timing,   'deepc_computation_time';
    fig_state,    'deepc_state_trajectory'
};

for i = 1:size(plot_files, 1)
    fig = plot_files{i, 1};
    name = plot_files{i, 2};

    savefig(fig, fullfile(plot_dir, [name '.fig']));
    exportgraphics(fig, fullfile(plot_dir, [name '.png']), 'Resolution', 300);
end

fprintf('\nPlots saved to %s\n', plot_dir);
