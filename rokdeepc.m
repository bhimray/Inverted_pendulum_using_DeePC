% RoKDeePC for cart-pole using OUTPUT-BASED formulation
% aligned with the structure of:
%   Huang, Lygeros, Dorfler,
%   "Robust and Kernelized Data-Enabled Predictive Control for Nonlinear Systems"
%
% USER-SELECTED DESIGN CHOICES FOR THIS VERSION
%   1) outputs = [x ; phi]
%   2) output-based formulation as in the paper
%   3) exponential kernel
%   4) constrained g-subproblem (no closed-form g update)
%
% -------------------------------------------------------------------------
% IMPORTANT
% -------------------------------------------------------------------------
% This is NOT the paper's academic SISO example. This is a cart-pole
% adaptation that preserves the paper's STRUCTURE:
%
%   - offline data of inputs/outputs
%   - trajectory columns z_j = col(U_P(:,j), Y_P(:,j), U_F(:,j))
%   - kernel predictor built from K and k(z)
%   - quadratic RoKDeePC cost c_q(u,g)
%   - projected gradient step on u
%   - constrained convex subproblem in g solved numerically
%   - receding-horizon implementation
%
% Since you requested the constrained g-subproblem, the closed-form update
% for g is NOT used here. Instead, at each projected-gradient iteration on u,
% we solve the convex g-subproblem directly as a linearly constrained QP.
%
% -------------------------------------------------------------------------
% WHAT THIS SCRIPT EXPECTS IN THE DATASET
% -------------------------------------------------------------------------
% A MAT file named: deepc_cartpole_dataset.mat
%
% Required variables:
%   Ad, Bd, Cd, Dd           % discrete-time linearized model matrices
%   Ts                       % sampling time
%   Tini                     % initial trajectory length
%   Npred                    % prediction horizon to use as default N
%   u_data                   % offline input sequence, size [T,1] or [1,T]
%   y_data                   % offline output sequence, size [T,2] or [2,T]
%
% OPTIONAL but strongly recommended:
%   x_data                   % offline state sequence, only for validation/plots
%
% The outputs must be exactly:
%   y = [x ; phi]
%
% If your MAT file currently does not contain u_data and y_data, then you must
% regenerate offline data in output form. Your earlier linear/state DeePC code
% is built around Xp/Xf and state trajectories, which is a different setup.
%
% -------------------------------------------------------------------------
% NUMERICAL METHOD IMPLEMENTED HERE
% -------------------------------------------------------------------------
% Outer loop (nonconvex in u): projected gradient descent
%
%   u^{i+1} = Proj_U( u^i - alpha * dJ/du )
%
% Inner loop (convex in g for fixed u): solve constrained g-subproblem
%
%   min_g  ||YF g - r||_Q^2 + lambda_g ||g||_2^2
%        + lambda_k' ||(K + gamma I) g - k(z)||_2^2
%
%   s.t.   y_min <= YF g <= y_max    (componentwise over horizon)
%
% where
%   z = col(u_ini, y_ini, u)
%
% This preserves the paper's output-based structure and your requested use of
% a constrained g-subproblem.
%
% -------------------------------------------------------------------------
clear; clc; close all;

%% ========================================================================
% 1) LOAD DATASET
% ========================================================================

datafile = 'deepc_cartpole_dataset.mat';
assert(exist(datafile, 'file') == 2, 'Dataset file %s not found.', datafile);
load(datafile);

% Accept both the RoKDeePC alias names and the dataset names produced by
% deepc_data_collection.m.
if exist('u_data', 'var') ~= 1 && exist('U', 'var') == 1
    u_data = U;
end
if exist('y_data', 'var') ~= 1 && exist('Y', 'var') == 1
    y_data = Y;
end
if exist('x_data', 'var') ~= 1 && exist('X', 'var') == 1
    x_data = X;
end

required_vars = {'Ad','Bd','Cd','Dd','Ts','Tini','Npred','u_data','y_data'};
for k = 1:numel(required_vars)
    assert(exist(required_vars{k}, 'var') == 1, 'Required variable missing: %s', required_vars{k});
end

u_data = u_data(:);
if size(y_data,1) == 2 && size(y_data,2) ~= 2
    y_data = y_data.';
end
assert(size(y_data,2) == 2, 'y_data must contain exactly two outputs [x, phi].');
assert(size(y_data,1) == numel(u_data), ...
       'u_data and y_data must have the same number of samples.');

nx = size(Ad,1);
nu = size(Bd,2);
ny = size(Cd,1);
assert(nu == 1, 'This implementation assumes single-input cart-pole.');
assert(size(y_data,2) == 2, 'This implementation is configured for outputs [x, phi].');

fprintf('\n==============================================================\n');
fprintf('RoKDeePC for cart-pole (output-based, constrained g-subproblem)\n');
fprintf('Dataset: %s\n', datafile);
fprintf('nx=%d, nu=%d, ny(model)=%d, y_data outputs=%d\n', nx, nu, ny, size(y_data,2));
fprintf('Tini=%d, Npred=%d, Ts=%.6f s\n', Tini, Npred, Ts);
fprintf('==============================================================\n');

%% ========================================================================
% 2) USER / CONTROLLER SETTINGS
% ========================================================================

cfg = struct();

% Horizon
cfg.N = Npred;

% Kernel settings: EXPONENTIAL kernel requested
cfg.kernel_type = 'exponential';
cfg.exponential_den = 0.2;   % same style as the paper's exponential kernel
cfg.kernel_exponent_limit = 50; % prevents exp overflow in finite precision
cfg.gamma = 1e-2;

% Quadratic reformulation weights
cfg.lambda_g = 1.0;
cfg.lambda_k_prime = 1e6;

% Outer projected-gradient settings on u
cfg.alpha = 1e-2;
cfg.max_outer_iter = 60;
cfg.outer_cost_tol = 1e-7;
cfg.outer_u_tol = 1e-7;
cfg.grad_eps = 1e-6;         % finite-difference step for dJ/du
cfg.use_backtracking = true;
cfg.bt_beta = 0.5;
cfg.bt_c1 = 1e-4;

% Input constraints over horizon
cfg.use_input_bounds = true;
cfg.u_min = -9.5;
cfg.u_max =  9.5;

% Output constraints over horizon, for y = [x; phi]
cfg.use_output_bounds = true;
cfg.x_min = -0.40;
cfg.x_max =  0.40;
cfg.phi_min = -0.15;
cfg.phi_max =  0.15;

% Closed-loop run length
cfg.Nsim = 250;

% Measured output noise used online (set to zero if not desired)
cfg.online_meas_noise_std = 0.0;

% References
cfg.r_stage = [0.0; 0.0];  % [x_ref; phi_ref]

% Solver for constrained g-subproblem
cfg.g_solver = 'quadprog';          % default fast path
cfg.enable_yalmip_osqp_fallback = true;
cfg.debug = false;

fprintf('\nController settings:\n');
fprintf('Kernel          : exponential\n');
fprintf('gamma           : %.3e\n', cfg.gamma);
fprintf('lambda_g        : %.3e\n', cfg.lambda_g);
fprintf('lambda_k''       : %.3e\n', cfg.lambda_k_prime);
fprintf('u bounds        : [%.2f, %.2f]\n', cfg.u_min, cfg.u_max);
fprintf('x bounds        : [%.2f, %.2f]\n', cfg.x_min, cfg.x_max);
fprintf('phi bounds      : [%.2f, %.2f]\n', cfg.phi_min, cfg.phi_max);

%% ========================================================================
% 3) BUILD OUTPUT-BASED TRAJECTORY SAMPLE MATRICES
% ========================================================================
%
% For each sample column j:
%   z_j = col( U_P(:,j), Y_P(:,j), U_F(:,j) )
%
% with outputs Y = [x; phi]
%
% Dimensions:
%   U_P : (nu*Tini) x Hc
%   Y_P : (ny_out*Tini) x Hc, ny_out = 2 here
%   U_F : (nu*N) x Hc
%   Y_F : (ny_out*N) x Hc
%
% We use y_data directly, not Cd*x unless you explicitly want the offline
% output reconstructed from states.

ny_out = size(y_data,2);
T = length(u_data);
L = Tini + cfg.N;
Hc = T - L + 1;
assert(Hc > 0, 'Offline data length is too short for Tini+N.');

UP = zeros(nu*Tini, Hc);
YP = zeros(ny_out*Tini, Hc);
UF = zeros(nu*cfg.N, Hc);
YF = zeros(ny_out*cfg.N, Hc);

for j = 1:Hc
    idx = j:(j+L-1);
    u_seg = u_data(idx, :);
    y_seg = y_data(idx, :);

    UP(:,j) = reshape(u_seg(1:Tini,:).', [], 1);
    YP(:,j) = reshape(y_seg(1:Tini,:).', [], 1);
    UF(:,j) = reshape(u_seg(Tini+1:end,:).', [], 1);
    YF(:,j) = reshape(y_seg(Tini+1:end,:).', [], 1);
end

Zcols = [UP; YP; UF];
nz_traj = size(Zcols,1);

fprintf('\nBuilt output-based trajectory data:\n');
fprintf('Hc = %d sample columns\n', Hc);
fprintf('dim(z_j) = %d\n', nz_traj);

%% ========================================================================
% 4) BUILD EXPONENTIAL KERNEL GRAM MATRIX
% ========================================================================
%
% K_ij = exp( z_i'' z_j / den )

K = zeros(Hc, Hc);
num_kernel_clipped = 0;
for i = 1:Hc
    zi = Zcols(:,i);
    for j = i:Hc
        zj = Zcols(:,j);
        [kernel_arg, was_clipped] = bounded_kernel_exponent((zi' * zj) / cfg.exponential_den, cfg);
        kij = exp(kernel_arg);
        num_kernel_clipped = num_kernel_clipped + was_clipped;
        K(i,j) = kij;
        K(j,i) = kij;
    end
end
Kgamma = K + cfg.gamma * eye(Hc);

fprintf('Kernel Gram matrix built: %d x %d\n', size(K,1), size(K,2));
fprintf('Kernel exponent clips: %d\n', num_kernel_clipped);

%% ========================================================================
% 5) PRECOMPUTE COST WEIGHTS / REFERENCE STACK
% ========================================================================

Q_stage = diag([200, 1000]);   % heavier penalty on phi tracking
Qbar = kron(eye(cfg.N), Q_stage);

% small effort + move suppression
R_stage = 0.05;
Rbar = kron(eye(cfg.N), R_stage);
Ddu = diff_operator(cfg.N);
Sdu = 1.0 * (Ddu' * Ddu);

% Bundle data for solvers
pre = struct();
pre.UP = UP;
pre.YP = YP;
pre.UF = UF;
pre.YF = YF;
pre.Zcols = Zcols;
pre.K = K;
pre.Kgamma = Kgamma;
pre.Qbar = Qbar;
pre.Rbar = Rbar;
pre.Sdu = Sdu;
pre.cfg = cfg;
pre.ny_out = ny_out;
pre.nu = nu;
pre.Hc = Hc;
pre.g_qp = precompute_g_qp_constants(pre);

%% ========================================================================
% 6) WARM START PAST WINDOW FROM INITIAL SYSTEM STATE
% ========================================================================

x0 = [0.0; 0.0; -0.10; 0.0];

x_now = x0;
u_ini_hist = zeros(nu, Tini);
y_ini_hist = zeros(ny_out, Tini);

for k = 1:Tini
    u_w = 0.0;
    y_w = [x_now(1); x_now(3)];

    u_ini_hist(:,k) = u_w;
    y_ini_hist(:,k) = y_w;

    x_now = Ad * x_now + Bd * u_w;
end

u_ini = reshape(u_ini_hist, [], 1);
y_ini = reshape(y_ini_hist, [], 1);

%% ========================================================================
% 7) RECeding-HORIZON CLOSED-LOOP SIMULATION
% ========================================================================

xCL = zeros(nx, cfg.Nsim+1);
yCL = zeros(ny_out, cfg.Nsim);
uCL = zeros(nu, cfg.Nsim);

outer_iters = zeros(1, cfg.Nsim);
solve_time = zeros(1, cfg.Nsim);
g_norm = zeros(1, cfg.Nsim);
kernel_residual = zeros(1, cfg.Nsim);
track_residual = zeros(1, cfg.Nsim);
g_subproblem_status = strings(1, cfg.Nsim);
g_solver_used = strings(1, cfg.Nsim);
quadprog_failures = zeros(1, cfg.Nsim);
yalmip_fallback_calls = zeros(1, cfg.Nsim);

xCL(:,1) = x_now;

% initial guess for u horizon
u_guess = zeros(cfg.N,1);

for t = 1:cfg.Nsim
    step_timer = tic;

    r_stack = repmat(cfg.r_stage, cfg.N, 1);

    % Solve outer nonconvex problem over u with constrained g-subproblem
    sol = solve_rokdeepc_cartpole(u_ini, y_ini, r_stack, u_guess, pre);

    solve_time(t) = toc(step_timer);
    outer_iters(t) = sol.outer_iter;
    g_norm(t) = norm(sol.g, 2);
    kernel_residual(t) = norm(pre.Kgamma * sol.g - sol.kz, 2);
    track_residual(t) = norm(pre.YF * sol.g - r_stack, 2);
    g_subproblem_status(t) = sol.g_status;
    g_solver_used(t) = sol.g_solver_used;
    quadprog_failures(t) = sol.quadprog_failures;
    yalmip_fallback_calls(t) = sol.yalmip_fallback_calls;

    u_apply = sol.u(1);
    uCL(:,t) = u_apply;

    % measured output y = [x; phi]
    y_true = [xCL(1,t); xCL(3,t)];
    y_meas = y_true + cfg.online_meas_noise_std * randn(2,1);
    yCL(:,t) = y_meas;

    % plant propagation
    xCL(:,t+1) = Ad * xCL(:,t) + Bd * u_apply;

    % shift past window using latest applied input and measured output
    u_ini = [u_ini(nu+1:end); u_apply];
    y_ini = [y_ini(ny_out+1:end); y_meas];

    % shift warm start
    u_guess = [sol.u(2:end); sol.u(end)];

    if cfg.debug
        fprintf(['step %03d | outer=%d | solver=%s | status=%s | ' ...
                 'quadprog_failures=%d | yalmip_fallback_calls=%d | solve=%.4fs\n'], ...
                t, sol.outer_iter, sol.g_solver_used, sol.g_status, ...
                sol.quadprog_failures, sol.yalmip_fallback_calls, solve_time(t));
    end
end

%% ========================================================================
% 8) DIAGNOSTICS
% ========================================================================

Jcl = 0;
for k = 1:cfg.Nsim
    e = yCL(:,k) - cfg.r_stage;
    Jcl = Jcl + e' * Q_stage * e + uCL(:,k)' * R_stage * uCL(:,k);
end

x_viol = nnz(yCL(1,:) < cfg.x_min | yCL(1,:) > cfg.x_max);
phi_viol = nnz(yCL(2,:) < cfg.phi_min | yCL(2,:) > cfg.phi_max);
u_viol = nnz(uCL < cfg.u_min | uCL > cfg.u_max);

fprintf('\n================== CLOSED-LOOP DIAGNOSTICS ==================\n');
fprintf('Average solve time          = %.6f s\n', mean(solve_time));
fprintf('Worst solve time            = %.6f s\n', max(solve_time));
fprintf('Average outer iterations    = %.2f\n', mean(outer_iters));
fprintf('Mean ||g||_2                = %.4e\n', mean(g_norm));
fprintf('Mean kernel residual        = %.4e\n', mean(kernel_residual));
fprintf('Mean tracking residual      = %.4e\n', mean(track_residual));
fprintf('g solver used               = %s\n', char(strjoin(unique(g_solver_used), ', ')));
fprintf('Total quadprog failures     = %d\n', sum(quadprog_failures));
fprintf('Total YALMIP fallback calls = %d\n', sum(yalmip_fallback_calls));
fprintf('Input violations            = %d\n', u_viol);
fprintf('x violations                = %d\n', x_viol);
fprintf('phi violations              = %d\n', phi_viol);
fprintf('Closed-loop stage cost      = %.6f\n', Jcl);
fprintf('============================================================\n');

%% ========================================================================
% 9) PLOTS
% ========================================================================

t = (0:cfg.Nsim-1) * Ts;
t_state = (0:cfg.Nsim) * Ts;

figure('Color','w','Name','RoKDeePC outputs');
subplot(3,1,1);
plot(t, yCL(1,:), 'LineWidth',1.5); hold on;
yline(cfg.x_min, ':r'); yline(cfg.x_max, ':r');
grid on; ylabel('x'); title('Cart position');

subplot(3,1,2);
plot(t, yCL(2,:), 'LineWidth',1.5); hold on;
yline(cfg.phi_min, ':r'); yline(cfg.phi_max, ':r');
grid on; ylabel('\phi'); title('Pole angle');

subplot(3,1,3);
plot(t, uCL, 'LineWidth',1.5); hold on;
yline(cfg.u_min, ':r'); yline(cfg.u_max, ':r');
grid on; ylabel('u'); xlabel('Time [s]'); title('Input');

figure('Color','w','Name','RoKDeePC states');
plot(t_state, xCL', 'LineWidth',1.2);
grid on; xlabel('Time [s]'); ylabel('state');
legend('x','xdot','phi','phidot','Location','best');

figure('Color','w','Name','RoKDeePC internals');
subplot(3,1,1);
plot(t, g_norm, 'LineWidth',1.3); grid on; ylabel('||g||_2');
subplot(3,1,2);
plot(t, kernel_residual, 'LineWidth',1.3); grid on; ylabel('kernel res');
subplot(3,1,3);
plot(t, track_residual, 'LineWidth',1.3); grid on; ylabel('track res'); xlabel('Time [s]');

figure('Color','w','Name','RoKDeePC solve time');
plot(t, 1e3*solve_time, 'LineWidth',1.4);
grid on; xlabel('Time [s]'); ylabel('ms'); title('Per-step solve time');

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function sol = solve_rokdeepc_cartpole(u_ini, y_ini, r_stack, u0, pre)
% Outer projected-gradient solver over u.
% For each candidate u, it solves the constrained convex g-subproblem.

    cfg = pre.cfg;
    u = u0(:);
    cost_prev = inf;
    solver_stats = init_solver_stats();

    for it = 1:cfg.max_outer_iter
        [J, ~, ~, ~, g_info] = total_cost_given_u(u_ini, y_ini, u, r_stack, pre);
        solver_stats = accumulate_solver_stats(solver_stats, g_info);

        [grad, grad_stats] = finite_difference_grad_u(u_ini, y_ini, u, r_stack, pre);
        solver_stats = accumulate_solver_stats(solver_stats, grad_stats);

        if abs(cost_prev - J) <= cfg.outer_cost_tol
            break;
        end
        if norm(grad,2) <= 1e-8
            break;
        end

        alpha = cfg.alpha;
        if cfg.use_backtracking
            accepted = false;
            for bt = 1:20
                u_trial = project_input_horizon(u - alpha*grad, cfg);
                [J_trial, ~, ~, ~, trial_info] = total_cost_given_u(u_ini, y_ini, u_trial, r_stack, pre);
                solver_stats = accumulate_solver_stats(solver_stats, trial_info);
                if J_trial <= J - cfg.bt_c1 * alpha * (grad' * grad)
                    u_new = u_trial;
                    accepted = true;
                    break;
                else
                    alpha = alpha * cfg.bt_beta;
                end
            end
            if ~accepted
                u_new = project_input_horizon(u - alpha*grad, cfg);
            end
        else
            u_new = project_input_horizon(u - alpha*grad, cfg);
        end

        if norm(u_new - u, 2) <= cfg.outer_u_tol
            u = u_new;
            break;
        end

        u = u_new;
        cost_prev = J;
    end

    [J, g, kz, g_status, g_info] = total_cost_given_u(u_ini, y_ini, u, r_stack, pre);
    solver_stats = accumulate_solver_stats(solver_stats, g_info);

    sol = struct();
    sol.u = u;
    sol.g = g;
    sol.kz = kz;
    sol.cost = J;
    sol.outer_iter = it;
    sol.g_status = g_status;
    sol.g_solver_used = g_info.solver_used;
    sol.g_solver_status = g_status;
    sol.g_solver_output = g_info.solver_output;
    sol.qp_info = g_info.solver_output;
    sol.quadprog_failures = solver_stats.quadprog_failures;
    sol.yalmip_fallback_calls = solver_stats.yalmip_fallback_calls;
end

function [J, g, kz, g_status, g_info] = total_cost_given_u(u_ini, y_ini, u, r_stack, pre)
% Evaluate the full objective for fixed u by solving the constrained convex
% g-subproblem.

    cfg = pre.cfg;

    z = [u_ini; y_ini; u];
    kz = kernel_vector_exponential(z, pre.Zcols, cfg);

    [g, g_status, g_info] = solve_g_subproblem_constrained(kz, r_stack, pre);

    ypred = pre.YF * g;
    kernel_res = pre.Kgamma * g - kz;

    Ju = control_cost(u, pre.Rbar, pre.Sdu);
    J = Ju + (ypred - r_stack)' * pre.Qbar * (ypred - r_stack) ...
           + cfg.lambda_g * (g' * g) ...
           + cfg.lambda_k_prime * (kernel_res' * kernel_res);
end

function [g, status_text, solver_info] = solve_g_subproblem_constrained(kz, r_stack, pre)
% Constrained convex g-subproblem for fixed u:
%
%   min_g (YF g-r)'Q(YF g-r) + lambda_g g'g
%        + lambda_k' (Kgamma g-kz)'(Kgamma g-kz)
%
%   s.t. y_min <= YF g <= y_max
%
% This is a smooth convex quadratic problem with linear inequality
% constraints. It is solved in standard QP form.

    cfg = pre.cfg;
    qp = build_g_qp(kz, r_stack, pre);
    solver_info = init_g_solver_info();

    switch lower(cfg.g_solver)
        case 'quadprog'
            [g, status_text, qp_info] = solve_g_subproblem_quadprog(kz, r_stack, pre);
            solver_info.solver_used = 'quadprog';
            solver_info.solver_status = status_text;
            solver_info.solver_output = qp_info;
            solver_info.quadprog_failures = qp_info.num_failures;

            if ~qp_info.success
                if cfg.enable_yalmip_osqp_fallback
                    solver_info.yalmip_fallback_calls = solver_info.yalmip_fallback_calls + 1;
                    [g, status_text, yalmip_info] = solve_g_subproblem_yalmip_osqp(kz, r_stack, pre);
                    solver_info.solver_used = 'yalmip-osqp';
                    solver_info.solver_status = status_text;
                    solver_info.solver_output = yalmip_info;
                else
                    error('quadprog failed for g-subproblem. %s. Output: %s', ...
                          status_text, solver_output_to_text(qp_info));
                end
            end

        case 'yalmip-osqp'
            [g, status_text, yalmip_info] = solve_g_subproblem_yalmip_osqp(kz, r_stack, pre);
            solver_info.solver_used = 'yalmip-osqp';
            solver_info.solver_status = status_text;
            solver_info.solver_output = yalmip_info;

        otherwise
            error('Unsupported g solver: %s', cfg.g_solver);
    end

    [valid, violation, validation_message] = validate_g_qp_solution(g, qp);
    solver_info.max_constraint_violation = violation;
    solver_info.validation_message = validation_message;
    if ~valid
        error('g-subproblem solution failed validation: %s', validation_message);
    end
end

function qp_const = precompute_g_qp_constants(pre)
% Precompute the fixed pieces of 0.5*g'*H*g + f'*g subject to Aineq*g <= bineq.

    cfg = pre.cfg;
    Hc = pre.Hc;

    H = 2 * (pre.YF' * pre.Qbar * pre.YF ...
             + cfg.lambda_g * eye(Hc) ...
             + cfg.lambda_k_prime * (pre.Kgamma' * pre.Kgamma));

    H = 0.5 * (H + H');
    H = H + 1e-10 * eye(size(H));

    if ~all(isfinite(H(:)))
        error(['QP Hessian contains NaN or Inf before scaling. Check the kernel ' ...
               'exponent limit and dataset magnitude.']);
    end

    objective_scale = max([1; abs(H(:))]);
    H = H / objective_scale;
    H = 0.5 * (H + H');

    y_lower = repmat([cfg.x_min; cfg.phi_min], cfg.N, 1);
    y_upper = repmat([cfg.x_max; cfg.phi_max], cfg.N, 1);

    qp_const = struct();
    qp_const.H = H;
    qp_const.Aineq = [pre.YF; -pre.YF];
    qp_const.bineq = [y_upper; -y_lower];
    qp_const.objective_scale = objective_scale;

    if ~all(isfinite(qp_const.H(:))) || ~all(isfinite(qp_const.Aineq(:))) || ...
       ~all(isfinite(qp_const.bineq(:)))
        error('Constant QP data contains NaN or Inf after scaling.');
    end
end

function qp = build_g_qp(kz, r_stack, pre)
% Build the full QP for the current kernel vector.

    cfg = pre.cfg;
    f = -2 * (pre.YF' * pre.Qbar * r_stack ...
              + cfg.lambda_k_prime * (pre.Kgamma' * kz));
    f = f / pre.g_qp.objective_scale;

    qp = pre.g_qp;
    qp.f = f;

    if ~all(isfinite(qp.f(:)))
        error(['QP linear term contains NaN or Inf. Check the online kernel ' ...
               'vector and exponent limit.']);
    end
end

function [g, status_text, qp_info] = solve_g_subproblem_quadprog(kz, r_stack, pre)
% Default constrained g-QP solver.

    qp = build_g_qp(kz, r_stack, pre);
    H = qp.H;
    f = qp.f;
    Aineq = qp.Aineq;
    bineq = qp.bineq;

    qp_info = struct();
    qp_info.success = false;
    qp_info.exitflag = NaN;
    qp_info.output = [];
    qp_info.attempts = {};
    qp_info.num_failures = 0;

    try
        opts = optimoptions('quadprog', ...
            'Algorithm','interior-point-convex', ...
            'Display','off', ...
            'OptimalityTolerance',1e-9, ...
            'ConstraintTolerance',1e-9, ...
            'StepTolerance',1e-12);
    catch ME
        g = [];
        qp_info.output = ME;
        qp_info.num_failures = qp_info.num_failures + 1;
        status_text = sprintf('quadprog options failed: %s', ME.message);
        return;
    end

    try
        H_rcond = rcond(H);
        if isfinite(H_rcond) && H_rcond > 1e-14
            g0 = -(H \ f);
        else
            g0 = [];
        end

        if ~isempty(g0) && all(isfinite(g0))
            [g, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], [], [], g0, opts);
            qp_info.attempts{end+1} = struct('warm_start',true, ...
                                             'exitflag',exitflag, ...
                                             'output',output);
            if exitflag > 0
                qp_info.success = true;
                qp_info.exitflag = exitflag;
                qp_info.output = output;
                status_text = sprintf('quadprog exitflag=%d', exitflag);
                return;
            end
            qp_info.num_failures = qp_info.num_failures + 1;
        end
    catch ME
        qp_info.attempts{end+1} = struct('warm_start',true, ...
                                         'exitflag',NaN, ...
                                         'output',ME);
        qp_info.num_failures = qp_info.num_failures + 1;
    end

    try
        [g, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], [], [], [], opts);
        qp_info.attempts{end+1} = struct('warm_start',false, ...
                                         'exitflag',exitflag, ...
                                         'output',output);
        qp_info.exitflag = exitflag;
        qp_info.output = output;
        if exitflag > 0
            qp_info.success = true;
        else
            qp_info.num_failures = qp_info.num_failures + 1;
        end
        status_text = sprintf('quadprog exitflag=%d', exitflag);
    catch ME
        g = [];
        qp_info.exitflag = NaN;
        qp_info.output = ME;
        qp_info.attempts{end+1} = struct('warm_start',false, ...
                                         'exitflag',NaN, ...
                                         'output',ME);
        qp_info.num_failures = qp_info.num_failures + 1;
        status_text = sprintf('quadprog failed: %s', ME.message);
    end
end

function [g, status_text, solver_info] = solve_g_subproblem_yalmip_osqp(kz, r_stack, pre)
% Optional verification/fallback path for the same constrained g-QP.

    qp = build_g_qp(kz, r_stack, pre);
    H = qp.H;
    f = qp.f;
    Aineq = qp.Aineq;
    bineq = qp.bineq;
    Hc = pre.Hc;

    solver_info = struct();
    solver_info.success = false;
    solver_info.diagnostic = [];

    if exist('sdpvar', 'file') ~= 2 || exist('optimize', 'file') ~= 2
        error('YALMIP is not available on the MATLAB path.');
    end

    gvar = sdpvar(Hc, 1);
    objective = 0.5 * gvar' * H * gvar + f' * gvar;
    constraints = Aineq * gvar <= bineq;
    ops = sdpsettings('solver','osqp','verbose',0,'debug',0);

    diagnostic = optimize(constraints, objective, ops);
    solver_info.diagnostic = diagnostic;

    if diagnostic.problem ~= 0
        status_text = sprintf('YALMIP+OSQP problem=%d: %s', ...
                              diagnostic.problem, diagnostic.info);
        error('YALMIP+OSQP failed for g-subproblem. %s', status_text);
    end

    g = value(gvar);
    solver_info.success = true;
    status_text = sprintf('YALMIP+OSQP problem=%d: %s', ...
                          diagnostic.problem, diagnostic.info);
end

function [valid, violation, message] = validate_g_qp_solution(g, qp)
    valid = true;
    message = 'ok';

    if isempty(g) || ~all(isfinite(g))
        valid = false;
        violation = inf;
        message = 'g contains non-finite values or is empty';
        return;
    end

    violation = max(qp.Aineq * g - qp.bineq);
    if violation > 1e-6
        valid = false;
        message = sprintf('max(Aineq*g - bineq)=%.4e', violation);
    end
end

function info = init_g_solver_info()
    info = struct();
    info.solver_used = "";
    info.solver_status = "";
    info.solver_output = [];
    info.quadprog_failures = 0;
    info.yalmip_fallback_calls = 0;
    info.max_constraint_violation = NaN;
    info.validation_message = "";
end

function stats = init_solver_stats()
    stats = struct();
    stats.quadprog_failures = 0;
    stats.yalmip_fallback_calls = 0;
end

function stats = accumulate_solver_stats(stats, info)
    if isempty(info)
        return;
    end
    if isfield(info, 'quadprog_failures')
        stats.quadprog_failures = stats.quadprog_failures + info.quadprog_failures;
    end
    if isfield(info, 'yalmip_fallback_calls')
        stats.yalmip_fallback_calls = stats.yalmip_fallback_calls + info.yalmip_fallback_calls;
    end
end

function txt = solver_output_to_text(info)
    try
        txt = evalc('disp(info)');
        txt = strtrim(txt);
    catch
        txt = '<unprintable solver output>';
    end
end

function [grad, solver_stats] = finite_difference_grad_u(u_ini, y_ini, u, r_stack, pre)
% Gradient wrt u using central finite differences on the full reduced cost
% J(u) = min_g c_q(u,g) subject to output constraints.
%
% Because the constrained g-subproblem changes the active set, this is the
% most reliable gradient implementation for the requested constrained case.
% It is more expensive than the unconstrained analytic gradient but is robust.

    eps_fd = pre.cfg.grad_eps;
    n = length(u);
    grad = zeros(n,1);
    solver_stats = init_solver_stats();

    for i = 1:n
        e = zeros(n,1); e(i) = 1;
        up = project_input_horizon(u + eps_fd * e, pre.cfg);
        um = project_input_horizon(u - eps_fd * e, pre.cfg);

        [Jp, ~, ~, ~, info_p] = total_cost_given_u(u_ini, y_ini, up, r_stack, pre);
        [Jm, ~, ~, ~, info_m] = total_cost_given_u(u_ini, y_ini, um, r_stack, pre);
        solver_stats = accumulate_solver_stats(solver_stats, info_p);
        solver_stats = accumulate_solver_stats(solver_stats, info_m);

        if isstruct(Jp)
            error('Unexpected struct return.');
        end
        grad(i) = (Jp - Jm) / (2 * eps_fd);
    end
end

function uproj = project_input_horizon(u, cfg)
    if cfg.use_input_bounds
        uproj = min(max(u, cfg.u_min), cfg.u_max);
    else
        uproj = u;
    end
end

function kz = kernel_vector_exponential(z, Zcols, cfg)
% k(z) = [ exp(z'z_1 / den); ... ; exp(z'z_Hc / den) ]
    Hc = size(Zcols,2);
    kz = zeros(Hc,1);
    for j = 1:Hc
        kernel_arg = bounded_kernel_exponent((z' * Zcols(:,j)) / cfg.exponential_den, cfg);
        kz(j) = exp(kernel_arg);
    end
end

function [kernel_arg, was_clipped] = bounded_kernel_exponent(kernel_arg, cfg)
% Keep the requested exponential kernel finite in double precision.
    was_clipped = false;
    if isnan(kernel_arg)
        error('Exponential kernel argument is NaN. Check u_data/y_data for NaN values.');
    end
    if isfield(cfg, 'kernel_exponent_limit') && ~isempty(cfg.kernel_exponent_limit)
        limit = cfg.kernel_exponent_limit;
        was_clipped = ~isfinite(kernel_arg) || kernel_arg > limit || kernel_arg < -limit;
        kernel_arg = min(max(kernel_arg, -limit), limit);
    end
end

function J = control_cost(u, Rbar, Sdu)
% Input magnitude + move suppression
    J = u' * Rbar * u + u' * Sdu * u;
end

function D = diff_operator(N)
% First-difference operator so that D*u = [u2-u1; ...; uN-u_{N-1}]
    if N <= 1
        D = zeros(0,1);
        return;
    end
    D = zeros(N-1, N);
    for i = 1:N-1
        D(i,i) = -1;
        D(i,i+1) = 1;
    end
end
