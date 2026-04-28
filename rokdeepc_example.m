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
% we solve the convex g-subproblem with fmincon (interior-point) or quadprog
% if you later remove the norm/robustification pieces and keep only box-type
% linear constraints.
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

required_vars = {'Ad','Bd','Cd','Dd','Ts','Tini','Npred','u_data','y_data'};
for k = 1:numel(required_vars)
    assert(exist(required_vars{k}, 'var') == 1, 'Required variable missing: %s', required_vars{k});
end

u_data = u_data(:);
if size(y_data,1) == 2 && size(y_data,2) ~= 2
    y_data = y_data.';
end
assert(size(y_data,2) == 2, 'y_data must contain exactly two outputs [x, phi].');

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
cfg.g_solver = 'fmincon';  % constrained convex smooth subproblem
cfg.fmincon_opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','off', ...
    'SpecifyObjectiveGradient',true, ...
    'HessianApproximation','lbfgs', ...
    'MaxIterations',200, ...
    'OptimalityTolerance',1e-9, ...
    'StepTolerance',1e-12, ...
    'ConstraintTolerance',1e-9);

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
for i = 1:Hc
    zi = Zcols(:,i);
    for j = i:Hc
        zj = Zcols(:,j);
        kij = exp((zi' * zj) / cfg.exponential_den);
        K(i,j) = kij;
        K(j,i) = kij;
    end
end
Kgamma = K + cfg.gamma * eye(Hc);

fprintf('Kernel Gram matrix built: %d x %d\n', size(K,1), size(K,2));

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

    u_apply = sol.u(1);
    uCL(:,t) = u_apply;

    % measured output y = [x; phi]
    y_true = [xCL(1,t); xCL(3,t)];
    y_meas = y_true + cfg.online_meas_noise_std * randn(2,1);
    yCL(:,t) = y_meas;

    % plant propagation (you can replace with nonlinear simulation if you have it)
    xCL(:,t+1) = Ad * xCL(:,t) + Bd * u_apply;

    % shift past window using latest applied input and measured output
    u_ini = [u_ini(nu+1:end); u_apply];
    y_ini = [y_ini(ny_out+1:end); y_meas];

    % shift warm start
    u_guess = [sol.u(2:end); sol.u(end)];
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
fprintf('Input violations            = %d\n', nu_viol);
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

    for it = 1:cfg.max_outer_iter
        [J, g, kz, g_status] = total_cost_given_u(u_ini, y_ini, u, r_stack, pre);
        grad = finite_difference_grad_u(u_ini, y_ini, u, r_stack, pre, J);

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
                [J_trial, ~, ~, ~] = total_cost_given_u(u_ini, y_ini, u_trial, r_stack, pre);
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

    [J, g, kz, g_status] = total_cost_given_u(u_ini, y_ini, u, r_stack, pre);

    sol = struct();
    sol.u = u;
    sol.g = g;
    sol.kz = kz;
    sol.cost = J;
    sol.outer_iter = it;
    sol.g_status = g_status;
end

function [J, g, kz, g_status] = total_cost_given_u(u_ini, y_ini, u, r_stack, pre)
% Evaluate the full objective for fixed u by solving the constrained convex
% g-subproblem.

    cfg = pre.cfg;

    z = [u_ini; y_ini; u];
    kz = kernel_vector_exponential(z, pre.Zcols, cfg.exponential_den);

    [g, g_status] = solve_g_subproblem_constrained(kz, r_stack, u, pre);

    ypred = pre.YF * g;
    kernel_res = pre.Kgamma * g - kz;

    Ju = control_cost(u, pre.Rbar, pre.Sdu);
    J = Ju + (ypred - r_stack)' * pre.Qbar * (ypred - r_stack) ...
           + cfg.lambda_g * (g' * g) ...
           + cfg.lambda_k_prime * (kernel_res' * kernel_res);
end

function [g, status_text] = solve_g_subproblem_constrained(kz, r_stack, u, pre)
% Constrained convex g-subproblem for fixed u:
%
%   min_g (YF g-r)'Q(YF g-r) + lambda_g g'g
%        + lambda_k' (Kgamma g-kz)'(Kgamma g-kz)
%
%   s.t. y_min <= YF g <= y_max
%
% This is a smooth convex quadratic problem with linear inequality
% constraints. We solve it with fmincon because you explicitly requested the
% constrained g-subproblem and a solver that is robust in the current script.
% If desired, this can be rewritten into quadprog directly.

    cfg = pre.cfg;
    Hc = pre.Hc;

    H = 2 * (pre.YF' * pre.Qbar * pre.YF + cfg.lambda_g * eye(Hc) + cfg.lambda_k_prime * (pre.Kgamma' * pre.Kgamma));
    f = -2 * (pre.YF' * pre.Qbar * r_stack + cfg.lambda_k_prime * (pre.Kgamma' * kz));

    % linear inequality constraints from y = YF g bounds
    % build stacked lower/upper bounds over horizon for [x; phi]
    y_lower_stage = [cfg.x_min; cfg.phi_min];
    y_upper_stage = [cfg.x_max; cfg.phi_max];
    y_lower = repmat(y_lower_stage, cfg.N, 1);
    y_upper = repmat(y_upper_stage, cfg.N, 1);

    Aineq = [ pre.YF; -pre.YF ];
    bineq = [ y_upper; -y_lower ];

    % initialize g by unconstrained minimizer if possible
    g0 = -(H \ f);

    switch lower(cfg.g_solver)
        case 'fmincon'
            obj = @(g) quad_obj_grad(g, H, f);
            nonlcon = [];
            [g, ~, exitflag] = fmincon(obj, g0, Aineq, bineq, [], [], [], [], nonlcon, cfg.fmincon_opts);
            status_text = sprintf('fmincon exitflag=%d', exitflag);
        otherwise
            error('Unsupported g solver: %s', cfg.g_solver);
    end
end

function [fval, grad] = quad_obj_grad(g, H, f)
    fval = 0.5 * g' * H * g + f' * g;
    grad = H * g + f;
end

function grad = finite_difference_grad_u(u_ini, y_ini, u, r_stack, pre, J0)
% Gradient wrt u using central finite differences on the full reduced cost
% J(u) = min_g c_q(u,g) subject to output constraints.
%
% Because the constrained g-subproblem changes the active set, this is the
% most reliable gradient implementation for the requested constrained case.
% It is more expensive than the unconstrained analytic gradient but is robust.

    eps_fd = pre.cfg.grad_eps;
    n = length(u);
    grad = zeros(n,1);

    for i = 1:n
        e = zeros(n,1); e(i) = 1;
        up = project_input_horizon(u + eps_fd * e, pre.cfg);
        um = project_input_horizon(u - eps_fd * e, pre.cfg);

        Jp = total_cost_given_u(u_ini, y_ini, up, r_stack, pre);
        Jm = total_cost_given_u(u_ini, y_ini, um, r_stack, pre);

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

function kz = kernel_vector_exponential(z, Zcols, den)
% k(z) = [ exp(z'z_1 / den); ... ; exp(z'z_Hc / den) ]
    Hc = size(Zcols,2);
    kz = zeros(Hc,1);
    for j = 1:Hc
        kz(j) = exp((z' * Zcols(:,j)) / den);
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
