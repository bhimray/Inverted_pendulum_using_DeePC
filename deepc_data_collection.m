clear; clc; close all;

%% 1. Continuous-time linearized inverted-pendulum model
M = 0.5;       % cart mass [kg]
m = 0.2;       % pole mass [kg]
b = 0.1;       % cart friction [N/(m/s)]
I = 0.006;     % pole inertia [kg*m^2]
g = 9.81;      % gravity [m/s^2]
l = 0.3;       % pole center-of-mass length [m]

p = I*(M + m) + M*m*l^2;

A = [0, 1, 0, 0;
     0, -(I + m*l^2)*b/p, (m^2*g*l^2)/p, 0;
     0, 0, 0, 1;
     0, -(m*l*b)/p, m*g*l*(M + m)/p, 0];

B = [0;
     (I + m*l^2)/p;
     0;
     m*l/p];

C = [1, 0, 0, 0;
     0, 0, 1, 0];

D = [0;
     0];

sys_c = ss(A, B, C, D);

%% 2. Zero-order-hold discretization
Ts = 0.01;
sys_d = c2d(sys_c, Ts, 'zoh');

Ad = sys_d.A;
Bd = sys_d.B;
Cd = sys_d.C;
Dd = sys_d.D;

nx = size(Ad, 1);
nu = size(Bd, 2);
ny = size(Cd, 1);

fprintf('\n=== Open-loop properties ===\n');
fprintf('Continuous-time poles:\n');
disp(eig(A));
fprintf('Discrete-time poles:\n');
disp(eig(Ad));
fprintf('Controllability rank = %d / %d\n', rank(ctrb(Ad, Bd)), nx);

%% 3. Stabilizing baseline for usable data collection
Qx = diag([30, 2, 30, 10]);
R = 10;
K = dlqr(Ad, Bd, Qx, R);

fprintf('\n=== LQR baseline for data collection ===\n');
fprintf('K = \n');
disp(K);
fprintf('Closed-loop poles:\n');
disp(eig(Ad - Bd*K));

%% 4. DeePC data and PE design parameters
Tini = 8;
Npred = 100;
L = Tini + Npred;
order_req = Tini + Npred + nx;

N = 3000;            % data length required by the idinput PRBS design
Ndata = N;           % descriptive alias used throughout the collection script
Umax = 10;           % physical actuator bound used 

collection_mode = "closed_loop_lqr_prbs";  % "closed_loop_lqr_prbs" or "strict_open_loop"

switch collection_mode
    case "closed_loop_lqr_prbs"
        prbs_max = 4;  % excitation force added before LQR; tune if needed
    case "strict_open_loop"
        prbs_max = Umax;
    otherwise
        error('Unknown collection_mode.');
end

T_min = (nu + 1)*order_req - 1;

%% 5. Bounded PRBS input from MATLAB idinput
rng(7, 'twister');  % reproducibility for the PRBS and measurement noise
u_prbs = idinput(N, 'prbs', [0 0.3], [-prbs_max prbs_max]);
u_prbs = reshape(u_prbs, 1, []);

if any(u_prbs < -prbs_max) || any(u_prbs > prbs_max)
    error('idinput generated a signal outside the requested amplitude bounds.');
end

%% 6. One continuous simulation
x0 = [0; 0; 0.35; 0]; % //TODO: TRY TO CHANGE THIS VALUE TO 0.35

X = zeros(nx, Ndata + 1);
Y = zeros(ny, Ndata);
U = zeros(1, Ndata);
Ufb = zeros(1, Ndata);
X(:, 1) = x0;

add_measurement_noise = true;
noise_std = 0.001;
phi_linear_limit = 0.35; 

for k = 1:Ndata
    switch collection_mode
        case "closed_loop_lqr_prbs"
            Ufb(k) = -K*X(:, k);
            U(k) = Ufb(k) + u_prbs(k);
        case "strict_open_loop"
            U(k) = u_prbs(k);
    end

    yk = Cd*X(:, k) + Dd*U(k);

    if add_measurement_noise
        yk = yk + noise_std*randn(size(yk));
    end

    Y(:, k) = yk;
    X(:, k + 1) = Ad*X(:, k) + Bd*U(k);
end

%% 7. Hankel matrices and persistency-of-excitation diagnostics
Hu = block_hankel(U, L, nu);
Hy = block_hankel(Y, L, ny);
Xh_data = X(:, 1:Ndata);
Hx = block_hankel(Xh_data, L, nx);

Up = Hu(1:nu*Tini, :);
Uf = Hu(nu*Tini + 1:end, :);
Yp = Hy(1:ny*Tini, :);
Yf = Hy(ny*Tini + 1:end, :);
Xp = Hx(1:nx*Tini, :);
Xf = Hx(nx*Tini + 1:end, :);

if size(Hu,2) ~= size(Hy,2) || size(Hu,2) ~= size(Hx,2)
    error('Input, output, and state Hankel matrices must have the same column count.');
end

Hu_pe = block_hankel(U, order_req, nu);
rank_Hu_pe = rank(Hu_pe);
full_row_rank_target = nu*order_req;
sv_Hu_pe = svd(Hu_pe);
min_sv_Hu_pe = min(sv_Hu_pe);
cond_Hu_pe = max(sv_Hu_pe)/max(min_sv_Hu_pe, eps);

fprintf('\n=== Hankel sizes ===\n');
fprintf('Hu: [%d x %d]\n', size(Hu, 1), size(Hu, 2));
fprintf('Hy: [%d x %d]\n', size(Hy, 1), size(Hy, 2));
fprintf('Hx: [%d x %d]\n', size(Hx, 1), size(Hx, 2));
fprintf('Up: [%d x %d]\n', size(Up, 1), size(Up, 2));
fprintf('Uf: [%d x %d]\n', size(Uf, 1), size(Uf, 2));
fprintf('Yp: [%d x %d]\n', size(Yp, 1), size(Yp, 2));
fprintf('Yf: [%d x %d]\n', size(Yf, 1), size(Yf, 2));
fprintf('Xp: [%d x %d]\n', size(Xp, 1), size(Xp, 2));
fprintf('Xf: [%d x %d]\n', size(Xf, 1), size(Xf, 2));

fprintf('\n=== Persistency of Excitation Check ===\n');
fprintf('Required PE order            = %d\n', order_req);
fprintf('rank(H_%d(U))                = %d\n', order_req, rank_Hu_pe);
fprintf('Full row-rank target         = %d\n', full_row_rank_target);
fprintf('min singular value H_%d(U)   = %.4e\n', order_req, min_sv_Hu_pe);
fprintf('condition number H_%d(U)     = %.4e\n', order_req, cond_Hu_pe);
if rank_Hu_pe == full_row_rank_target
    fprintf('PE rank test: PASSED\n');
else
    fprintf('PE rank test: FAILED\n');
end

%% 8. Dataset quality diagnostics
finite_X = all(isfinite(X(:)));
finite_Y = all(isfinite(Y(:)));
input_in_bounds = all(U >= -Umax) && all(U <= Umax);
actuator_bound_violations = nnz(abs(U) > Umax);
phi_limit_violations = nnz(abs(X(3, 1:Ndata)) > phi_linear_limit);
Y_std = std(Y, 0, 2);
X_range = max(X, [], 2) - min(X, [], 2);

fprintf('\n=== Data coverage diagnostics ===\n');
fprintf('U range                       = [%.4f, %.4f]\n', min(U), max(U));
fprintf('PRBS excitation range         = [%.4f, %.4f]\n', min(u_prbs), max(u_prbs));
fprintf('feedback input range          = [%.4f, %.4f]\n', min(Ufb), max(Ufb));
fprintf('cart position range           = [%.4f, %.4f] m\n', min(Y(1, :)), max(Y(1, :)));
fprintf('pole angle range              = [%.4f, %.4f] rad\n', min(Y(2, :)), max(Y(2, :)));
fprintf('cart output std               = %.4f m\n', Y_std(1));
fprintf('pole angle output std         = %.4f rad\n', Y_std(2));
fprintf('state ranges [x xd phi phid]  = [%.4f %.4f %.4f %.4f]\n', X_range);
fprintf('state finite                  = %d\n', finite_X);
fprintf('output finite                 = %d\n', finite_Y);
fprintf('input within bounds           = %d\n', input_in_bounds);
fprintf('actuator bound violations     = %d\n', actuator_bound_violations);
fprintf('pole angle |phi| > %.2f count = %d\n', phi_linear_limit, phi_limit_violations);
fprintf('input clipping count          = 0\n');

if max(abs(X(:))) > 1e6
    warning('Open-loop inverted-pendulum data grew very large');
end

%% 9. Save dataset
% Required output for Hankel construction and DeePC optimization.
u_data = U;
y_data = Y;
x_data = X;

save('deepc_dataset.mat', 'U', 'Y', 'X', 'u_data', 'y_data', 'x_data', 'Hx', 'Xp', 'Xf');

save('deepc_cartpole_dataset.mat', ...
     'U', 'Y', 'X', 'u_data', 'y_data', 'x_data', ...
     'Ad', 'Bd', 'Cd', 'Dd', 'Ts', ...
     'A', 'B', 'C', 'D', ...
     'M', 'm', 'b', 'I', 'g', 'l', ...
     'K', 'Qx', 'R', ...
     'Tini', 'Npred', 'L', 'order_req', 'Ndata', 'T_min', ...
     'Hu', 'Hy', 'Hx', 'Up', 'Uf', 'Yp', 'Yf', 'Xp', 'Xf', ...
     'Umax', 'prbs_max', 'u_prbs', 'Ufb', 'collection_mode', ...
     'phi_linear_limit', 'sv_Hu_pe', 'min_sv_Hu_pe', 'cond_Hu_pe', ...
     'x0', 'add_measurement_noise', 'noise_std');

fprintf('\nDataset saved to deepc_dataset.mat\n');
fprintf('Extended dataset saved to deepc_cartpole_dataset.mat\n');

%% 10. Plots
t = (0:Ndata - 1)*Ts;
t_state = (0:Ndata)*Ts;

figure('Name', 'DeePC PRBS Training Data', 'Color', 'w');

subplot(3, 1, 1);
stairs(t, U, 'LineWidth', 1.0); hold on;
stairs(t, u_prbs, ':', 'LineWidth', 0.8);
yline(Umax, '--r');
yline(-Umax, '--r');
grid on;
ylabel('u (N)');
legend('actual U', 'PRBS excitation', 'Location', 'best');
title('Actual plant input and idinput PRBS excitation');

subplot(3, 1, 2);
plot(t, Y(1, :), 'LineWidth', 1.2);
grid on;
ylabel('x (m)');
title('Cart position output');

subplot(3, 1, 3);
plot(t, Y(2, :), 'LineWidth', 1.2);
grid on;
ylabel('\phi (rad)');
xlabel('Time (s)');
title('Pole angle output');

figure('Name', 'Full State Trajectory', 'Color', 'w');
plot(t_state, X', 'LineWidth', 1.0);
grid on;
xlabel('Time (s)');
ylabel('states');
legend('x', 'x_dot', 'phi', 'phi_dot', 'Location', 'best');
title('One continuous open-loop state trajectory');



function H = block_hankel(signal, L, dim)
    % Build a block Hankel matrix from a dim x T signal.
    if isvector(signal) && dim == 1
        signal = reshape(signal, 1, []);
    end

    T = size(signal, 2);
    ncol = T - L + 1;

    if ncol <= 0
        error('Not enough data points to build a Hankel matrix.');
    end

    H = zeros(dim*L, ncol);

    for i = 1:L
        rows = (i - 1)*dim + (1:dim);
        H(rows, :) = signal(:, i:i + ncol - 1);
    end
end
