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
N = 20; % getting infeasiblity for N < 30 because of terminal constraint

Qx = diag([200 20 500 20]);
R = 0.01;
x_ref = [0.2; 0; 0; 0];   % desired state
x = [0.0;0.0;0.1;0.0];

umin = -10;
umax = 10;

%% Terminal invariant set under local LQR feedback
% Error dynamics: e(k+1) = Ad*e(k) + Bd*v(k), v(k) = u(k) - u_ref.
% For this reference, u_ref = 0 and e = x - x_ref.
[K_terminal,P,~] = dlqr(Ad,Bd,Qx,R);
Acl_terminal = Ad - Bd*K_terminal;

% Finite approximation of the invariant set:
% Xf = {e | umin <= -K_terminal*Acl_terminal^j*e <= umax, j = 0,...,Mset}
Mset = 80;
Hf = [];
hf = [];
terminal_set_tol = 1e-7;

for j = 0:Mset
    Aclj = Acl_terminal^j;
    Hf = [Hf;
          -K_terminal*Aclj;
           K_terminal*Aclj];
    hf = [hf;
          umax*ones(nu,1);
         -umin*ones(nu,1)];
end

terminal_set_verified = false;

if exist('linprog','file') == 2
    lp_options = optimoptions('linprog','Display','none');
    max_invariance_violation = -inf;
    terminal_set_verified = true;

    for row = 1:size(Hf,1)
        objective = -(Hf(row,:)*Acl_terminal)';
        [~,fval,exitflag] = linprog(objective,Hf,hf,[],[],[],[],lp_options);

        if exitflag <= 0
            terminal_set_verified = false;
            warning('Could not verify terminal set invariance. linprog exit flag: %d', exitflag);
            break;
        end

        max_value = -fval;
        max_invariance_violation = max(max_invariance_violation, max_value - hf(row));

        if max_value > hf(row) + terminal_set_tol
            terminal_set_verified = false;
            warning('Terminal set is not invariant for Mset = %d. Increase Mset or tighten the set.', Mset);
            break;
        end
    end
else
    max_invariance_violation = nan;
    warning('linprog is unavailable. Terminal set invariance was not verified.');
end

%% Simulation
Tsim = 500;
t = (0:Tsim-1)*Ts;


x_mpc = zeros(nx,Tsim+1);
x_hist = zeros(nx,Tsim);
u_hist = zeros(nu,Tsim);
y_hist = zeros(ny,Tsim);
solve_time_hist = zeros(1,Tsim);
step_time_hist = zeros(1,Tsim);
terminal_error_hist = zeros(nx,Tsim);

x_mpc(:,1) = x;

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

        % state and input cost
        obj = obj + ((x_var(:,i) - x_ref)'*Qx*(x_var(:,i) - x_ref)) + (u_var(:,i)'*R*u_var(:,i));

    end

    terminal_error = x_var(:,N+1) - x_ref;
    con = [con, Hf*terminal_error <= hf];
    obj = obj + terminal_error'*P*terminal_error;

    %% Solve
    solve_timer = tic;
    sol = optimize(con,obj,ops);
    solve_time_hist(k) = toc(solve_timer);

    if sol.problem ~= 0
        warning('MPC optimization failed at step %d: %s', k, sol.info);
        u_apply = 0;
        terminal_error_hist(:,k) = nan(nx,1);
    else
        %% Apply first control
        u_opt = value(u_var);

        if any(isnan(u_opt(:)))
            warning('MPC returned NaN control at step %d. Applying zero input.', k);
            u_apply = 0;
            terminal_error_hist(:,k) = nan(nx,1);
        else
            u_apply = u_opt(:,1);
            terminal_error_hist(:,k) = value(terminal_error);
        end
    end

    %% System update
    x = Ad*x + Bd*u_apply;
    y = Cd*x;

    %% Store
    x_mpc(:,k+1) = x;
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

%% Save results for animation and comparison
xMpc = x_mpc;
yMpc = y_hist;
uMpc = u_hist;
xAnim = xMpc;
uAnim = uMpc;
controller_name = 'Recursive MPC';
results_file = 'mpc_recursive_results.mat';

save(results_file, ...
    'xMpc','yMpc','uMpc', ...
    'x_hist','y_hist','u_hist', ...
    'xAnim','uAnim','controller_name', ...
    'solve_time_hist','step_time_hist', ...
    'terminal_error_hist','K_terminal','Acl_terminal','Hf','hf','Mset', ...
    'terminal_set_verified','max_invariance_violation','terminal_set_tol', ...
    'max_solve_time','avg_solve_time', ...
    'max_step_time','avg_step_time', ...
    'Ad','Bd','Cd','Ts','N','Qx','P','R','x_ref','umin','umax');

fprintf('Results saved to %s\n', results_file);

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
