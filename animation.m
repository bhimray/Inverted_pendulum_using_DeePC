clear; clc; close all;

%% Animate saved closed-loop simulation output
% Use 'lqr_baseline.mat' after running IP_LQR_mod.m.
% Use 'deepc_yalmip_results_versionB.mat' after running deepc_reg.m.

% results_file = 'lqr_baseline.mat';
% output_video = 'inverted_pendulum_lqr.mp4';

results_file = 'mpc_recursive_results.mat';
output_video = 'inverted_pendulum_mpc.mp4';

% results_file = 'deepc_yalmip_results_versionB.mat';
% output_video = 'inverted_pendulum_deepc.mp4';


if ~isfile(results_file)
    error('Missing %s. Run deepc_reg.m before creating the animation.', results_file);
end

R = load(results_file);

required_vars = {'Ts'};
for i = 1:numel(required_vars)
    if ~isfield(R, required_vars{i})
        error('%s does not contain required variable "%s".', results_file, required_vars{i});
    end
end

if isfield(R, 'xAnim')
    x_traj = R.xAnim;
elseif isfield(R, 'xLqr')
    x_traj = R.xLqr;
elseif isfield(R, 'xDeepc')
    x_traj = R.xDeepc;
elseif isfield(R, 'x')
    x_traj = R.x;
else
    error('%s must contain xAnim, xLqr, xDeepc, or x.', results_file);
end

if isfield(R, 'controller_name')
    controller_name = R.controller_name;
elseif contains(lower(results_file), 'lqr')
    controller_name = 'LQR';
elseif contains(lower(results_file), 'deepc')
    controller_name = 'DeePC';
else
    controller_name = 'Closed-loop';
end

Ts = R.Ts;

if size(x_traj, 1) < 3
    error('State trajectory must contain at least [cart position; cart velocity; pole angle].');
end

x_pos = x_traj(1, :);
theta = x_traj(3, :);
t = (0:numel(x_pos)-1) * Ts;

if isfield(R, 'uAnim')
    u_traj = [R.uAnim, R.uAnim(:, end)];
elseif isfield(R, 'uLqr')
    u_traj = [R.uLqr, R.uLqr(:, end)];
elseif isfield(R, 'uDeepc')
    u_traj = [R.uDeepc, R.uDeepc(:, end)];
elseif isfield(R, 'u')
    u_traj = [R.u, R.u(:, end)];
else
    u_traj = nan(1, numel(x_pos));
end

%% System drawing parameters
cart_width = 0.40;
cart_height = 0.20;

if isfile('deepc_cartpole_dataset.mat')
    P = load('deepc_cartpole_dataset.mat', 'l');
    pole_length = 2 * P.l;
else
    pole_length = 0.60;
end

%% Figure and axes
target_frame_rate = min(50, max(1, round(1 / Ts)));
frame_stride = max(1, round(1 / (Ts * target_frame_rate)));
frame_idx = unique([1:frame_stride:numel(t), numel(t)]);

x_span = max(0.9, max(abs(x_pos)) + pole_length + cart_width);
y_top = cart_height + pole_length + 0.20;

fig = figure('Name', sprintf('%s inverted pendulum animation', controller_name), 'Color', 'w');
set(fig, 'Position', [100 100 900 500]);
ax = axes(fig);
hold(ax, 'on');
axis(ax, 'equal');
grid(ax, 'on');
xlim(ax, [-x_span, x_span]);
ylim(ax, [-0.10, y_top]);
xlabel(ax, 'cart position (m)');
ylabel(ax, 'height (m)');

plot(ax, [-x_span, x_span], [0, 0], 'k', 'LineWidth', 2);

cart = rectangle(ax, ...
    'Position', [x_pos(1)-cart_width/2, 0, cart_width, cart_height], ...
    'FaceColor', [0 0.447 0.741], ...
    'EdgeColor', 'k');
rod = plot(ax, [0 0], [0 0], 'k', 'LineWidth', 2.5);
bob = plot(ax, 0, 0, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
time_label = title(ax, '');

video = VideoWriter(output_video, 'MPEG-4');
video.FrameRate = target_frame_rate;
open(video);

for idx = frame_idx
    x = x_pos(idx);

    pivot_x = x;
    pivot_y = cart_height;
    tip_x = pivot_x + pole_length * sin(theta(idx));
    tip_y = pivot_y + pole_length * cos(theta(idx));

    cart.Position = [x-cart_width/2, 0, cart_width, cart_height];
    rod.XData = [pivot_x, tip_x];
    rod.YData = [pivot_y, tip_y];
    bob.XData = tip_x;
    bob.YData = tip_y;

    if isfinite(u_traj(idx))
        time_label.String = sprintf('%s closed-loop simulation   t = %.2f s   u = %.3f N', ...
            controller_name, t(idx), u_traj(idx));
    else
        time_label.String = sprintf('%s closed-loop simulation   t = %.2f s', controller_name, t(idx));
    end

    drawnow;
    writeVideo(video, getframe(fig));
end

close(video);

fprintf('Animation saved to %s using %s\n', output_video, results_file);
