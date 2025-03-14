clear; clc; close all;

%% Load Data
data = load('pinn_results_1ds.mat');

params = data.params;
loss_history = data.loss_history(:);  % Ensure column vector
psi_squared_pred = data.psi_squared_pred;
l2_errors = data.l2_errors(:);  % Ensure column vector

%% Domain Setup
L = 10; N_x = 512;
T = 2.5; N_t = 256;
x = linspace(-L, L, N_x);
t = linspace(0, T, N_t);
[xx, tt] = meshgrid(x, t);

%% Analytical Solution
chi = sqrt(1/2);
nu = sqrt(1 - chi^2);
psi_squared_true = abs((nu*tanh(nu*(xx - chi*tt)) + 1i*chi).*exp(-1i*tt)).^2;

%% Compute L2 Error
l2_error = sqrt(mean((psi_squared_pred - psi_squared_true).^2, 'all')) / sqrt(mean(psi_squared_true.^2, 'all'));
disp(['Relative L2 error of |ψ|²: ', num2str(l2_error, '%.2e')]);

temporal_error = sqrt(mean((psi_squared_pred - psi_squared_true).^2, 2)) ./ sqrt(mean(psi_squared_true.^2, 2));

%% Generate Correct Iteration Indices
num_loss = length(loss_history);
num_l2 = length(l2_errors);
loss_iterations = (1:num_loss) ; % Loss stored every 5 iterations
l2_iterations = (1:num_l2) * 100; % L2 error stored every 100 iterations

%% Plot Solution Evolution
time_indices = round(linspace(1, N_t, 5));

figure;
hold on;
for i = 1:length(time_indices)
    plot(x, psi_squared_pred(time_indices(i), :), 'DisplayName', sprintf('t = %.2f', t(time_indices(i))),LineWidth=2.);
end
xlabel('x'); ylabel('|ψ|²'); title('Evolution of |ψ|²');
legend('show'); grid on;
saveas(gcf, 'nls_evolution.png');

%% Plot Training Metrics
figure;

% Loss vs Iterations
subplot(2,2,1);
semilogy(loss_iterations, loss_history, 'LineWidth', 1.5);
xlabel('Iterations'); ylabel('Training Loss'); grid on;
title('Loss vs Iterations');

% Loss vs Time
subplot(2,2,2);
semilogy(linspace(0, T, length(loss_history)), loss_history, 'LineWidth', 1.5);
xlabel('Time'); ylabel('Training Loss'); grid on;
title('Loss vs Time');

% L2 Error vs Iterations (Fixing incorrect indexing)
subplot(2,2,3);
semilogy(l2_iterations, l2_errors, 'LineWidth', 1.5);
xlabel('Iterations'); ylabel('L2 Error'); grid on;
title('L2 Error vs Iterations');

% L2 Error vs Time
subplot(2,2,4);
semilogy(t, temporal_error, 'LineWidth', 1.5);
xlabel('Time'); ylabel('L2 Error'); grid on;
title('L2 Error vs Time');

saveas(gcf, 'training_metrics_1ds.png');

%% ✅ Compute Point-wise Error
E_xt = abs(psi_squared_pred - psi_squared_true).^2;

%% ✅ Plot 3D Surface of Point-wise Error
figure;
surf(x, t, E_xt, 'EdgeColor', 'none');
colorbar;
xlabel('Spatial Position x', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Time t', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Point-wise Error |ψ_{pred} - ψ_{true}|^2', 'FontSize', 12, 'FontWeight', 'bold');
title('Point-wise Error: PINN vs Analytical Solution', 'FontSize', 14, 'FontWeight', 'bold');
colormap turbo;  % Vivid color mapping
view(135, 30);   % Adjust viewing angle for better clarity
grid on;

% Save the figure
saveas(gcf, 'pointwise_error_1ds.png');
fprintf("✅ Point-wise error plot saved as 'pointwise_error_1ds.png'\n");
