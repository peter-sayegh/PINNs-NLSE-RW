% Load Data
clear; clc; close all;
data = load('pinn_results_rw_foc.mat');

psi_squared_pred = data.psi_squared_pred;
l2_errors = data.l2_errors;
loss_history = data.loss_history;

L = 2.5*pi;
N_x = 512; 
N_t = 256;
T = 1.25;
x = linspace(-L, L, N_x);
t = linspace(0, 2*T, N_t);

% Compute Analytical Solution
[xx, tt] = meshgrid(x, t);
chi = sqrt(1/2);
nu = sqrt(1 - chi^2);
t_shifted = tt - T;

denominator = 4 * (xx.^2 + t_shifted.^2) + 1;
psi_squared_true = abs((1 - 4 * (1 + 2i*t_shifted) ./ denominator) .* exp(1i * t_shifted)).^2;

% Select Snapshots
time_indices = [1, N_t/4, N_t/2, 3*N_t/4, N_t]; % 5 snapshot times
ymin = min(psi_squared_true(:));
ymax = max(psi_squared_true(:));
figure;
for i = 1:length(time_indices)
    subplot(length(time_indices), 1, i);
    idx = round(time_indices(i));
    
    plot(x, psi_squared_pred(idx, :), 'b-', 'LineWidth', 1.5); hold on;
    plot(x, psi_squared_true(idx, :), 'r--', 'LineWidth', 1.5);
    
    xlabel('x');
    ylabel(['|\psi|^2 (t = ' num2str(t(idx), '%.2f') ')']);
    ylim([ymin ymax])
    grid on;
    
    if i == 1
        legend('PINN', 'Analytical', 'Location', 'northeast');
    end
end
sgtitle('Comparison of PINN and Analytical Solutions');
saveas(gcf, 'comparison_pinn_vs_analytical.png');

% Training Metrics
figure;
subplot(2,2,1);
semilogy(1:length(loss_history), loss_history, 'LineWidth', 1.5);
xlabel('Iterations'); ylabel('Training Loss');
grid on; title('Training Loss vs Iterations');

subplot(2,2,2);
semilogy(linspace(0, 2*T, length(loss_history)), loss_history, 'LineWidth', 1.5);
xlabel('Time'); ylabel('Training Loss');
grid on; title('Training Loss vs Time');

subplot(2,2,3);
semilogy(1:100:length(loss_history), l2_errors, 'LineWidth', 1.5);
xlabel('Iterations'); ylabel('L2 Error');
grid on; title('L2 Error vs Iterations');

subplot(2,2,4);
semilogy(t, sqrt(mean((psi_squared_pred - psi_squared_true).^2, 2)) ./ sqrt(mean(psi_squared_true.^2, 2)), 'LineWidth', 1.5);
xlabel('Time'); ylabel('L2 Error');
grid on; title('L2 Error vs Time');

sgtitle('Training Metrics');
saveas(gcf, 'training_metrics_pinn.png');


% Compute Point-Wise Error
pointwise_error = abs(psi_squared_pred - psi_squared_true);

% Plot the Point-Wise Error as a Heatmap
figure;
surf(x, t, pointwise_error, 'EdgeColor', 'none'); % Log scale for better contrast
colormap jet;
colorbar;
xlabel('x');
ylabel('Time');
zlabel(' Point-Wise Error');
title('3D Error Surface: |ψ|^2 (PINN - Analytical)');
view(140, 30); % Adjust viewing angle for better visualization
shading interp;
saveas(gcf, '3d_pointwise_error_pinn_vs_analytical.png');