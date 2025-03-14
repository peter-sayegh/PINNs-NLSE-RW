clear; clc; close all;

%% Load Data
data = load('pinn_results_2ds.mat');

% Extract variables
params = data.params;
loss_history = data.loss_history;
l2_errors = data.l2_errors;
x = data.x(:);
t = data.t(:);
psi_squared_pinn = data.psi_squared_pred;
psi_squared_fft=data.psi_squared_fft;

%% ✅ Compute Conserved Quantities for PINN and FFT
Ns_pinn = trapz(x, psi_squared_pinn, 2);  % Particle Number (integral over x)
Ns_fft = trapz(x, psi_squared_fft, 2);

%% Compute Spatial Derivative for Hamiltonian
[dt, dx] = deal(mean(diff(t)), mean(diff(x)));  % Get time step and space step
[grad_t, grad_x] = gradient(psi_squared_pinn, dt, dx);  % Compute both derivatives

%% Compute Hamiltonian Correctly
Hs_pinn = trapz(x, (1/2) * grad_x .^2 - (1/2) * psi_squared_pinn .^4, 2);
Hs_fft = trapz(x, (1/2) * gradient(psi_squared_fft, dx) .^2 - (1/2) * psi_squared_fft .^4, 2);


Ps_pinn = trapz(x, imag(conj(psi_squared_pinn) .* gradient(psi_squared_pinn, dx)), 2); % Momentum
Ps_fft = trapz(x, imag(conj(psi_squared_fft) .* gradient(psi_squared_fft, dx)), 2);

%% ✅ Plot Conserved Quantities
figure;
subplot(3,1,1);
plot(t, Ns_pinn, 'b-', 'LineWidth', 1.5); hold on;
plot(t, Ns_fft, 'r--', 'LineWidth', 1.5);
ylabel('Particle Number N'); legend('PINN', 'FFT'); grid on;

subplot(3,1,2);
plot(t, Hs_pinn, 'b-', 'LineWidth', 1.5); hold on;
plot(t, Hs_fft, 'r--', 'LineWidth', 1.5);
ylabel('Hamiltonian H'); legend('PINN', 'FFT'); grid on;

subplot(3,1,3);
plot(t, Ps_pinn, 'b-', 'LineWidth', 1.5); hold on;
plot(t, Ps_fft, 'r--', 'LineWidth', 1.5);
ylabel('Momentum P'); xlabel('Time t'); legend('PINN', 'FFT'); grid on;

saveas(gcf, 'conservation_laws.png');
fprintf("✅ Conservation plots saved as 'conservation_laws.png'\n");

%% ✅ Compare PINN and FFT Solutions at Different Time Steps
time_indices = round(linspace(1, length(t), 5)); % Pick 5 sample times

figure;
for i = 1:length(time_indices)
    tidx = time_indices(i);
    psi_sq_pinn = psi_squared_pinn(tidx, :);
    psi_sq_fft = psi_squared_fft(tidx, :);
    
    subplot(length(time_indices), 1, i);
    plot(x, psi_sq_pinn, 'b-', 'LineWidth', 1.5); hold on;
    plot(x, psi_sq_fft, 'r--', 'LineWidth', 1.5);
    ylabel(['|ψ|² (t = ', num2str(t(tidx), '%.2f'), ')']);
    legend('PINN', 'FFT');
    grid on;
end
xlabel('x');
saveas(gcf, 'solution_comparison.png');
fprintf("✅ PINN vs. FFT Comparison plots saved as 'solution_comparison.png'\n");

%% ✅ Plot Training Metrics
avg_l2_error = sqrt(mean((psi_squared_pinn - psi_squared_fft).^2, 'all')) / sqrt(mean(psi_squared_fft.^2, 'all'));
disp(['Relative L2 error of |ψ|²: ', num2str(avg_l2_error, '%.2e')]);

temporal_error = sqrt(mean((psi_squared_pinn - psi_squared_fft).^2, 2)) ./ sqrt(mean(psi_squared_fft.^2, 2));

%% Generate Correct Iteration Indices
num_loss = length(loss_history);
num_l2 = length(l2_errors);
loss_iterations = (1:num_loss) ; % Loss stored every 5 iterations
l2_iterations = (1:num_l2) * 100; % L2 error stored every 100 iterations
N_t=250;T=7.;L=10.;
%% Plot Solution Evolution
time_indices = round(linspace(1, N_t, 5));


%% ✅ Plot Training Metrics
figure;

% 🔹 Loss vs Iterations
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

saveas(gcf, 'training_metrics_2ds.png');
fprintf("✅ Training metrics saved as 'training_metrics.png'\n");

figure;

% Plot the temporal error on a logarithmic scale
semilogy(t, temporal_error, 'LineWidth', 1.5);

% Set axis labels with improved descriptions
xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');  % Add units if applicable
ylabel('L2 Error', 'FontSize', 12, 'FontWeight', 'bold');  % Add units if applicable

% Set title with improved description
title('L2 Error vs Time', 'FontSize', 14, 'FontWeight', 'bold');

% Enable grid for better readability
grid on;

% Customize the grid lines
grid minor;  % Add minor grid lines for better precision

y_ticks = [1e-3, 5e-3, 1e-2, 2e-2,3e-2,5e-2];  % Adjust based on your data
set(gca, 'YTick', y_ticks); 
set(gca, 'YTickLabel', arrayfun(@(x) sprintf('%.1e', x), y_ticks, 'UniformOutput', false));


% Optionally, set the x and y limits if needed
xlim([0, max(t)]);  % Adjust based on your data
ylim([0, max(temporal_error)]);  % Adjust based on your data
yscale('linear')

% Save the figure with a specified resolution
saveas(gcf, 'l2_vs_time_2ds.png');


%% ✅ Compute Point-wise Error
E_xt = abs(psi_squared_pinn - psi_squared_fft).^2;

%% ✅ Plot 3D Surface of Point-wise Error
figure;
surf(x, t, E_xt, 'EdgeColor', 'none'); % Surface plot without mesh lines
colorbar; % Add color scale
xlabel('Spatial Position x', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Time t', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Error E(x,t)', 'FontSize', 12, 'FontWeight', 'bold');
title('Point-wise Error E(x,t) = |ψ_{PINN}(x,t) - ψ_{FFT}(x,t)|²', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
view(135, 30); % Adjust view angle for better visualization
colormap turbo; % Use a vibrant colormap for clarity

% Save the figure
saveas(gcf, 'pointwise_error_3d.png');
fprintf("✅ Point-wise error plot saved as 'pointwise_error_3d.png'\n");
