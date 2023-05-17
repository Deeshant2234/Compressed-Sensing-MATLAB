clc,clear,close all;
%% Generate signal, FFT of signal
n = 4096;    % points in high resolution signal
t = linspace(0, 1, n); 
x = cos(2* 97 * pi * t) + cos(2* 777 * pi * t); 
xt = fft(x); % Fourier transformed signal
PSD = xt.*conj(xt)/n;  % Power spectral density

%% Randomly sample signal
p = 128; % num. random samples, p=n/32
perm = round(rand(p, 1) * n);
y = x(perm); % compressed measurement

%% plot 1
time_window=[1024 1280]/4096;

figure
subplot(2,2,1)
plot(t,x,'k','LineWidth',2);
hold on
plot(perm/n,y,'rx','LineWidth',3);
xlabel('Time,s');
set(gca,'Fontsize',12)
axis([time_window -2 2]);
set(gcf,'Position',[1500 100 1800 1200])
title('Input Signal');
legend('input signal','spared samples');

subplot(2,2,2)
freq=n/(n)*(0:n);
L=1:floor(n/2);
plot(freq(L),PSD(L),'k','LineWidth',2);
xlabel('Frequency,Hz');
set(gca,'Fontsize',12);
axis([0 1024 0 1200]);


%% Solve compressed sensing problem
Psi = dct(eye(n, n));  % build Psi
Theta = Psi(perm, :);  % Measure rows of Psi
s = cosamp(Theta,y',10,1.e-10,10); % CS via matching pursuit
xrecon = idct(s);      % reconstruct full signal

%% Plot 2
subplot(2,2,3)
plot(t, xrecon, 'r', 'LineWidth', 2);
ylim([-2 2])
xlabel('Time,s'); set(gca,'Fontsize',12)
axis([time_window -2 2]);
title('Reconstructed 2D-Signal using Compressed Sensing');

subplot(2,2,4)
xtrecon = fft(xrecon,n);  % computes the (fast) discrete fourier transform
PSDrecon = xtrecon.*conj(xtrecon)/n;  % Power spectrum (how much power in each freq)
plot(freq(L),PSDrecon(L),'r', 'LineWidth', 2);
xlabel('Frequency,Hz');
set(gca,'Fontsize',12);
axis([0 1024 0 1200]);
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')

%%Cosamp function for finding the solution
function [s] = cosamp(Phi, y, K, tol, max_iter)
    % CoSaMP algorithm for sparse signal recovery
    
    % Initialize variables
    n = size(Phi, 2);
    residual = y;
    support = [];
    s = zeros(n, 1);
    
    iter = 0;
    while (iter < max_iter) && (norm(residual) > tol)
        iter = iter + 1;
        
        % Step 1: Estimate the signal support
        projection = abs(Phi' * residual);
        [~, indices] = sort(projection, 'descend');
        new_support = union(support, indices(1:2*K));  % Update support
        
        % Step 2: Solve the least squares problem
        submatrix = Phi(:, new_support);
        x = pinv(submatrix) * y;
        
        % Step 3: Update the estimate
        [~, sorted_indices] = sort(abs(x), 'descend');
        support = new_support(sorted_indices(1:K));  % Update support
        s(support) = x(sorted_indices(1:K));  % Update estimate
        
        % Step 4: Update the residual
        residual = y - Phi(:, support) * s(support);
    end
end
