% Clear previous data and close figures
close all; clear; clc;
%Šis kodas naudoja RBF tinklą sinusoidinei funkcijai aproksimuoti. Svoriai yra mokomi naudojant gradientinį nusileidimą. Grafikai rodo, 
% kaip gerai tinklas aproksimuoja tikslinę funkciją, ir paklaidą tarp tikrosios funkcijos ir tinklo išėjimo.
% Input data
x = 0.1:1/22:1; 
func = ((1 + 0.6*sin(2*pi*x/0.7)) + 0.3*sin(2*pi*x)) / 2;

% Alternative to findpeaks
dy = diff(func); % pirmosios išvestinės skirtumai.
peaks_idx = find(dy(1:end-1) > 0 & dy(2:end) < 0) + 1; % randa vietas, kur funkcijos išvestinė kerta nulį (reiškia piko tašką).
peaks = func(peaks_idx); % saugo atitinkamai pikų reikšmes ir jų vietas.
locs = x(peaks_idx); % Peak locations

% Initialize weights and RBF parameters vietoj daug nueorunu paslaptime
% naudojame tik 2 gauso funkcijos o isejimo 2 funkcijos svorio nes ten
% neurons pasislepes
w0 = randn(1); 
w1 = randn(1); 
w2 = randn(1);
c1 = locs(1,1); r1 = 0.2; %RBF centrų vietos (pasiimamos iš rastų pikų).
c2 = locs(1,2);r2 = 0.2;%RBF funkcijų spinduliai (abu nustatyti kaip 0.2).


% Plot target function with RBF centers and radii
figure(1);
plot(x, func, 'b'); 
hold on;
theta = linspace(0, 2*pi, 100);
xR1 = c1 + r1 * cos(theta); 
yR1 = (peaks(1) - r1) + r1 * sin(theta);
plot(xR1, yR1, 'r--');
xR2 = c2 + r2 * cos(theta); 
yR2 = (peaks(2) - r2) + r2 * sin(theta);
plot(xR2, yR2, 'g--');
grid on; 
hold off;

% Training step size and iterations
eta = 0.1;
for k = 1:4000
    for n = 1:length(x)
        % RBF outputs
        phi1 = exp(-((x(n)-c1)^2) / (2*(r1^2)));
        phi2 = exp(-((x(n)-c2)^2) / (2*(r2^2)));
        % Network output
         y_pred = phi1*w1 + phi2*w2 + w0;
        % Error calculation
        e = func(n) - y_pred;
        
        error = func(n) - y_pred;
        % Update weights
         w1 = w1 + eta * e * phi1;
        w2 = w2 + eta * e * phi2;%Klaida e tarp tikrosios funkcijos reikšmės ir tinklo išėjimo.
        w0 = w0 + eta * e;
    end
end

% Testing the RBF network
x_test = linspace(0.1, 1, 200);
functest = ((1 + 0.6*sin(2*pi*x_test/0.7)) + 0.3*sin(2*pi*x_test)) / 2;

for i = 1:length(x_test)
    % RBF outputs
    phi1 = exp(-((x_test(i)-c1)^2) / (2*(r1^2)));
    phi2 = exp(-((x_test(i)-c2)^2) / (2*(r2^2)));
    % Network output
    y_test(i) = phi1*w1 + phi2*w2 + w0;
    % Offset calculation
    offset(i) = y_test(i) - functest(i);
end

% Plot target vs. RBF output
figure(2);
plot(x, func, "b", x_test, y_test, "r--");
legend('tikslas', 'RBF Output');

% Plot the offset (difference)
figure(3);
plot(x_test, offset);
title('skirtumas tarp rbf i tikslo ');
xlabel('x');
ylabel('Offset');
grid on;
