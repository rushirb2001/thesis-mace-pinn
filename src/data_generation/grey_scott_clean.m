%% Cleaned-Up Gray-Scott Reaction-Diffusion Simulation

% Parameters
k = 1000; % Nondimensionalization factor

% Diffusion coefficients and reaction rates
ep1 = 0.0002 * k; % Diffusion coefficient for u
ep2 = 0.0001 * k; % Diffusion coefficient for v
b1 = 0.04 * k;    % Feed rate for u
b2 = 0.1 * k;     % Feed rate for v
c1 = k;           % Conversion rate for u
c2 = k;           % Conversion rate for v

% Discretization
nn = 200;         % Number of grid points
steps = 100;      % Number of time steps
dt = 1 / k;       % Time step size

% Domain and time
dom = [-1 1 -1 1]; % Spatial domain
t = linspace(0, 2000/k, steps+1); % Time vector

% Initialize spin operator
S = spinop2(dom, t);
S.lin = @(u,v) [ep1*lap(u); ep2*lap(v)]; % Linear part (diffusion)
S.nonlin = @(u,v) [b1*(1-u) - c1*u.*v.^2; -b2*v + c2*u.*v.^2]; % Nonlinear part (reaction)

% Initial conditions
S.init = chebfun2v(@(x,y) 1-exp(-10*((x+.05).^2 + (y+.02).^2)), ...
                   @(x,y) exp(-10*((x-.05).^2 + (y-.02).^2)), dom);

% Solve the system
tic;
u = spin2(S, nn, dt, 'plot', 'off'); % Solve using spin2
time_in_seconds = toc;

% Plot the final state of u
figure;
plot(u{1, steps}), view(0, 90), axis equal, axis off;
title('Final State of u');

% Save solutions to a grid
N = 200;
[X, Y] = meshgrid(linspace(-1, 1, N), linspace(-1, 1, N));

usol = zeros(steps+1, N, N);
vsol = zeros(steps+1, N, N);

for i = 1:steps+1
    usol(i,:,:) = u{1, i}(X, Y);
    vsol(i,:,:) = u{2, i}(X, Y);
end

% Save results
x = linspace(-1, 1, N);
y = linspace(-1, 1, N);
save('grey_scott_o.mat', 'b1', 'b2', 'c1', 'c2', 'ep1', 'ep2', 'usol', 'vsol', 't', 'x', 'y');

%% Parameter Variations for Different Patterns
% Here are some variations of the parameters to test different patterns:

% Variation 1: Spots
% ep1 = 0.00016 * k; ep2 = 0.00008 * k;
% b1 = 0.04 * k; b2 = 0.06 * k;

% Variation 2: Stripes
% ep1 = 0.00019 * k; ep2 = 0.00009 * k;
% b1 = 0.05 * k; b2 = 0.1 * k;

% Variation 3: Mixed Patterns
% ep1 = 0.00018 * k; ep2 = 0.0001 * k;
% b1 = 0.03 * k; b2 = 0.08 * k;

% Variation 4: Chaotic Patterns
% ep1 = 0.0002 * k; ep2 = 0.0001 * k;
% b1 = 0.04 * k; b2 = 0.12 * k;

% Variation 5: Self-Replicating Spots
% ep1 = 0.00015 * k; ep2 = 0.00007 * k;
% b1 = 0.03 * k; b2 = 0.05 * k;

% To test a variation, uncomment the corresponding lines and re-run the simulation.
