STEP_SIZE  = step_size();
MAX_POINTS = 20 / step_size();

sdf = @(x) (sd_hexgram(x));
x0 = move_to_surface([1.0, 0.0], sdf);
samples = {x0};

while length(samples) < 2 || norm(samples{end} - samples{1}) > STEP_SIZE / 2 && length(samples) < MAX_POINTS
    s = move_to_surface(counter_clockwise(samples{end}, sdf), sdf);
    samples{end + 1} = s;
end

% Make the curve closed
samples{end + 1} = samples{1};

X = 1:length(samples);
Y = 1:length(samples);

for i = 1:length(samples)
    X(i) = samples{i}(1);
    Y(i) = samples{i}(2);
end

a = 0;
b = length(samples) - 1;

t_nodes = a:1:b;
v = a:STEP_SIZE:b;

N = length(t_nodes) - 1;
xlagrange_poly = polyfit(t_nodes, X, N);
ylagrange_poly = polyfit(t_nodes, Y, N);

xlagrange = @(t) (polyval(xlagrange_poly, t));
ylagrange = @(t) (polyval(ylagrange_poly, t));

xcubic = @(t) (interp1(t_nodes, X, t, "spline"));
ycubic = @(t) (interp1(t_nodes, Y, t, "spline"));

distances = [];
for i = 1:length(samples) - 1
    distances(end + 1) = norm(samples{i + 1} - samples{i});
end


%fplot(xcubic, ycubic, [0, N], '-');
%xlim([-1, 1])
%ylim([-1, 1])
%pbaspect([1 1 1])

fplot(xlagrange, ylagrange, [0, N], '-');
xlim([-1, 1])
ylim([-1, 1])
pbaspect([1 1 1])

%plot(X, Y, '.');
%xlim([-1, 1]);
%ylim([-1, 1]);
%pbaspect([1 1 1]);

disp("Mean distance");
display(mean(distances));
disp("Standard deviation");
display(std(distances));

dist_from_curve_spline = @(t) (abs(sdf([xcubic(t), ycubic(t)])));
dist_from_curve_lagrange = @(t) (abs(sdf([xlagrange(t), ylagrange(t)])));

avg_error_spline = integral(dist_from_curve_spline, 0, length(samples), ...
    ArrayValued=true) / length(samples);

avg_error_lagrange = integral(dist_from_curve_lagrange, 0, length(samples), ...
    ArrayValued=true) / length(samples);

disp("Average distance from curve (spline):");
disp(avg_error_spline);

disp("Average distance from curve (Lagrange):");
disp(avg_error_lagrange);

% Move the given point onto the level curve f(x,y) = 0
function p = move_to_surface(point, f)
    p = point;
    tolerance = 0.0001;

    % Iterate the modified version of Newton's method to find a close point
    % on the curve
    while abs(f(p)) > tolerance
        gr = grad(p, f);
        p  = p - f(p) * gr / norm(gr)^2;
    end
end

% Move approximately counter-clockwise along the level curve
% by moving along the line tangent to it
function p = counter_clockwise(point, f)
    g = grad(point, f);
    n = g / norm(g);
    tangent = [-n(2), n(1)];

    p = point + step_size() * tangent;
end

% Compute the gradient of the given function handle at the given point
% using the 5 point midpoint formula
function g = grad_5point(point, f)
    h = mesh_size();
    dx = [h, 0];
    dy = [0, h];

    dfdx = (f(point - 2*dx) ...
        - 8*f(point - dx) ...
        + 8*f(point + dx) ...
        - f(point + 2*dx) ...
        )/(12 * h);

    dfdy = (f(point - 2*dy) ...
        - 8*f(point - dy) ...
        + 8*f(point + dy) ...
        - f(point + 2*dy) ...
        )/(12 * h);

    g = [dfdx, dfdy];
end

% Compute the gradient of the given function handle at the given point
% using the central difference method
function g = grad_midpoint(point, f)
    h = mesh_size();
    dx = [h, 0];
    dy = [0, h];

    dfdx = (f(point + dx) - f(point - dx))/(2 * h);
    dfdy = (f(point + dy) - f(point - dy))/(2 * h);

    g = [dfdx, dfdy];
end

% Compute the gradient of the given function handle at the given point
% using the forward difference method
function g = grad_forward(point, f)
    h = mesh_size();
    dx = [h, 0];
    dy = [0, h];

    dfdx = (f(point + dx) - f(point))/h;
    dfdy = (f(point + dy) - f(point))/h;

    g = [dfdx, dfdy];
end

function g = grad(point, f)
    g = grad_5point(point, f);
end

function y = clamp(x, a, b)
    y = min(max(x, a), b);
end

function step_size = step_size()
    step_size = 0.2;
end

function mesh_size = mesh_size()
    mesh_size = 0.2;
end

% Credit goes to Inigo Quilez for designing these distance functions.
% https://iquilezles.org/articles/distfunctions2d/
function d = sd_hexgram(point)
    r = 0.5;
    k = [-0.5, 0.866025403, 0.5773502692, 1.7320508076];
    p = [abs(point(1)), abs(point(2))];
    p = p - (2.0 * min(dot([k(1), k(2)], p), 0.0) * [k(1), k(2)]);
    p = p - (2.0 * min(dot([k(2), k(1)], p), 0.0) * [k(2), k(1)]);
    p = p - [clamp(p(1), r*k(3), r*k(4)), r];

    d = norm(p) * sign(p(2));
end

function d = sd_circle(point)
    r = 0.5;
    d = norm(point) - r;
end

function d = sd_uneven_capsule(point)
    r1 = 0.25;
    r2 = 0.1;
    h  = 0.4;

    p = point;
    p(1) = abs(p(1));
    b = (r1 - r2)/h;
    a = sqrt(1.0 - b*b);
    k = dot(p, [-b, a]);
    if k < 0.0
        d = norm(p) - r1;
    elseif k > a*h
        d = norm(p - [0.0, h]) - r2;
    else
        d = dot(p, [a, b]) - r1;
    end
end