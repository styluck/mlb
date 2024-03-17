% 估算sin(t)/t + (1-b)/b*cos(t) = 0 的根的上界，
% b是参数，t是变量
% f(x) = sin(x)/x + (1-b)/b*cos(x)

b = 0.1:0.1:20;
l_b = length(b);
t0 = 1;
t = zeros(l_b,1);

for i = 1:l_b
    bi = b(i);
    func = @(x) sin(x)/x + (1-bi)/bi*cos(x);

    t(i) = fzero(func, t0);
end

figure
plot(b,t,'LineWidth',4)
