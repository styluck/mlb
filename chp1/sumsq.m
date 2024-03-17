function [n,y] = sumsq(m)
n = 0;
y = 0;
while y < m
    n = n + 1;
    y = y + n^2;
end
y = y - n^2;
n = n - 1;

