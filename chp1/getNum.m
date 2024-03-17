function out = getNum()
y(1) = 1;
for n = 1:inf
    y(n+1) = y(n)*n;
    if y(n+1) > 10^15
        break;
    end
end

out = y(end-1);