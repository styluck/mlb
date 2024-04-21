function y = fib(k)

if k <= 1
    y = [1;1];
else
    y = [1;1];
    for i = 3:inf
        y(i) = y(i-1) + y(i-2);
        if y(i) >= k
            y = y(1:end-1);
            break;
        end
    end
end
