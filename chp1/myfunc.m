function [output1, output2] = myfunc(x, y)
if nargin < 2
    y = 2;
end
% x = 10

output1 = mysubfunc2(x);

if nargout > 1
    output2 = x - y;
end
    function z1 = mysubfunc2(x1)
        z1 = x1*y;
    end
end

function z = mysubfunc1()
    z = 5;
end
