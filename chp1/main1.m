% 单变量函数定义
func = @(x) x + 1;
x = 100;
y = func(x)

% 多变量函数定义
func2 = @(x,y) sin(x) + cos(y);
x = 1:.01:pi;
y = 1:.01:pi;
lx = length(x);
ly = length(y);
tic
z = zeros(lx,ly);
for i = 1:lx
    for j = 1:ly
        z(i,j) = func2(x(i),y(j));
    end
end
toc
mesh(z)
x = 10;
y = 2
z1 = myfunc(x,y)

fib(1000000)

getNum()
head = 35;
foot = 94;
out = fowlnbunnyincage(head, foot)