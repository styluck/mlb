function out = fowlnbunnyincage(head, foot)
syms fowl bunny real;

eq1 = fowl + bunny == head;
eq2 = 2*fowl + 4*bunny == foot;

out = solve(eq1, eq2, fowl, bunny);


