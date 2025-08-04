

A = rand(2, 2)
A = A' * A
b = rand(2)

B = rand(2, 2)
B = B' * B
c = rand(2)

C = rand(2, 2)
C = C' * C
d = rand(2)

f(x) = x' * A * x - b' * x
g(y) = y' * B * y - c' * y
h(z) = z' * C * z - d' * z

hole_obj = PrimalObjective(FinSet(3), z -> f(z[1:2]) + g(z[2:3]) + h(z[3:1]))
filled_obj = PrimalObjective(FinSet(4), z -> f(z[1:2]) + g(z[1:3]) + h(z[1:4]))