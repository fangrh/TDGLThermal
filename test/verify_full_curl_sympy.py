from sympy import Function, Matrix, diff, simplify, symbols

x, y = symbols("x y")
Ax = Function("Ax")(x, y)
Ay = Function("Ay")(x, y)

curl_A_z = diff(Ay, x) - diff(Ax, y)
curlcurl_A = Matrix([
    diff(curl_A_z, y),
    -diff(curl_A_z, x),
    0,
])

expected = Matrix([
    diff(Ay, x, y) - diff(Ax, y, y),
    diff(Ax, x, y) - diff(Ay, x, x),
    0,
])

assert simplify(curlcurl_A[0] - expected[0]) == 0
assert simplify(curlcurl_A[1] - expected[1]) == 0

print("curl curl A matches the 2D mixed-derivative form")
