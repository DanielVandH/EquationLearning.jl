@testset "Basis Function Evaluation" begin
    # Is evaluate_basis working correctly?
    coefficients = [1.0, 0.57, -0.3]
    basis = [(u, p) -> u * p[1], (u, p) -> u + u^2 * p[2], (u, p) -> u^p[3]]
    p = [1.3, -3.0, 5.0]
    point = 0.005
    value = 1.0 * basis[1](point, p) + 0.57 * basis[2](point, p) - 0.3 * basis[3](point, p)
    @test EquationLearning.evaluate_basis(coefficients, basis, point, p) == value

    # Is evaluate_basis! working correctly?
    coefficients = [0.3, 6.0, 1.0]
    coefficients = [1.0, 0.57, -0.3]
    basis = [(u, p) -> u * p[1], (u, p) -> u + u^2 * p[2], (u, p) -> u^p[3]]
    p = [1.3, -3.0, 5.0]
    point = [0.05, 0.18, 1.0, 2.0, 5.0]
    val2 = zeros(length(point))
    val2[1] = EquationLearning.evaluate_basis(coefficients, basis, point[1], p)
    val2[2] = EquationLearning.evaluate_basis(coefficients, basis, point[2], p)
    val2[3] = EquationLearning.evaluate_basis(coefficients, basis, point[3], p)
    val2[4] = EquationLearning.evaluate_basis(coefficients, basis, point[4], p)
    val2[5] = EquationLearning.evaluate_basis(coefficients, basis, point[5], p)
    val = zeros(length(point))
    A = zeros(length(point), length(basis))
    EquationLearning.evaluate_basis!(val, coefficients, basis, point, p, A)
    @test val == val2
end