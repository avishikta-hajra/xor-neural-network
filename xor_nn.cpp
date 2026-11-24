#include <iostream>
#include "matrix.hpp"
#include "functions.hpp"

int main()
{
    // Training Data
    Matrix train_x = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};

    Matrix train_y = {
        0,
        1,
        1,
        0};

    // train_x.print_matrix();
    // train_y.print_matrix();

    // Matrix Initialisation -> A : 4x3 (values initialised to 1.0f so that bias are also 1), W : 3X2, Z : 4X2 (values initialised to 0.0f)
    // 3x3x1 | Input & Hidden Layer has 3 neurons : 2 for input/activation, 1 for bias | Output Layer has 1 neuron

    // Training data is stored as row-major, so each row denotes a set of training data and the columns represent parameters

    // Layer 0 (Input) -> Hidden Layer
    Matrix A0(4, 3, 1);
    Matrix W0(3, 2);
    Matrix Z0(4, 2);

    // Hidden Layer -> Output Layer
    Matrix A1(4, 3, 1);
    Matrix W1(3, 1);
    Matrix Z1(4, 1);

    Matrix A2(4, 1); // Output Layer

    // Weight Initialisation (Random)
    W0.randomise();
    W1.randomise();
    // W0.print_matrix();
    // W1.print_matrix();

    // FORWARD PASS
    // A0 initialisation (bias col remains 1)
    matrix_copy(A0, train_x);
    // A0.print_matrix();

    Z0 = matrix_mul(A0, W0);

    matrix_activation(A1, Z0, sigmoid_forward);
    A1.print_matrix();

    Z1 = matrix_mul(A1, W1);
    matrix_activation(A2, Z1, sigmoid_forward);

    // Z0.print_matrix();
    // Z1.print_matrix();
    A2.print_matrix();
    return 0;
}