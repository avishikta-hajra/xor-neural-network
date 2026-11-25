#include <iostream>
#include "matrix.hpp"
#include "functions.hpp"

constexpr int EPOCHS = 100000;
constexpr size_t BATCH_SIZE = 4;
constexpr size_t LAYER_0 = 2;
constexpr size_t LAYER_1 = 2;
constexpr size_t LAYER_2 = 1;
constexpr size_t BIAS = 1;

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

    // MATRIX INITIALISATION

    // A0 : 4x3 (values initialised to 1.0f so that bias are also 1), W0 : 3X2, Z0 : 4X2 (initialised to 0.0f)
    // A1 : 4x3, W1 : 3X1, Z1 : 4X1, A2 : 4X1
    // 3x3x1 | Input & Hidden Layer has 3 neurons : 2 for input/activation, 1 for bias | Output Layer has 1 neuron

    // Training data is stored as row-major, so each row denotes a set of training data and the columns represent parameters

    // Layer 0 (Input) -> Hidden Layer
    Matrix A0(BATCH_SIZE, (LAYER_0 + BIAS), 1);
    Matrix W0((LAYER_0 + BIAS), LAYER_1);
    Matrix Z0(BATCH_SIZE, LAYER_1);

    // Hidden Layer -> Output Layer
    Matrix A1(BATCH_SIZE, (LAYER_1 + BIAS), 1);
    Matrix W1((LAYER_1 + BIAS), LAYER_2);
    Matrix Z1(BATCH_SIZE, LAYER_2);

    Matrix A2(BATCH_SIZE, LAYER_2); // Output Layer

    // for the gradient
    Matrix dW0((LAYER_0 + BIAS), LAYER_1);
    Matrix dW1(BATCH_SIZE, LAYER_1);

    // Weight Initialisation (Random)
    W0.randomise();
    W1.randomise();
    // W0.print_matrix();
    // W1.print_matrix();

    // A0 initialisation (bias col remains 1) -> SETTING THE INPUT
    matrix_copy(A0, train_x);
    // A0.print_matrix();

    // Running the epoch loops for reducing loss
    for (int ep = 0; ep < EPOCHS; ++ep)
    {
        // FORWARD PASS
        Z0 = matrix_mul(A0, W0);

        matrix_activation(A1, Z0, sigmoid_forward);
        // A1.print_matrix();

        Z1 = matrix_mul(A1, W1);
        matrix_activation(A2, Z1, sigmoid_forward);

        // Z0.print_matrix();
        // Z1.print_matrix();
        // A2.print_matrix();

        // LOSS CALCULATION
        float loss = loss_func_sse(A2, train_y);
        if (ep % 5000 == 0) // printing the new loss after 5000 iterations
            std::cout << loss << "\n";

        // NUMERICAL GRADIENTS -> each wt by wt
        float eps = 1e-4f; // change in wt

        size_t r_W0 = W0.get_rows();
        size_t c_W0 = W0.get_cols();
        for (size_t i = 0; i < r_W0; ++i)
        {
            for (size_t j = 0; j < c_W0; ++j)
            {
                float og_w = W0(i, j);
                W0(i, j) += eps;

                // Forward pass and Loss
                Z0 = matrix_mul(A0, W0);
                matrix_activation(A1, Z0, sigmoid_forward);
                Z1 = matrix_mul(A1, W1);
                matrix_activation(A2, Z1, sigmoid_forward);
                float new_loss = loss_func_sse(A2, train_y);
                // std::cout << new_loss;

                // Gradient -> slope of (loss by weight) curve
                dW0(i, j) = (new_loss - loss) / eps;

                W0(i, j) = og_w;
            }
        }

        size_t r_W1 = W1.get_rows();
        size_t c_W1 = W1.get_cols();
        for (size_t i = 0; i < r_W1; ++i)
        {
            for (size_t j = 0; j < c_W1; ++j)
            {
                float og_w = W1(i, j);
                W1(i, j) += eps;

                // Forward pass and Loss
                Z0 = matrix_mul(A0, W0);
                matrix_activation(A1, Z0, sigmoid_forward);
                Z1 = matrix_mul(A1, W1);
                matrix_activation(A2, Z1, sigmoid_forward);
                float new_loss = loss_func_sse(A2, train_y);
                // std::cout << new_loss;

                // Gradient -> slope of loss / weight curve
                dW1(i, j) = (new_loss - loss) / eps;

                W1(i, j) = og_w;
            }
        }

        // GRADIENT DESCENT for updating wts
        float alpha = 1e-2f; // learning rate or step size
        gradient_descent(W0, dW0, alpha);
        gradient_descent(W1, dW1, alpha);
    }

    A2.print_matrix();      // output of neural network
    train_y.print_matrix(); // expected output

    return 0;
}