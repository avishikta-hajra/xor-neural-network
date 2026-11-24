#pragma once

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <initializer_list>
#include <random>

class Matrix
{
private:
    size_t rows;
    size_t cols;
    std::vector<float> arr;

public:
    // Default constructor : creates an empty 0x0 matrix
    Matrix() : rows(0), cols(0) {}

    // Constructor: setting default values to 0
    Matrix(size_t r, size_t c, float val = 0.0f) : rows(r), cols(c), arr(r * c, val) {}

    // Constructor: 2D initializer list -> row-major order
    Matrix(std::initializer_list<std::initializer_list<float>> values)
    {
        rows = values.size();
        cols = (rows > 0) ? values.begin()->size() : 0;
        arr.reserve(rows * cols);

        for (const auto &row : values)
        {
            if (row.size() != cols) // rejects jagged inputs
                throw std::invalid_argument("All rows must have the same number of columns\n");
            arr.insert(arr.end(), row.begin(), row.end()); // appends row by row to the end of arr
        }
    }

    // Constructor: 1D initializer list -> stored as a column vector (N x 1)
    Matrix(std::initializer_list<float> data) : rows(data.size()), cols(data.size() > 0 ? 1 : 0), arr(data) {}

    // Dimension getters
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    bool is_empty() const { return rows == 0 || cols == 0; }

    void resize(size_t r, size_t c, float val = 0.0f) // empty or already existing Matrix
    {
        rows = r;
        cols = c;
        arr.assign(r * c, val);
    }

    // Read-only reference: for const Matrix
    const float &operator()(size_t r, size_t c) const
    {
        if (r >= rows || c >= cols)
            throw std::out_of_range("Matrix index out of range\n");
        return arr[r * cols + c];
    }

    // Read-write operator()
    float &operator()(size_t r, size_t c)
    {
        if (r >= rows || c >= cols)
            throw std::out_of_range("Matrix index out of range\n");
        return arr[r * cols + c];
    }

    // Initialisation using randomised weights -> default : uniform [-1.0, 1.0]
    void randomise(float min = -1.0f, float max = 1.0f)
    {
        if (is_empty())
        {
            std::cout << "Empty Matrix\n";
            return;
        }

        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(min, max);

        for (float &val : arr)
        {
            val = dist(gen);
        }
    }

    // Printing as 2D matrix
    void print_matrix() const
    {
        if (is_empty())
        {
            std::cout << "Empty Matrix\n";
            return;
        }

        std::cout << "\n";
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << arr[i * cols + j] << " ";
            }
            std::cout << "\n";
        }
    }

    // Matrix Multiplication
    friend Matrix matrix_mul(const Matrix &m1, const Matrix &m2)
    {
        size_t r1 = m1.rows;
        size_t c1 = m1.cols;
        size_t r2 = m2.rows;
        size_t c2 = m2.cols;

        if (c1 != r2)
        {
            throw std::invalid_argument("Matrix Multiplication Invalid -> Dimensions do not match\n");
        }

        Matrix m3(r1, c2);

        for (size_t i = 0; i < r1; ++i)
        {
            for (size_t j = 0; j < c2; ++j)
            {
                for (size_t k = 0; k < c1; ++k)
                {
                    m3(i, j) += m1(i, k) * m2(k, j);
                }
            }
        }

        return m3;
    }

    // Matrix copy -> later to be implemented as batch sizes if required
    friend void matrix_copy(Matrix &dest, const Matrix &src)
    {
        int r = src.rows;
        int c = src.cols;

        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < c; ++j)
            {
                dest(i, j) = src(i, j);
            }
        }
    }
};