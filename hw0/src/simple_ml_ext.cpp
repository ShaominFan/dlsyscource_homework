#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


int init_zero(float* res, size_t n) {
    for (int i = 0; i < n; i++) {
        res[i] = 0;
    }
    return 0;
}


int matrix_dot(float* res, float *x, size_t x_r, size_t x_c,
                const float *y, size_t y_r, size_t y_c) {
    if (x_c != y_r) {
        std::cout << "[matrix_dot]: x_c != y_r, error" << std::endl;
        return -1;
    }
    init_zero(res, x_r * y_c);
    for (int x_row = 0; x_row < x_r; x_row++) {
        for (int x_col = 0; x_col < x_c; x_col++) {
            for (int y_col = 0; y_col < y_c; y_col++) {
                res[x_row * y_c + y_col] += x[x_row * x_c + x_col] * y[x_col * y_c + y_col];
            }
        }
    }
    return 0;
}


int mul_exp(float* res, float* input, size_t row, size_t col) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            res[i * col + j] = exp(input[i * col + j]);
        }
    } 
    return 0;
}


float all_sum(float* input, size_t row, size_t col) {
    float res = 0;
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            res += input[i * col + j];
        }
    }
    return res;
}


int mul_div(float* input, size_t row, size_t col, float div_n) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            input[i * col + j] /= div_n;
        }
    }
    return 0;
}


int one_hot(float* output, uint8_t y, size_t row, size_t col) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            if ((uint8_t)j == y) {
                output[i * col + j] = 1;
            } else {
                output[i * col + j] = 0;
            }
        }
    }
    return 0;
}


int sub(float* x, float* y, size_t row, size_t col) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            x[i * col + j] -= y[i * col + j];
        }
    }
    return 0;
}


int trans(float* x_T, float* x, size_t row, size_t col) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            x_T[j * row + i] = x[i * col + j];
        }
    }
    return 0;
}


int add(float* x_add_res, float* x, size_t row, size_t col) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            x_add_res[i * col + j] += x[i * col + j];
        }
    }
    return 0;
}


int multiply(float* res, const float& n, size_t row, size_t col) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            res[i * col + j] *= n;
        }
    }
    return 0;
}


void debug_print(const std::string& info, const float* res, size_t row, size_t col) {
    std::string debug_info = info;
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            debug_info += std::to_string(res[i * col + j]) + " ";
        }
        debug_info += "\n";
    }
    std::cout << debug_info << std::endl;
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    if (m == 0) {
        return;
    }
    float sum_h;
    float *x_i = new float[n]; // 1 * n
    float *h = new float[k]; // 1 * k
    float *exp_h = new float[k]; // 1 * k
    float *e_y = new float[k]; // 1 * k

    float *x_i_T = new float[n]; // n * 1
    float *grad = new float[n * k]; // n * k
    float *batch_grad = new float[n * k]; // n * k

    uint8_t y_i = 0;
    int batch_num = 0;

    for (size_t batch_id = 0; batch_id <= m / batch; batch_id++) {
        //std::cout << "batch_id" << batch_id << std::endl;
        init_zero(batch_grad, n * k);
        batch_num = 0;
        for (size_t idx = batch_id * batch; idx < m && idx < batch_id * batch + batch; idx++) {
            //x_i copy
            //std::cout << "idx : " << idx << std::endl; 
            for (size_t n_i = 0; n_i < n; n_i++) {
                x_i[n_i] = X[idx * n + n_i];
            }
            //debug_print("x_i:\n", x_i, 1, n);
            y_i = y[idx]; // 1
            matrix_dot(h, x_i, 1, n, theta, n, k); // 1 * k
            mul_exp(exp_h, h, 1, k); // 1 * k
            sum_h = all_sum(exp_h, 1, k); // 1 * k
            mul_div(exp_h, 1, k, sum_h); // 1 * k
            
            one_hot(e_y, y_i, 1, k); // 1 * k
            sub(exp_h, e_y, 1, k); // 1 * k
            trans(x_i_T, x_i, 1, n); // n * 1
            
            matrix_dot(grad, x_i_T, n, 1, exp_h, 1, k); // n * k
            add(batch_grad, grad, n, k); // n * k
            batch_num += 1;
        }    

        if (batch_num > 0) {
            mul_div(batch_grad, n, k, batch_num); // n * k
            debug_print("batch_grad: \n", batch_grad, n, k);
            multiply(batch_grad, lr, n, k); // n * k
            debug_print("multiply: \n", batch_grad, n, k);
            sub(theta, batch_grad, n, k); // n * k
            debug_print("theta:\n", theta, n, k);
        }
    }
   
    delete[] x_i;
    delete[] h;
    delete[] exp_h;
    delete[] e_y;
    delete[] x_i_T;
    delete[] grad;
    delete[] batch_grad;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
