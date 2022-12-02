use ndarray::prelude::*;

/// Randomise a 2D array
pub fn randomise2(arr: &mut Array2<f64>) {
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            arr[[i, j]] = rand::random::<f64>() * 2.0 - 1.0;
        }
    }
}

/// Randomise a 1D array
pub fn randomise1(arr: &mut Array1<f64>) {
    for i in 0..arr.shape()[0] {
        arr[i] = rand::random::<f64>() * 2.0 - 1.0;
    }
}

/// Sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of the sigmoid function
pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}