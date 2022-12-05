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
    x * (1.0 - x)
}

/// Mean Squared Error
pub fn mse(output: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..output.shape()[0] {
        sum += (output[i] - target[i]).powi(2);
    }
    sum / output.shape()[0] as f64
}