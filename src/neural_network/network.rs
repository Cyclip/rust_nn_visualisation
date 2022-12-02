use ndarray::prelude::*;
use super::utils::{randomise1, randomise2};

/// Structure for a neural network
/// Creating a network will intialize the weights and biases to random values
/// The activation function must be given
#[derive(Debug)]
pub struct Network {
    /// Stores the values for each neuron in the network
    pub neurons: Vec<Array1<f64>>,
    /// An individual weight matrix is a 2D array for the weights between two layers
    pub weights: Vec<Array2<f64>>,
    /// An individual bias matrix is a 1D array for the biases of a layer
    pub biases: Vec<Array1<f64>>,
    /// Activation function for the network
    pub activation: fn(f64) -> f64,
    /// Derivative of the activation function for the network
    pub activation_prime: fn(f64) -> f64,
    /// Shape
    pub shape: Vec<usize>,
}

impl Network {
    pub fn new(shape: Vec<usize>, activation: fn(f64) -> f64, activation_prime: fn(f64) -> f64) -> Network {
        let mut neurons = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Create the neurons
        for i in 0..shape.len() {
            neurons.push(Array1::zeros(shape[i]));
        }

        // Create the weights
        for i in 0..shape.len() - 1 {
            let mut w = Array2::zeros((shape[i], shape[i + 1]));
            randomise2(&mut w);
            weights.push(w);
        }

        // Create the biases
        for i in 1..shape.len() {
            let mut b = Array1::zeros(shape[i]);
            randomise1(&mut b);
            biases.push(b);
        }

        Network {
            neurons,
            weights,
            biases,
            activation,
            activation_prime,
            shape,
        }
    }
}