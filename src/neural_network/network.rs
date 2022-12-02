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

    /// Feed forward the neural network
    pub fn feed_forward(&mut self, input: Array1<f64>) {
        // Set the input layer
        self.neurons[0] = input;

        // Feed forward
        for i in 0..self.shape.len() - 1 {
            // Modify the neurons in the next layer by multiplying the
            // neurons in the current layer by the weights between the layers
            // and adding the biases
            // aₙ₊₁ = σ(wₙ₊₁ₙaₙ + bₙ₊₁) for each neuron in the next layer
            self.neurons[i + 1] = (
                self.neurons[i].dot(&self.weights[i]) 
                + &self.biases[i]
            ).mapv(self.activation);
        }

        // The output layer is now the result of the feed forward
    }

    /// Get outputs
    pub fn get_outputs(&self) -> Array1<f64> {
        self.neurons[self.shape.len() - 1].clone()
    }

    /// Find most activated output node index
    pub fn get_highest_output(&self) -> usize {
        let mut highest = 0;
        let mut highest_value = 0.0;

        for i in 0..self.shape[self.shape.len() - 1] {
            if self.neurons[self.shape.len() - 1][i] > highest_value {
                highest = i;
                highest_value = self.neurons[self.shape.len() - 1][i];
            }
        }

        highest
    }
}