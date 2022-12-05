use ndarray::prelude::*;
use super::utils::{randomise1, randomise2};

/// Structure for a neural network
/// Creating a network will intialize the weights and biases to random values
/// The activation function must be given
#[derive(Debug)]
pub struct Network {
    /// Stores the values for each neuron in the network
    pub neurons: Vec<Layer>,
    /// An individual weight matrix is a 2D array for the weights between two layers
    pub weights: Vec<Array2<f64>>,
    /// An individual bias matrix is a 1D array for the biases of a layer
    pub biases: Vec<Array2<f64>>,
    /// Activation function for the network
    pub activation: fn(f64) -> f64,
    /// Derivative of the activation function for the network
    pub activation_prime: fn(f64) -> f64,
    /// Shape
    pub shape: Vec<usize>,
    /// Layers num
    pub layers: usize,
}

type Layer = Array2<f64>;

impl Network {
    pub fn new(shape: Vec<usize>, activation: fn(f64) -> f64, activation_prime: fn(f64) -> f64) -> Network {
        let mut neurons = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Create the neurons shape (n, 1)
        // 2d array of zeros for each layer
        for i in 0..shape.len() {
            let n = shape[i];
            let neuron = Array2::zeros((n, 1));
            neurons.push(neuron);
        }

        // Create the weights
        for i in 0..shape.len() - 1 {
            let mut w = Array2::zeros((shape[i], shape[i + 1]));
            randomise2(&mut w);
            weights.push(w);
        }

        // Create the biases in shape (n, 1)
        // Only for the hidden and output layers
        for i in 1..shape.len() {
            // for each layer, create a bias vector. For example:
            // [
            //     [1.0],
            //     [1.0],
            //     [1.0],
            // ]
            let mut b = Array1::zeros(shape[i]);
            randomise1(&mut b);
            let b = b.into_shape((shape[i], 1)).unwrap();
            biases.push(b);
        }

        let layers = shape.len();

        Network {
            neurons,
            weights,
            biases,
            activation,
            activation_prime,
            shape,
            layers,
        }
    }

    /// Feed forward the neural network
    pub fn feed_forward(&mut self, input: &Layer) -> Layer {
        // Set the input layer to the input
        self.neurons[0] = input.clone();

        // Feed forward
        for i in 0..self.layers - 1 {
            // z = a * w + b
            // a = activation(z)
            println!("\n\nLayer: {} -> {}", i, i + 1);
            println!("Weights: {}", self.weights[i]);
            println!("Biases: {}", self.biases[i]);
            println!("Neurons: {}", self.neurons[i]);
        }

        // Return the output layer
        self.get_outputs()
    }

    /// Get outputs
    pub fn get_outputs(&self) -> Layer {
        self.neurons[self.layers - 1].clone()
    }
}