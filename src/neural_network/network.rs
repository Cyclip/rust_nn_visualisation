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
    /// Layers num
    pub layers: usize,
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
    pub fn feed_forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        // Set the input layer
        self.neurons[0] = input.clone();

        // Feed forward
        for i in 0..self.layers - 1 {
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
        self.get_outputs()
    }

    /// Get outputs
    pub fn get_outputs(&self) -> Array1<f64> {
        self.neurons[self.layers - 1].clone()
    }

    /// Find most activated output node index
    pub fn get_highest_output(&self) -> usize {
        let mut highest = 0;
        let mut highest_value = 0.0;

        for i in 0..self.shape[self.layers - 1] {
            if self.neurons[self.layers - 1][i] > highest_value {
                highest = i;
                highest_value = self.neurons[self.layers - 1][i];
            }
        }

        highest
    }

    /// Get all node activations
    pub fn get_all_activations(&self) -> Vec<Array1<f64>> {
        self.neurons.clone()
    }

    /// Get all weights
    pub fn get_all_weights(&self) -> Vec<Array2<f64>> {
        self.weights.clone()
    }

    /// Get all biases
    pub fn get_all_biases(&self) -> Vec<Array1<f64>> {
        self.biases.clone()
    }

    /// MSE cost function
    pub fn cost(&self, expected: &Array1<f64>) -> f64 {
        let mut cost = 0.0;

        for i in 0..self.shape[self.layers - 1] {
            cost += (self.neurons[self.layers - 1][i] - expected[i]).powi(2);
        }

        cost / (self.shape[self.layers - 1] as f64)
    }
}

/// Backpropagation implementation
impl Network {
    pub fn train(&mut self,
        inputs: &Vec<Array1<f64>>,
        expected: &Vec<Array1<f64>>,
        learning_rate: f64,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            for i in 0..inputs.len() {
                self.backpropagate(&inputs[i], &expected[i], learning_rate);
            }
            println!("Epoch: {}, Cost: {}", epoch + 1, self.cost(&expected[0]));
        }
    }
    
    /// Backpropagation the network based on a single input to reduce the error
    fn backpropagate(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
        // Feed forward
        self.feed_forward(input);

        // Calculate the error
        let errors = target - &self.neurons[self.layers - 1];

        // Calculate delta matrix
        // First entry is the error of the output layer
        // The rest are the errors of the hidden layers
        let mut deltas = Vec::new();
        deltas.push(errors * (self.neurons[self.layers - 1].mapv(self.activation_prime)));

        // Calculate the rest of the deltas in reverse order
        for i in (1..self.layers - 1).rev() {
            let delta = self.weights[i].dot(&deltas[0]).mapv(self.activation_prime);
            deltas.insert(0, delta);
        }

        // Update the weights and biases
        for i in 0..self.layers - 1 {
            // Update the weights
            self.weights[i] = &self.weights[i] + learning_rate * &self.neurons[i].insert_axis(Axis(1)).dot(&deltas[i].insert_axis(Axis(0)));

            // Update the biases
            self.biases[i] = &self.biases[i] + learning_rate * &deltas[i];
        }
    }
}