extern crate ndarray;
extern crate rand;

pub mod neural_network;

const ALPHA: f64 = 0.5;
const EPOCHS: usize = 1000;

fn main() {
    let mut network = neural_network::network::Network::new(vec![2, 3, 1], neural_network::utils::sigmoid, neural_network::utils::sigmoid_prime);
    println!("{:#?}", network);

    // We'll be training a XOR gate (for now)
    let inputs = vec![
        ndarray::arr2(&[[0.0], [0.0]]),
        ndarray::arr2(&[[0.0], [1.0]]),
        ndarray::arr2(&[[1.0], [0.0]]),
        ndarray::arr2(&[[1.0], [1.0]]),
    ];

    let targets = vec![
        ndarray::arr1(&[0.0]),
        ndarray::arr1(&[1.0]),
        ndarray::arr1(&[1.0]),
        ndarray::arr1(&[0.0]),
    ];

    // Test all inputs
    for i in 0..inputs.len() {
        let output = network.feed_forward(&inputs[i]);
        // let cost = neural_network::utils::mse(&output, &targets[i]);
        // print dimensions of input and output
        println!("Input: {}, Output: {}", inputs[i], output);
        println!("Input dimensions: {:?}, Output dimensions: {:?}", inputs[i].shape(), output.shape());
    }
}
