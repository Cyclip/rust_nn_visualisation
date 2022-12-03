extern crate ndarray;
extern crate rand;

pub mod neural_network;

fn main() {
    let mut network = neural_network::network::Network::new(vec![2, 3, 1], neural_network::utils::sigmoid, neural_network::utils::sigmoid_prime);
    println!("{:#?}", network);

    let input = ndarray::arr1(&[1.0, 2.0]);

    // Backpropagation
    let expected = ndarray::arr1(&[1.0]);
    network.backpropagate(input, expected, 0.1);
    println!("{:#?}", network);
}
