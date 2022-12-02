extern crate ndarray;
extern crate rand;

pub mod neural_network;

fn main() {
    let mut network = neural_network::network::Network::new(vec![2, 3, 1], neural_network::utils::sigmoid, neural_network::utils::sigmoid_prime);
    println!("{:#?}", network);

    let input = ndarray::arr1(&[1.0, 2.0]);
    network.feed_forward(input);
    println!("{:#?}", network);

    // Print outputs
    println!("\n\nOutputs: {:#?}", network.get_outputs());
    println!("\nHighest output: {:#?}", network.get_highest_output());
}
