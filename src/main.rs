extern crate ndarray;
extern crate rand;

pub mod neural_network;

const ALPHA: f64 = 0.001;
const EPOCHS: usize = 1000;

fn main() {
    let mut network = neural_network::network::Network::new(vec![2, 3, 1], neural_network::utils::sigmoid, neural_network::utils::sigmoid_prime);
    println!("{:#?}", network);

    // Train network
    let inputs = vec![
        ndarray::arr1(&[0.0, 0.0]),
        ndarray::arr1(&[0.0, 1.0]),
        ndarray::arr1(&[1.0, 0.0]),
        ndarray::arr1(&[1.0, 1.0]),
    ];
    let expected = vec![
        ndarray::arr1(&[0.0]),
        ndarray::arr1(&[1.0]),
        ndarray::arr1(&[1.0]),
        ndarray::arr1(&[0.0]),
    ];
    network.train(&inputs, &expected, ALPHA, EPOCHS);

    // Test all inputs
    for i in 0..4 {
        network.feed_forward(&inputs[i]);
        let inp = inputs[i].to_vec();
        let out = network.get_outputs().to_vec();
        let expected = expected[i].to_vec();
        println!("Input: {:?}, Output: {:?}, Expected: {:?}", inp, out, expected);
    }
}
