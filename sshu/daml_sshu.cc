#include <torch/torch.h>

// Define the structure of the neural network.
constexpr int inputSize = 2;
constexpr int outputSize = 1;
constexpr int hiddenSize = 50;
constexpr int dataSize = 20000;

// Define the Feed Forward Neural Net model.
struct Net : torch::nn::Module {
  Net() {
    // Define the layers.
    fc1 = register_module("fc1", torch::nn::Linear(inputSize, hiddenSize));
    fc2 = register_module("fc2", torch::nn::Linear(hiddenSize, outputSize));
  }

  // Implement the forward pass.
  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc1(x));
    x = fc2(x);
    return x;
  }

  // Define the layers.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
  // Generate synthetic data for training.
  auto data = torch::randn({dataSize, inputSize});
  auto target = torch::exp(data.sum(1));

  // Feed Forward Neural Net model.
  Net model;

  // Loss function and optimizer.
  torch::nn::MSELoss lossFn;
  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

  // Train the model.
  size_t epochs = 10000;
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    // Forward pass.
    auto output = model.forward(data);

    // Compute the loss.
    auto loss = lossFn(output.view({-1}), target);

    if (epoch % 400 == 0) {
      std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }

    // Backward pass and optimization step.
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
  }

  // Dummy prediction.
  torch::Tensor input = 0.5 * torch::ones({inputSize});
  torch::Tensor prediction = model.forward(input);
  std::cout << "Prediction: " << prediction.index({0}) << std::endl;
  std::cout << "Truth: " << std::exp(input.sum().item<float>()) << std::endl;

  return 0;
}
