#include "eckit/config/YAMLConfiguration.h"
#include "eckit/filesystem/PathName.h"

#include "oops/util/Logger.h"

#include <torch/torch.h>

// Define the Feed Forward Neural Net model.
struct Net : torch::nn::Module {
  Net(int inputSize, int hiddenSize, int outputSize) {
    std::cout << inputSize << outputSize << hiddenSize << std::endl;
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

int main(int argc, char* argv[]) {

  // Read configuration
  std::string  infilename = static_cast<std::string>(argv[1]);
  eckit::PathName infilepathname = infilename;
  eckit::YAMLConfiguration config(infilepathname);

  // Define the structure of the ffnn from the configuration.
  int inputSize;
  int outputSize;
  int hiddenSize;
  config.get("ffnn.inputSize", inputSize);
  config.get("ffnn.outputSize", outputSize);
  config.get("ffnn.hiddenSize", hiddenSize);
  oops::Log::info() << "FFNN with " << inputSize << " inputs, "
                    << outputSize << " outputs" << std::endl;

  // Generate synthetic data for training.
  int dataSize;
  config.get("training data.dataSize", dataSize);
  auto data = torch::randn({dataSize, inputSize});
  auto target = torch::exp(data.sum(1));

  // Feed Forward Neural Net model.
  Net model(inputSize, hiddenSize, outputSize);

  // Loss function and optimizer.
  torch::nn::MSELoss lossFn;
  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

  // Train the model.
  size_t epochs;
  config.get("training.epochs", epochs);
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
