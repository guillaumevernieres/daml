#include <netcdf>

#include "eckit/config/YAMLConfiguration.h"
#include "eckit/filesystem/PathName.h"

#include "oops/util/Logger.h"

#include <torch/torch.h>

// Define the Feed Forward Neural Net model
struct Net : torch::nn::Module {
  Net(int inputSize, int hiddenSize, int outputSize) {
    oops::Log::trace() << "Net: " << inputSize << outputSize << hiddenSize << std::endl;
    // Define the layers.
    fc1 = register_module("fc1", torch::nn::Linear(inputSize, hiddenSize));
    fc2 = register_module("fc2", torch::nn::Linear(hiddenSize, outputSize));
  }

  // Implement the forward pass
  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc1(x));
    x = torch::sigmoid(fc2(x));
    return x;
  }

  // Define the layers.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

class IceEmul {
public:
  int inputSize_;
  int outputSize_;
  int hiddenSize_;
  int dataSize_;
  size_t epochs_;
  std::shared_ptr<Net> model_;

  // Constructor
  explicit IceEmul(const std::string infileName) :
    inputSize_(getSize(infileName, "ffnn.inputSize")),
    outputSize_(getSize(infileName, "ffnn.outputSize")),
    hiddenSize_(getSize(infileName, "ffnn.hiddenputSize")) {

    // Parse the configuration
    eckit::PathName infilePathName = infileName;
    eckit::YAMLConfiguration config(infilePathName);

    // Get the basic design parameters of the ffnn from the configuration.
    config.get("ffnn.outputSize", outputSize_);
    config.get("ffnn.hiddenSize", hiddenSize_);
    oops::Log::info() << "FFNN with " << inputSize_ << " inputs, "
                      << outputSize_ << " outputs" << std::endl;

    // Get the training data info
    config.get("training data.dataSize", dataSize_);

    // Optimization parameters
    config.get("training.epochs", epochs_);

    // Initialize the FFNN
    // TODO(G): figure out how to move this as an initializer
    model_ = std::make_shared<Net>(inputSize_, hiddenSize_, outputSize_);
  }

  // Training
  void train (const torch::Tensor input, const torch::Tensor target) {
    // Loss function and optimizer.
    oops::Log::info() << "Define loss fction and optimizer " << std::endl;
    torch::nn::MSELoss lossFn;
    torch::optim::Adam optimizer(model_->parameters(), torch::optim::AdamOptions(1e-3));

    // Train the model
    oops::Log::info() << "Train ..." << std::endl;
    for (size_t epoch = 0; epoch < epochs_; ++epoch) {
      // Forward pass.
      auto output = model_->forward(input);

      // Compute the loss.
      auto loss = lossFn(output.view({-1}), target);

      if (epoch % 100 == 0) {
        std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        // Save the model to a file
        torch::save(model_, "model.pt");
      }

      // Backward pass and optimization step.
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }
  }

  // Prepare the inputs/targets
  std::pair<torch::Tensor, torch::Tensor> prepData() {
    // Open the NetCDF file in read-only mode
    netCDF::NcFile ncFile("/home/gvernier/sandboxes/daml/data/iced.2021-07-01-10800.nc",
                          netCDF::NcFile::read);

    oops::Log::info() << "Invent training data " << std::endl;
    auto input = torch::randn({dataSize_, inputSize_});
    auto target = input.sum(1)/static_cast<float>(inputSize_);
    return {input, target};
  }

  // Initializers
  int getSize(const std::string infileName, const std::string paramName) {
    eckit::PathName infilePathName = infileName;
    eckit::YAMLConfiguration config(infilePathName);
    int param;
    config.get(paramName, param);
    return param;
  }
};

int main(int argc, char* argv[]) {

  IceEmul iceEmul(static_cast<std::string>(argv[1]));

  // Generate synthetic data for training.
  auto result = iceEmul.prepData();
  torch::Tensor inputs = result.first;
  torch::Tensor targets = result.second;

  // Train
  iceEmul.train(inputs, targets);

  // Dummy prediction.
  torch::Tensor input = 0.8 * torch::ones({iceEmul.inputSize_});
  torch::Tensor prediction = iceEmul.model_->forward(input);
  std::cout << "Prediction: " << prediction.index({0}) << std::endl;
  std::cout << "Truth: " << input.sum().item<float>()/static_cast<float>(iceEmul.inputSize_) << std::endl;

  return 0;
}
