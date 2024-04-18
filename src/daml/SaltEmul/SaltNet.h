#pragma once

#include <utility>
#include <torch/torch.h>
#include <vector>

#include "oops/util/Logger.h"

// Define the Feed Forward Neural Net model
struct SaltNet : torch::nn::Module {
  SaltNet(int inputSize, int hiddenSize, int outputSize,
          int kernelSize, int stride) : inputSize_(inputSize), outputSize_(outputSize) {
    oops::Log::trace() << "Net: " << inputSize
                       << outputSize << hiddenSize << kernelSize << std::endl;

    // Define the convolution layers
    //conv = register_module("conv",
    //              torch::nn::Conv1d(torch::nn::Conv1dOptions(1,
    //                                                         hiddenSize,
    //                                                         kernelSize).stride(stride)));

    // Define the layers.
    int nin = inputSize;  //hiddenSize * ((inputSize - kernelSize)/stride + 1);
    fc1 = register_module("fc1", torch::nn::Linear(nin, hiddenSize));
    fc2 = register_module("fc2", torch::nn::Linear(hiddenSize, outputSize));

    // Register mean and std as buffers
    inputMean = register_buffer("input_mean", torch::full({inputSize}, 0.0));
    inputStd = register_buffer("input_std", torch::full({inputSize}, 1.0));
  }

  // Initialize normalization
  void initNorm(torch::Tensor mean, torch::Tensor std) {
    inputMean = mean;
    inputStd = std;
  }

  void saveNorm(const std::string modelFileName) {
//    std::vector<torch::Tensor> moments = {this->inputMean, this->inputStd};
//    std::cout << moments[0] << std::endl;
//    torch::save(moments, "normalization." + modelFileName);
  }

  void loadNorm(const std::string modelFileName) {
//    std::vector<torch::Tensor> moments;
//    torch::load(moments, "normalization." + modelFileName);
//    this->inputMean = moments[0];
//    this->inputStd = moments[1];
  }

  void initWeights() {
    // Xavier initialization for the first two layers
    //torch::nn::init::xavier_normal_(fc1->weight);
    //torch::nn::init::xavier_normal_(fc2->weight);
  }

  // Implement the forward pass
  torch::Tensor forward(torch::Tensor x) {
    //x = conv(x);
    //x = x.view({x.size(0), -1}); // Assuming x.size(0) is the batch size
    //x = fc1(x);
    //x = fc2(x);

    //x = torch::tanh(fc1(x));
    x = fc1(x);
    x = fc2(x);

    return x;
  }

  // Compute the Jacobian (dout/dx)
  std::vector<std::vector<double>> jac(torch::Tensor x, double eps) {
    // Initialize the Jacobian matrix to zeros
    //torch::Tensor jacobian = torch::ones({inputSize_, outputSize_});
    std::vector<std::vector<double>> jacobian(inputSize_, std::vector<double>(outputSize_, 0.0));
    //std::vector<std::vector<double>> jacobian;

    for (int i = 0; i < inputSize_; ++i) {
      // Save original value
      auto original_value = x[0][0][i].item<double>();

      // Perturb the current input element by eps
      x[0][0][i] = original_value + eps;
      auto output_plus_eps = this->forward(x);

      // Restore the original value
      x[0][0][i] = original_value;

      // Calculate the difference (approximation of the derivative)
      auto output = this->forward(x);

      for (int iv = 0; iv < outputSize_; iv++) {
        double fp = output_plus_eps[0][iv].item<double>();
        double f = output[0][iv].item<double>();
        jacobian[i][iv] = (fp - f)/eps;
      }
    }

    return jacobian;
  }

  // Compute the Jacobian (dout/dx)
  torch::Tensor jacNorm(torch::Tensor input) {
    input.set_requires_grad(true);
    auto output = this->forward(input);
    torch::Tensor frobeniusNorm = torch::tensor(0.0); //, torch::requires_grad(true));

    for (int i = 0; i < input.size(1); ++i) {
      auto one_hot = torch::zeros_like(input);
      one_hot.select(1, i) = 1.0;

      // Reset gradients
      this->zero_grad();

      // Backward pass for this input dimension
      output.backward(one_hot, true /* retain_graph */, true /* create_graph */);

      // Accumulate the square of gradients (approximation of Frobenius norm of Jacobian)
      //auto grad = input.grad();
      frobeniusNorm = input.grad();
      //std::cout << grad.sizes() << std::endl;
      //input.grad().zero_(); // Reset gradients for next iteration
    }
    return frobeniusNorm;
  }

  // Data members
  //torch::nn::Conv1d conv{nullptr};
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::Tensor inputMean;
  torch::Tensor inputStd;
  int inputSize_;
  int outputSize_;
};
