#pragma once

#include <memory>
#include <utility>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "oops/util/Logger.h"

// Define the Feed Forward Neural Net model
struct IceNet : torch::nn::Module {
  IceNet(int inputSize, int hiddenSize, int outputSize, int kernelSize=1, int stride=1) {
    oops::Log::trace() << "Net: " << inputSize << outputSize << hiddenSize << std::endl;
    // Define Batch Normalization layer
    //batch_norm = register_module("batch_norm", torch::nn::BatchNorm1d(1, inputSize));

    // Define the layers.
    fc1 = register_module("fc1", torch::nn::Linear(inputSize, hiddenSize));
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
    std::vector<torch::Tensor> moments = {this->inputMean, this->inputStd};
    std::cout << moments[0] << std::endl;
    torch::save(moments, "normalization." + modelFileName);
  }

  void loadNorm(const std::string modelFileName) {
    std::vector<torch::Tensor> moments;
    torch::load(moments, "normalization." + modelFileName);
    this->inputMean = moments[0];
    this->inputStd = moments[1];
  }

  //void load(const std::string modelFileName) {
  // torch::load(*this, modelFileName);
    /*
    for (const auto& pair : this->named_parameters()) {
      std::cout << "Parameter name: " << pair.key()
                << ", Size: " << pair.value().sizes() << std::endl;
    }
    for (const auto& pair : this->named_buffers()) {
      std::cout << "Buffer name: " << pair.key()
                << ", Size: " << pair.value().sizes() << std::endl;
      std::cout << "       values: " << pair.value() << std::endl;
      }*/
  //}

  // Implement the forward pass
  torch::Tensor forward(torch::Tensor x) {
    // Normalize the input
    x = (x - inputMean) / inputStd;
    x = torch::sigmoid(fc1(x));
    x = torch::sigmoid(fc2(x));
    return x;
  }

  // Compute the Jacobian (dout/dx)
  torch::Tensor jac(torch::Tensor x) {
    auto xp = torch::from_blob(x.data_ptr(), {x.size(0)}, torch::requires_grad());
    auto y = this->forward(xp);
    y.backward();
    return xp.grad();
  }

  // Define the layers.
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  //torch::nn::BatchNorm1d batch_norm{nullptr};
  torch::Tensor inputMean;
  torch::Tensor inputStd;
};
