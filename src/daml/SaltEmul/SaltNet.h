#pragma once

#include <utility>
#include <torch/torch.h>

#include "oops/util/Logger.h"

// Define the Feed Forward Neural Net model
struct SaltNet : torch::nn::Module {
  SaltNet(int inputSize, int hiddenSize, int outputSize,
          int kernelSize, int stride) {
    oops::Log::trace() << "Net: " << inputSize
                       << outputSize << hiddenSize << kernelSize << std::endl;

    // Define the convolution layers
    conv = register_module("conv",
                  torch::nn::Conv1d(torch::nn::Conv1dOptions(3,
                                                             hiddenSize,
                                                             kernelSize).stride(stride)));
    conv2 = register_module("conv2",
                  torch::nn::Conv1d(torch::nn::Conv1dOptions(hiddenSize,
                                                             hiddenSize,
                                                             kernelSize).stride(1)));
    // Define the layers.
    int nin = hiddenSize * (inputSize - kernelSize + 1);
    fc1 = register_module("fc1", torch::nn::Linear(nin, hiddenSize));
    fc2 = register_module("fc2", torch::nn::Linear(hiddenSize, outputSize));
  }

  // Initialize normalization
  void initNorm(torch::Tensor mean, torch::Tensor std) {
    inputMean = mean;
    inputStd = std;
  }

  void saveNorm(const std::string modelFileName) {
    //std::vector<torch::Tensor> moments = {this->inputMean, this->inputStd};
    //std::cout << moments[0] << std::endl;
    //torch::save(moments, "normalization." + modelFileName);
  }

  void loadNorm(const std::string modelFileName) {
    //std::vector<torch::Tensor> moments;
    //torch::load(moments, "normalization." + modelFileName);
    //this->inputMean = moments[0];
    //this->inputStd = moments[1];
  }

  void initWeights() {
    // Xavier initialization for the first two layers
    //torch::nn::init::xavier_normal_(fc1->weight);
    //torch::nn::init::xavier_normal_(fc2->weight);
  }

  // Implement the forward pass
  torch::Tensor forward(torch::Tensor x) {
    x = conv(x);
    x = x.view({x.size(0), -1});
    //x = torch::sigmoid(fc1(x));
    x = fc1(x);
    torch::Tensor ds = torch::tanh(fc2(x));
    return ds;
  }

  // Compute the Jacobian (dout/dx)
  torch::Tensor jac(torch::Tensor x) {
    auto xp = torch::from_blob(x.data_ptr(), {x.size(0)}, torch::requires_grad());
    auto y = this->forward(xp);
    y.backward();
    return xp.grad();
  }

  // Data members
  torch::nn::Conv1d conv{nullptr};
  torch::nn::Conv1d conv2{nullptr};
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::Tensor inputMean;
  torch::Tensor inputStd;
};
