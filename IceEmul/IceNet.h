#pragma once

#include <utility>
#include "torch/torch.h"

// Define the Feed Forward Neural Net model
struct IceNet : torch::nn::Module {
  IceNet(int inputSize, int hiddenSize, int outputSize) {
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

  // Compute the Jacobian (dout/dx)
  torch::Tensor jac(torch::Tensor x) {
    auto xp = torch::from_blob(x.data_ptr(), {x.size(0)}, torch::requires_grad());
    auto y = this->forward(xp);
    y.backward();
    return xp.grad();
  }

  // Define the layers.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
