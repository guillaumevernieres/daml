#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "eckit/config/YAMLConfiguration.h"
#include "eckit/filesystem/PathName.h"

#include "nlohmann/json.hpp"
#include "oops/util/Logger.h"
#include "torch/torch.h"

#include "IceNet.h"

// -----------------------------------------------------------------------------

namespace daml {

  /// Utilities:
  std::vector<float> readCice(const std::string fileName, const std::string varName);
  void updateProgressBar(int progress, int total, float loss);

  /// IceEmul Class
  template <typename Net>
  class IceEmul {
    int inputSize_;
    int outputSize_;
    int hiddenSize_;
    size_t epochs_;
    std::string modelOutputFileName_;
    std::shared_ptr<IceNet> model_;

    public:
    /// Constructor, destructor
    explicit IceEmul(const std::string infileName);

    /// Training
    void train(const torch::Tensor input, const torch::Tensor target);

    /// Prepare patterns/targets pairs
    std::tuple<torch::Tensor, torch::Tensor, std::vector<float>, std::vector<float>>
    prepData(std::string fileName, bool geoloc = false);

    /// Forward propagation and Jacobian
    void predict(std::string fileName, std::string fileNameResults);

    /// Initializer
    int getSize(const std::string infileName, const std::string paramName);
  };
  // -----------------------------------------------------------------------------

}  // namespace daml
