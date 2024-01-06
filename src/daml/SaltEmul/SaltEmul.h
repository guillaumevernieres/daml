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

#include "daml/Base/BaseEmul.h"
#include "SaltNet.h"

// -----------------------------------------------------------------------------

namespace daml {

  // SaltEmul class derived from BaseEmul
  class SaltEmul : public BaseEmul<SaltNet> {
   public:
    // Constructor
    explicit SaltEmul(const std::string& infileName) : BaseEmul<SaltNet>(infileName) {}

    // -----------------------------------------------------------------------------
    // Override prepData
    std::tuple<torch::Tensor, torch::Tensor, std::vector<float>, std::vector<float>>
    prepData(const std::string& fileName, bool geoloc = false) override {
      int numPatterns(2000);
      torch::Tensor patterns = torch::randn({numPatterns, 1, inputSize_}, torch::kFloat32);
      torch::Tensor targets = torch::randn({numPatterns, outputSize_}, torch::kFloat32);
      std::vector<float> lat;
      std::vector<float> lon;

      int cnt(0);
      for (size_t i = 0; i < numPatterns; ++i) {
        for (size_t z = 0; z < inputSize_; ++z) {
          patterns[i][0][z] *= 15.0;
          targets[i][z] *= 35.0;
        }
      }

      return std::make_tuple(patterns, targets, lon, lat);
    }

    // -----------------------------------------------------------------------------
    // Override predict
    void predict(const std::string& fileName, const std::string& fileNameResults) override {
    }
  };
}  // namespace daml
