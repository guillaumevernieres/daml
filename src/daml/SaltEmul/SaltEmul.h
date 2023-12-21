#pragma once

#include <netcdf>
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
      int numPatterns(200);
      torch::Tensor patterns = torch::empty({numPatterns, inputSize_}, torch::kFloat32);
      torch::Tensor targets = torch::empty({numPatterns, outputSize_}, torch::kFloat32);
      std::vector<float> lat;
      std::vector<float> lon;

      int zlev(25);
      int cnt(0);
      for (size_t i = 0; i < numPatterns; ++i) {
        for (size_t z = 0; z < inputSize_; ++z) {
          patterns[cnt][z] = 15.0;
          patterns[cnt][z + zlev] = static_cast<float>(z);
        }
        targets[cnt] = 35.0;
        cnt+=1;
      }
      return std::make_tuple(patterns, targets, lon, lat);
    }

    // -----------------------------------------------------------------------------
    // Override predict
    void predict(const std::string& fileName, const std::string& fileNameResults) override {
    }
  };
}  // namespace daml
