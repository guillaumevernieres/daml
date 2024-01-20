#include "oops/util/Logger.h"
#include "daml/IceEmul/IceEmul.h"

int main(int argc, char* argv[]) {
  eckit::PathName infilePathName = static_cast<std::string>(argv[1]);
  eckit::YAMLConfiguration config(infilePathName);

  oops::Log::info() << "Initialize the FFNN" << std::endl;
  daml::IceEmul iceEmul(static_cast<std::string>(argv[1]));

  // Generate patterns-targets pairs and train
  if (config.has("training")) {
    oops::Log::info() << "Prepare patterns/targets pairs" << std::endl;
    std::string fileName;
    config.get("training.cice history", fileName);
    auto result = iceEmul.prepData(fileName);
    torch::Tensor inputs = std::get<0>(result);
    torch::Tensor targets = std::get<1>(result);

    oops::Log::info() << "Initialize the normalization" << std::endl;
    torch::Tensor mean = std::get<4>(result);
    torch::Tensor std = std::get<5>(result);
    iceEmul.model_->initNorm(mean, std);

    oops::Log::info() << "Train the FFNN" << std::endl;
    iceEmul.train(inputs, targets);
  }

  // Predictions
  if (config.has("prediction")) {
    oops::Log::info() << "Predict" << std::endl;
    std::string fileName;
    config.get("prediction.cice history", fileName);
    std::string fileNameResults;
    config.get("prediction.output filename", fileNameResults);
    iceEmul.predict(fileName, fileNameResults);
  }

  return 0;
}
