#include "oops/util/Logger.h"
#include "daml/SaltEmul/SaltEmul.h"

int main(int argc, char* argv[]) {
  eckit::PathName infilePathName = static_cast<std::string>(argv[1]);
  eckit::YAMLConfiguration config(infilePathName);

  oops::Log::info() << "Initialize the FFNN" << std::endl;
  daml::SaltEmul saltEmul(static_cast<std::string>(argv[1]));

  // Generate patterns-targets pairs and train
  if (config.has("training")) {
    oops::Log::info() << "Prepare patterns/targets pairs" << std::endl;
    std::string fileName;
    config.get("training.mom6 restart", fileName);
    auto result = saltEmul.prepData(fileName);
    torch::Tensor inputs = std::get<0>(result);
    torch::Tensor targets = std::get<1>(result);

    oops::Log::info() << "Train the FFNN" << std::endl;
    saltEmul.train(inputs, targets);
  }

  // Predictions
  if (config.has("prediction")) {
    oops::Log::info() << "Predict" << std::endl;
    std::string fileName;
    std::string fileNameResults;
    config.get("prediction.output filename", fileNameResults);
    config.get("prediction.mom6 restart", fileName);
    saltEmul.predict(fileName, fileNameResults);
  }

  return 0;
}
