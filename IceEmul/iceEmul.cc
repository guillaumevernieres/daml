#include "IceEmul.h"

int main(int argc, char* argv[]) {
  // TODO(G): Separate into 2 applications, or pass what needs to be done
  //          through the configuration. For ex., read precomputed weights
  //          instead of training.
  IceEmul iceEmul(static_cast<std::string>(argv[1]));

  // Generate synthetic data for training.
  std::string fileName("/home/gvernier/data/gdas.t00z.icef009.nc");
  auto result = iceEmul.prepData(fileName);
  torch::Tensor inputs = std::get<0>(result);
  torch::Tensor targets = std::get<1>(result);

  // Train
  iceEmul.train(inputs, targets);

  // Predictions
  std::string fileNameResults("gdas.t00z.icef009.ffnn.nc");
  iceEmul.predict(fileName, fileNameResults);

  return 0;
}
