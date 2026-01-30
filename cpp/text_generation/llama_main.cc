#include "llama_class_example.hpp"

#include <argparse/argparse.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  // Parse arguments.

  argparse::ArgumentParser program("text_generation");
  program.add_argument("-i", "--input")
      .default_value(std::string("c_input_ids.bin"))
      .help("specify the input IDs binary file.");
  program.add_argument("-m", "--model")
      .default_value(std::string("Meta-Llama-3-8B-Instruct"))
      .help("specify the model directory or a .rbln file.");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  auto input_path = program.get<std::string>("--input");
  auto model_path = program.get<std::string>("--model");

  LLamaClass llama_cls;
  // Init Model configuration
  llama_cls.InitConfig();
  llama_cls.SetIdsPath(input_path);
  llama_cls.SetModelPath(model_path);

  // Create Model & Runtime
  llama_cls.Prepare();
  // Init LLamaClass
  llama_cls.Init();

  auto input_ids = Tensor<int64_t>(1, 23);
  assert(LoadBinary<int64_t>(llama_cls.GetIdsPath(), input_ids) == true);

  auto past_cached_length = Tensor<int32_t>();
  llama_cls.PrepareInput(input_ids, past_cached_length);

  // Process of Prefill phase
  llama_cls.DoPrefill();
  // Process of Decode phase
  llama_cls.DoDecode();
  // Generate c_text2text_generation_gen_id.bin
  llama_cls.GenerateBinary();

  // Reset LLamaClass for iteration
  llama_cls.Reset();
  // Deinit LLamaClass
  llama_cls.DeInit();

  return 0;
}
