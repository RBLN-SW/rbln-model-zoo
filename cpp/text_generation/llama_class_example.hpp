#ifndef RBLN_LLAMA_H
#define RBLN_LLAMA_H

#include <assert.h>
#include <rbln/rbln.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "llama_tensor_example.hpp"
#include "llama_tensor_op_example.hpp"

constexpr uint32_t kDecodeInputCount = 3;
constexpr uint32_t kPrefillInputCount = 4;

template <typename T>
bool LoadBinary(const std::string &filename, Tensor<T> &data) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Could not open file: " + filename << std::endl;
    return false;
  }

  file.seekg(0, std::ios::end);
  const size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  if (fileSize % sizeof(T) != 0) {
    std::cout << "File size(" << fileSize << ") is not a multiple of data type size(" 
          << sizeof(T) << ")" << std::endl;
    return false;
  }

  file.read(const_cast<char *>(static_cast<const char *>(data.GetData())),
            fileSize);
  if (file.fail()) {
    std::cout << "Failed to read file: " << filename << std::endl;
    return false;
  }
  return true;
}

int WriteToFile(const std::string &filePath, const void *data,
                uint32_t data_len);

class LLamaClass {
public:
  LLamaClass() = default;
  ~LLamaClass() = default;

  // Init Model configuration
  void InitConfig() {
    // Defaults are relative to the current working directory.
    // With default compile.py settings, artifacts are saved under:
    //   Meta-Llama-3-8B-Instruct/
    input_ids_path_ = "c_input_ids.bin";

    prefill_id_ = "Meta-Llama-3-8B-Instruct/prefill.rbln";
    dec_id_ = "Meta-Llama-3-8B-Instruct/decoder_batch_1.rbln";

    batch_size_ = 1;
    max_seq_len_ = 8192;
    prefill_chunk_size_ = 128;
  }

  // Set model path from command line argument
  void SetModelPath(const std::string& model_path) {
    // If model_path ends with .rbln, use its directory
    std::string base_path = model_path;
    if (base_path.size() > 5 && base_path.substr(base_path.size() - 5) == ".rbln") {
      size_t last_slash = base_path.find_last_of("/");
      if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
      } else {
        base_path = ".";
      }
    }
    
    // Construct prefill and decoder paths
    prefill_id_ = base_path + "/prefill.rbln";
    dec_id_ = base_path + "/decoder_batch_1.rbln";
  }

  // Init LLamaClass
  void Init();

  // Reset LLamaClass for iteration
  void Reset();

  // Deinit LLamaClass
  void DeInit();

  // Create Model & Runtime
  void Prepare();

  // Process of Prefill phase
  void DoPrefill();

  // Process of Decode phase
  void DoDecode();

  // Generate c_text2text_generation_gen_id.bin
  void GenerateBinary();

  template <typename T0, typename T1>
  void PrepareInput(Tensor<T0> &input_ids, Tensor<T1> &v0) {
    if (!v0.GetSize()) {
      auto input_tensors = input_ids;
      auto batch_size = input_tensors.GetRows();
      std::vector<Tensor<int64_t>> l_input_tensors;
      std::vector<Tensor<int32_t>> cache_positions;
      auto past_cached_length = Tensor<int32_t>(batch_size, 1);

      for (int i = 0; i < batch_size; i++) {
        auto input_tensor =
            tensor_ops::Reshape(input_tensors, input_tensors.GetCols());

        auto valid_len = input_tensor.GetCols();
        auto cache_position = Tensor<int32_t>();
        tensor_ops::Arange(cache_position, 0, valid_len);
        tensor_ops::Reshape(cache_position, 1, valid_len);

        past_cached_length[i] = valid_len;
        l_input_tensors.emplace_back(tensor_ops::UnSqueeze(input_tensor));
        cache_positions.emplace_back(tensor_ops::UnSqueeze(cache_position));
      }
      mdl_input_ = ModelInput{l_input_tensors[0], cache_positions[0],
                              past_cached_length};
    } else {
      auto input_tensor = tensor_ops::SelectLastColumn(input_ids);
      auto cache_positions = v0;
      auto past_cached_length = v0 + 1;
      mdl_input_ =
          ModelInput{input_tensor, cache_positions, past_cached_length};
    }
  }

  const std::string &GetIdsPath() { return input_ids_path_; }

  void SetIdsPath(const std::string &ids_path) { input_ids_path_ = ids_path; }

  RBLNModel *prefill_mdl_;
  RBLNModel *dec_mdl_;
  RBLNRuntime *prefill_rt_;
  RBLNRuntime *dec_rt_;

private:
  void ForwardPrefill();
  void ForwardDecode();

  typedef struct {
    Tensor<int64_t> input_ids;
    Tensor<int32_t> cache_position;
    Tensor<int32_t> past_cached_length;
  } ModelInput;

  ModelInput mdl_input_;
  int max_seq_len_;
  int batch_size_;
  int prefill_chunk_size_;
  bool unfinished_sequences_;

  std::string prefill_id_;
  std::string dec_id_;
  std::string input_ids_path_;

  Tensor<float> output_logits_;
  Tensor<int64_t> input_ids_;
};

#endif
