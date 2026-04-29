#include "llama_class_example.hpp"

#include <cstdlib>
#include <iostream>

namespace {

[[noreturn]] void report_fatal_error(const char *ctx) {
  std::cerr << "[rbln] " << ctx;
  if (const char *m = rbln_get_last_error_msg(); m && m[0])
    std::cerr << ": " << m;
  std::cerr << std::endl;
  std::exit(EXIT_FAILURE);
}

void check_success(RBLNRetCode rc, const char *ctx) {
  if (rc != RBLNRetCode_SUCCESS)
    report_fatal_error(ctx);
}

template <typename T>
T *check_nonnull(T *p, const char *ctx) {
  if (!p)
    report_fatal_error(ctx);
  return p;
}

} // namespace

int WriteToFile(const std::string &filePath, const void *data,
                uint32_t data_len) {
  std::ofstream fout;
  fout.open(filePath, std::ios::out | std::ios::binary);
  if (fout.is_open()) {
    fout.write((const char *)data, data_len);
    fout.close();
    return 1;
  }
  return 0;
}

// Prefill forward method
void LLamaClass::ForwardPrefill() {
  // Get input tensors and cache position
  auto input_tensors = mdl_input_.input_ids;
  auto cache_position = mdl_input_.cache_position;
  // Get query length (number of tokens in the input sequence)
  int query_length = input_tensors.GetCols();

  // Process input in chunks (divided into chunks of size prefill_chunk_size)
  for (auto step = 0; step < query_length; step += prefill_chunk_size_) {
    // If the last chunk is incomplete (remaining tokens less than chunk size)
    if ((step + prefill_chunk_size_) > query_length) {
      // Calculate and add necessary padding to the input tensor
      int padding_needed = step + prefill_chunk_size_ - query_length;
      input_tensors = tensor_ops::Pad(input_tensors, 0, padding_needed);

      // Extend cache positions (concatenate current cache positions with additional range)
      auto new_cache_position = tensor_ops::ConcatenateWithRange(
          cache_position, query_length, step + prefill_chunk_size_);

      // Slice input tensors and cache positions for the current chunk
      auto sliced_input_tensors = tensor_ops::VerticalSlicing(
          input_tensors, step, step + prefill_chunk_size_);
      auto sliced_cache_positions = tensor_ops::VerticalSlicing(
          new_cache_position, step, step + prefill_chunk_size_);

      // Create query position and empty block tables(with value 0) for KV-cache management
      Tensor<int16_t> query_position(query_length % prefill_chunk_size_ - 1);
      Tensor<int16_t> block_tables(0);

      // Check if prefill input count exceeds expected limit
      if (rbln_get_num_inputs(prefill_rt_) > kPrefillInputCount) {
        throw std::runtime_error(
            "You appear to be running on ATOM(RBLN-CA02). RSD is only "
            "available on ATOM+(RBLN-CA12). Check your NPU type with "
            "'rbln-stat' command.");
      }

      // Set inputs for the model runtime
      check_success(rbln_set_input(prefill_rt_, 0, sliced_input_tensors.GetData()),
                 "prefill.set_input[0]");
      check_success(rbln_set_input(prefill_rt_, 1, sliced_cache_positions.GetData()),
                 "prefill.set_input[1]");
      check_success(rbln_set_input(prefill_rt_, 2, block_tables.GetData()),
                 "prefill.set_input[2]");
      check_success(rbln_set_input(prefill_rt_, 3, query_position.GetData()),
                 "prefill.set_input[3]");

      // Run the model
      check_success(rbln_run(prefill_rt_), "prefill.run");

      // Get output logits and convert to tensor
      void *logit = check_nonnull(rbln_get_output(prefill_rt_, 0), "prefill.get_output");
      auto layout = check_nonnull(rbln_get_output_layout(prefill_rt_, 0),
                                  "prefill.get_output_layout");
      output_logits_ =
          Tensor<float>(static_cast<float *>(logit), layout->shape[1], layout->shape[2]);
    }
  }

  // Predict the next token from logits using Argmax
  auto next_tokens = tensor_ops::GetArgmax<float, int64_t>(output_logits_);

  // Concatenate existing input IDs with the predicted next token
  input_ids_ = tensor_ops::Concatenate(mdl_input_.input_ids, next_tokens);
}

// Decoder forward method
void LLamaClass::ForwardDecode() {
  // Get input tensors for decoding from prefill step
  auto dec_input_tensors = mdl_input_.input_ids;
  // Get batch size from the number of rows in input tensors
  auto dec_batch_size = dec_input_tensors.GetRows();
  // Get cache positions for decoding from prefill step
  auto dec_cache_position = mdl_input_.cache_position;

  // For each item in the batch
  for (auto b_idx = 0; b_idx < dec_batch_size; b_idx++) {
    // Get the current decoding step
    auto decoding_step = dec_cache_position[b_idx];
  }

  // Initialize block tables for KV-cache management with shape of (batch_size, 1)
  Tensor<int16_t> block_tables(batch_size_, 1);

  // Check if decoder input count exceeds expected limit
  if (rbln_get_num_inputs(dec_rt_) > kDecodeInputCount) {
    throw std::runtime_error(
        "You appear to be running on ATOM(RBLN-CA02). RSD is only available on "
        "ATOM+(RBLN-CA12). Check your NPU type with 'rbln-stat' command.");
  }

  // Set inputs for decoder runtime
  check_success(rbln_set_input(dec_rt_, 0, dec_input_tensors.GetData()),
             "decode.set_input[0]");
  check_success(rbln_set_input(dec_rt_, 1, dec_cache_position.GetData()),
             "decode.set_input[1]");
  check_success(rbln_set_input(dec_rt_, 2, block_tables.GetData()),
             "decode.set_input[2]");

  // Run the decoder
  check_success(rbln_run(dec_rt_), "decode.run");

  // Get output logits from the decoder
  float *dec_logit = static_cast<float *>(
      check_nonnull(rbln_get_output(dec_rt_, 0), "decode.get_output"));
  auto dec_layout = check_nonnull(rbln_get_output_layout(dec_rt_, 0),
                                  "decode.get_output_layout");
  // Convert output to tensor format
  output_logits_ =
      Tensor<float>(dec_logit, dec_layout->shape[1], dec_layout->shape[2]);
}

// Prefill forward wrapper
void LLamaClass::DoPrefill() {
  // Run the prefill phase to process the input sequence and generate the next token
  ForwardPrefill();
}

// Decoder forward wrapper
void LLamaClass::DoDecode() {
  while (unfinished_sequences_) {
    // Prepare input for the model with current token IDs and past cache info
    PrepareInput(input_ids_, mdl_input_.past_cached_length);
    // Run the decoder to get the next token logits
    ForwardDecode();
    // Get the next token using Argmax
    auto dec_next_tokens =
        tensor_ops::GetArgmax<float, int64_t>(output_logits_);
    // Append/Concatenate the new tokens to the existing sequence
    input_ids_ = tensor_ops::Concatenate(input_ids_, dec_next_tokens);

    auto stopping_criteria = [](const auto &array) -> bool {
      const int32_t eos_token_id = 128009;
      // Stop generation if EOS token is found at the last position
      if (array(0, array.GetCols() - 1) == eos_token_id)
        return false;
      return true;
    };
    unfinished_sequences_ = stopping_criteria(input_ids_);
  }
}

void LLamaClass::Init() {
  unfinished_sequences_ = true;
}

void LLamaClass::Reset() {
  output_logits_ = Tensor<float>();
  input_ids_ = Tensor<int64_t>();
}

void LLamaClass::DeInit() {
  // Destroy runtime
  rbln_destroy_runtime(prefill_rt_);
  rbln_destroy_runtime(dec_rt_);
  // Destroy model
  rbln_destroy_model(prefill_mdl_);
  rbln_destroy_model(dec_mdl_);
}

void LLamaClass::Prepare() {
  prefill_mdl_ =
      check_nonnull(rbln_create_model(prefill_id_.c_str()), "prefill.create_model");
  dec_mdl_ =
      check_nonnull(rbln_create_model(dec_id_.c_str()), "decode.create_model");

  prefill_rt_ = check_nonnull(
      rbln_create_runtime(prefill_mdl_, nullptr, 0, 0), "prefill.create_runtime");
  dec_rt_ = check_nonnull(rbln_create_runtime(dec_mdl_, nullptr, 0, 0),
                       "decode.create_runtime");
}

void LLamaClass::GenerateBinary() {
  if (!WriteToFile("c_text2text_generation_gen_id.bin", input_ids_.GetData(),
                   input_ids_.GetSize() * sizeof(int64_t))) {
    std::cout << "Fail to save c_text2text_generation_gen_id.bin" << std::endl;
  }
}
