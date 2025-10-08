#pragma once
#include <vector>
#include <string>
#include <iostream>
using namespace std;

struct Config {
    /*
    {
  "architectures": [
    "Lfm2ForCausalLM"
  ],*/
    bool block_auto_adjust_ff_dim = false;
    int block_dim = 2048;
    int block_ff_dim = 10752;
    double block_ffn_dim_multiplier = 1.0;
    double block_mlp_init_scale = 1.0;
    int block_multiple_of = 256;
    double block_norm_eps = 1e-05;
    double block_out_init_scale = 1.0;
    bool block_use_swiglu = true;
    bool block_use_xavier_init = true;
    int bos_token_id = 1;
    int conv_L_cache = 3;
    bool conv_bias = false;
    int conv_dim = 2048;
    int conv_dim_out = 2048;
    bool conv_use_xavier_init = true;
    int eos_token_id = 7;
    int hidden_size = 2048;
    double initializer_range = 0.02;
    int intermediate_size = 10752;
    vector<string> layer_types = {
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv",
      "full_attention",
      "conv",
      "conv"
    };
    int max_position_embeddings = 128000;
    string model_type = "lfm2";
    double norm_eps = 1e-05;
    int num_attention_heads = 32;
    int num_heads = 32;
    int num_hidden_layers = 30;
    int num_key_value_heads = 8;
    int pad_token_id = 0;
    double rope_theta = 1000000.0;
    double theta = 1000000.0;
    bool tie_embedding = true;
    string torch_dtype = "bfloat16";
    string transformers_version = "4.55.0.dev0";
    bool use_cache = true;
    bool use_pos_enc = true;
    int vocab_size = 65536;
};
