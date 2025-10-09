#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include "Config.hpp"
#include "../NumBong/Tensor.hpp"
#include "../BongTorch/Module.hpp"
#include "../BongTorch/GQAAttention.hpp"
#include "../BongTorch/RMSNorm.hpp"
#include "../BongTorch/FFN_SWiGLU.hpp"
#include "../BongTorch/Conv1d.hpp"
#include "../BongTorch/Embedding.hpp"
#include "../BongTorch/RoPE.hpp"

using namespace std;

typedef shared_ptr<bs::Variable> datatype;

struct Metadatainfo {
	size_t offset;
	size_t size_in_bytes;
	nb::Shape shape;
	std::string dtype;
};

using MetadataMap = map<string, Metadatainfo>;

class Layer {
public:
	virtual ~Layer() = default;
	virtual shared_ptr<bs::Variable> forward(shared_ptr<bs::Variable> x)
	{
		return x;
	};
	virtual void loadWeights(ifstream& file, MetadataMap& metadata) = 0;

};



class ConvLayer : public Layer { 

	string name;
	bs::RMSNorm operator_norm;
	bs::Conv1DLayer conv;
	bs::RMSNorm ffn_norm;
	bs::FFN_SWiGLU feed_forward;
public:

	ConvLayer(int hidden_size,
		int intermediate_size)
	{
		operator_norm = bs::RMSNorm(hidden_size);
		conv = bs::Conv1DLayer(hidden_size, hidden_size, 3);  //나중에 한번 더 재확인
		ffn_norm = bs::RMSNorm(hidden_size);
		feed_forward = bs::FFN_SWiGLU(hidden_size, intermediate_size);
	}

	shared_ptr<bs::Variable> forward(shared_ptr<bs::Variable> x) override {
		shared_ptr<bs::Variable> residual = x;
		x = operator_norm.forward(x);
		x = conv.forward(x);
		x = add(residual, x);

		residual = x;
		x = feed_forward.forward(ffn_norm.forward(x));
		x = add(residual, x);

		return x;
	}

	
	void loadWeights(ifstream& file, const MetadataMap& metadata) override {
		MetadataMap conv_meta;
		MetadataMap ffn_meta;
		MetadataMap operator_norm_meta;
		MetadataMap ffn_norm_meta;

		for(auto [key, value]: metadata) {
			if (key.starts_with("conv.")) {
            	conv_meta[key.substr(5)] = value; // "conv." 제외
			} 
			else if (key.starts_with("feed_forward.")) {
            	ffn_meta[key.substr(13)] = value; // "feed_forward." 제외
        	} // ... norm 등 ...
			else if (key.starts_with("operator_norm.")) {
				operator_norm_meta[key.substr(14)] = value; // "operator_norm." 제외
			}
			else if (key.starts_with("ffn_norm.")) {
				ffn_norm_meta[key.substr(9)] = value; // "ffn_norm." 제외
			}
		}
    
    conv.load_weights(file, conv_meta);
    feed_forward->load_weights(file, ffn_meta);
	operator_norm.load_weights(file, operator_norm_meta);
	ffn_norm.load_weights(file, ffn_norm_meta);
	}
	
};

class AttentionLayer : public Layer {

	string name;

	bs::RMSNorm operator_norm;
	bs::GQAAttention self_attn;
	bs::RMSNorm ffn_norm;
	bs::FFN_SWiGLU feed_forward;

public:

	AttentionLayer(int hidden_size,
		int num_attention_heads,
		int num_key_value_heads,
		int intermediate_size)
	{
		int head_dim = hidden_size / num_attention_heads;
		operator_norm = bs::RMSNorm(hidden_size);
		self_attn = bs::GQAAttention(hidden_size, num_attention_heads, num_key_value_heads, head_dim);
		ffn_norm = bs::RMSNorm(hidden_size);
		feed_forward = bs::FFN_SWiGLU(hidden_size, intermediate_size);
	}

	shared_ptr<bs::Variable> forward(shared_ptr<bs::Variable> x) override {
		shared_ptr<bs::Variable> residual = x;
		x = operator_norm.forward(x);
		x = self_attn.forward(x, x, x, x);
		x = add(residual, x);

		residual = x;
		x = feed_forward.forward(ffn_norm.forward(x));
		x = add(residual, x);

		return x;
	}

	
	void loadWeights(ifstream& file, const MetadataMap& metadata) override {
		MetadataMap self_attn_meta;
		MetadataMap ffn_meta;
		MetadataMap operator_norm_meta;
		MetadataMap ffn_norm_meta;

		for(auto [key, value]: metadata) {
			if (key.starts_with("self_attn.")) {
            	conv_meta[key.substr(10)] = value; // "conv." 제외
			} 
			else if (key.starts_with("feed_forward.")) {
            	ffn_meta[key.substr(13)] = value; // "feed_forward." 제외
        	} // ... norm 등 ...
			else if (key.starts_with("operator_norm.")) {
				operator_norm_meta[key.substr(14)] = value; // "operator_norm." 제외
			}
			else if (key.starts_with("ffn_norm.")) {
				ffn_norm_meta[key.substr(9)] = value; // "ffn_norm." 제외
			}
		}
    
    self_attn.load_weights(file, conv_meta);
    feed_forward.load_weights(file, ffn_meta);
	operator_norm.load_weights(file, operator_norm_meta);
	ffn_norm.load_weights(file, ffn_norm_meta);
	}

};


class Model {

	vector<unique_ptr<Layer>> layers;
	bs::Embedding embedding;
	bs::RMSNorm embednorm;
	bs::RoPE pe;

	std::map<int, MetadataMap> weights_by_layer;
    MetadataMap other_weights;
public:

	Model(Config config) {
		
		embedding = bs::Embedding(config.vocab_size, config.hidden_size);
		embednorm = bs::RMSNorm(config.hidden_size);
		pe = bs:RoPE(config.hidden_size, config.max_position_embeddings);

		int i = 0;
		for (string type : config.layer_types) {
			//string name = "model.layers." + to_string(i);
			if (type == "conv")
			{
				layers.push_back(make_unique<ConvLayer>(config.hidden_size,
					config.intermediate_size, config.norm_eps));
			}
			else if (type == "full_attention")
			{
				layers.push_back(make_unique<AttentionLayer>(config.hidden_size,
					config.num_attention_heads, config.num_key_value_heads, config.intermediate_size));
			}
			i++;
		}
	}

	Tensor forward(Tensor x) 
	{
		
		Tensor embed = embedding.forward(x);

		//embedding norm(2048) -> (batch, token, 2048)			W(2048)
		embed = embednorm.forward(embed);

		//positional encoding -> (batch, token, 2048)			
		Tensor current = pe.forward(embed);

		//layers (batch, token, 2048)
		for (auto& layer : layers) {
			current = layer->forward(current);
			cout << endl;
		}

		//embedding  -> (batch, token, 65536)


		return current;
	}

	
	void load_weights(ifstream& file, MetadataMap metadata)
	{
		for (const auto& [key, meta] : all_metadata) {
        std::vector<std::string> parts = split(key, '.'); // key를 '.' 기준으로 분리
        
        if (parts[0] == "model" && parts[1] == "layers") {
            int layer_idx = std::stoi(parts[2]);
            // "model.layers.0." 부분을 제외한 나머지 키를 생성
            // 예: "self_attn.q_proj.weight"
            std::string child_key = join(parts.begin() + 3, parts.end(), '.');
            weights_by_layer[layer_idx][child_key] = meta;
        } else {
            other_weights[key] = meta;
        }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        if (weights_by_layer.count(i)) {
            layers[i]->load_weights(file, weights_by_layer.at(i));
        }
    }
	}

};

/*일단 유기
BongSeek makeModel()
{
	//config �о���� (���� ���Ƿ� ����)
	Config config;

	//model ��ü�� ���� return;
	BongSeek Model(config);

	ifstream file("model.safetensors");

	//safetensors �а� map ����� (���� �׽�Ʈ�� ����)


	//���� model��ü load_weights�� �н��� ����� ����ġ ����


	//Model.load_weights(file, weights);
	return Model;
}

//�߷� ���(������� ���� ū �� ����)
Tensor greedy_Decode(BongSeek model, Tensor x, Tensor x_mask, int max_len, int start_symbol)
{

	//����� greedy_decode(65536�� Ȯ�� �������� ���� ū �� �̾�
	// ��)
	//->������ sequence�� �ִ� ���̸�ŭ ��ȯ�ϸ鼭 �Է����� ���������� �������� �־��ָ鼭 
	//���ο� ��������s �߰����ش�. -> �Է� ���̰� �ϳ��� �þ�� ����

	return x;  //���� ���� tensor (��ġũ��, ��ū ����)
}*/