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
#include "../BongTorch/Core.hpp"
#include "../BongTorch/GQAAttention.hpp"
#include "../BongTorch/RMSNorm.hpp"
#include "../BongTorch/FFN_SWiGLU.hpp"
#include "../BongTorch/Conv1d.hpp"
#include "../BongTorch/Embedding.hpp"
#include "../BongTorch/RoPE.hpp"

using namespace std;

typedef shared_ptr<bs::Variable> datatype;



class Layer {
public:
	virtual ~Layer() = default;
	virtual shared_ptr<bs::Variable> forward(shared_ptr<bs::Variable> x)
	{
		return x;
	};
	virtual void loadWeights(istream& file, const MetadataMap& metadata) = 0;

};



class ConvLayer : public Layer { 

	string name;
	bs::RMSNorm operator_norm;
	bs::Conv1d conv;
	bs::RMSNorm ffn_norm;
	bs::FFN_SWiGLU feed_forward;
public:

	ConvLayer(int hidden_size,
		int intermediate_size)
	{
		operator_norm = bs::RMSNorm(hidden_size);
		conv = bs::Conv1d(hidden_size, hidden_size, 3, 6144, 2048);  //나중에 한번 더 재확인
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

	
	void loadWeights(istream& file, const MetadataMap& metadata) override {
		MetadataMap conv_meta;
		MetadataMap ffn_meta;
		MetadataMap operator_norm_meta;
		MetadataMap ffn_norm_meta;

		for(auto [key, value]: metadata) {
			if (key.compare(0, 5, "conv.") == 0) {
				conv_meta[key.substr(5)] = value; // "conv." 제외
			} 
			else if (key.compare(0, 13, "feed_forward.") == 0) {
				ffn_meta[key.substr(13)] = value; // "feed_forward." 제외
			} // ... norm 등 ...
			else if (key.compare(0, 14, "operator_norm.") == 0) {
				operator_norm_meta[key.substr(14)] = value; // "operator_norm." 제외
			}
			else if (key.compare(0, 9, "ffn_norm.") == 0) {
				ffn_norm_meta[key.substr(9)] = value; // "ffn_norm." 제외
			}
		}
    
    conv.loadWeights(file, conv_meta);
    feed_forward.loadWeights(file, ffn_meta);
	operator_norm.loadWeights(file, operator_norm_meta);
	ffn_norm.loadWeights(file, ffn_norm_meta);
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
		x = self_attn.forward(x);
		x = add(residual, x);

		residual = x;
		x = feed_forward.forward(ffn_norm.forward(x));
		x = add(residual, x);

		return x;
	}

	
	void loadWeights(istream& file, const MetadataMap& metadata) override {
		MetadataMap self_attn_meta;
		MetadataMap ffn_meta;
		MetadataMap operator_norm_meta;
		MetadataMap ffn_norm_meta;

		for(auto [key, value]: metadata) {
			if (key.compare(0, 10, "self_attn.") == 0) {
				self_attn_meta[key.substr(10)] = value; // "self_attn." 제외
			} 
			else if (key.compare(0, 13, "feed_forward.") == 0) {
				ffn_meta[key.substr(13)] = value; // "feed_forward." 제외
			} // ... norm 등 ...
			else if (key.compare(0, 14, "operator_norm.") == 0) {
				operator_norm_meta[key.substr(14)] = value; // "operator_norm." 제외
			}
			else if (key.compare(0, 9, "ffn_norm.") == 0) {
				ffn_norm_meta[key.substr(9)] = value; // "ffn_norm." 제외
			}
		}
    
    self_attn.loadWeights(file, self_attn_meta);
    feed_forward.loadWeights(file, ffn_meta);
	operator_norm.loadWeights(file, operator_norm_meta);
	ffn_norm.loadWeights(file, ffn_norm_meta);
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
		//pe = bs::RoPE(config.hidden_size, config.max_position_embeddings);

		int i = 0;
		for (string type : config.layer_types) {
			//string name = "model.layers." + to_string(i);
			if (type == "conv")
			{
				layers.push_back(make_unique<ConvLayer>(config.hidden_size,
					config.intermediate_size));
			}
			else if (type == "full_attention")
			{
				layers.push_back(make_unique<AttentionLayer>(config.hidden_size,
					config.num_attention_heads, config.num_key_value_heads, config.intermediate_size));
			}
			i++;
		}
	}

	
	shared_ptr<bs::Variable> forward(shared_ptr<bs::Variable> x) 
	{
		
		shared_ptr<bs::Variable> embed = embedding.forward(x);

		//embedding norm(2048) -> (batch, token, 2048)			W(2048)
		embed = embednorm.forward(embed);

		//positional encoding -> (batch, token, 2048)			
		shared_ptr<bs::Variable> current = pe.forward(embed);

		//layers (batch, token, 2048)
		for (auto& layer : layers) {
			current = layer->forward(current);
			cout << endl;
		}

		//embedding  -> (batch, token, 65536)


		return current;
	}

	
	void load_weights(istream& file, MetadataMap metadata)
	{

        for (const auto& [key, meta] : metadata) {
    
            if (key.rfind("model.layers.", 0) == 0) {
                const std::size_t index_begin = 13;
                const std::size_t dot_pos = key.find('.', index_begin);
                if (dot_pos == std::string::npos || dot_pos <= index_begin) {
                    std::cerr << "[Model] 잘못된 레이어 키 형식: " << key << std::endl;
                    continue;
                }

                const std::string index_str = key.substr(index_begin, dot_pos - index_begin);
                int layer_idx = -1;
                try {
                    layer_idx = std::stoi(index_str);
                } catch (const std::exception& e) {
                    std::cerr << "[Model] 레이어 인덱스 파싱 실패(" << key << "): " << e.what() << std::endl;
                    continue;
                }

                if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
                    std::cerr << "[Model] 범위를 벗어난 레이어 인덱스 무시: " << key << std::endl;
                    continue;
                }

                const std::string child_key = key.substr(dot_pos + 1);
                if (child_key.empty()) {
                    std::cerr << "[Model] 하위 메타 키 누락: " << key << std::endl;
                    continue;
                }

                auto& layer_meta = weights_by_layer[layer_idx];
                layer_meta[child_key] = meta;
            } else {
                other_weights[key] = meta;
            }
        }
		cout<< "layer weight set test1" <<endl;

        for (std::size_t i = 0; i < layers.size(); ++i) {
            auto it = weights_by_layer.find(static_cast<int>(i));
            if (it == weights_by_layer.end()) {
                continue;
            }
            layers[i]->loadWeights(file, it->second);
        }
		cout<< "layer weight set test2" <<endl;
		
		for (auto& [key, value] : other_weights) {
			if( key.compare(0, 19, "model.embed_tokens.") == 0) {
				MetadataMap embed_meta;
				embed_meta[key.substr(19)] = value; // "model.embed_tokens." 제외
				embedding.loadWeights(file, embed_meta);
			}
			else if (key.compare(0, 21, "model.embedding_norm.")==0){
				MetadataMap embednorm_meta;
				embednorm_meta[key.substr(21)] = value; // "model.embedding_norm." 제외
				embednorm.loadWeights(file, embednorm_meta);
			} 
		}
		cout<<"all weights loaded"<<endl;
	

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
