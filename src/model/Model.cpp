#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include "../BongTorch/Module.hpp"
#include "Functions.hpp"
#include "ModelConfig.hpp"
#include "../BongTorch/GQAAttention.hpp"
#include "../BongTorch/RMSNorm.hpp"
#include "../BongTorch/FFN_SWiGLU.hpp"
#include "../BongTorch/Conv1d.hpp"
#include "../BongTorch/Embedding.hpp"
#include "../BongTorch/RoPE.hpp"


using namespace std;
using namespace Eigen;

typedef shared_ptr<bs::Variable> datatype;

class Layer {
	//AttentionLayer, ConvLayer �θ� Ŭ����(���)
public:
	virtual ~Layer() = default;
	virtual Tensor forward(Tensor x)
	{
		return x;
	};
	virtual void loadWeights(ifstream& file, MetadataMap metadata) = 0;

};



class ConvLayer : public Layer {  //Layer�� ���Ƿ� �����س��� �θ� Ŭ����(���߿� ���� ����)
	/*
	�ϳ��� conv ���̾ �ʿ��� �͵� ����
	�Է�->*/
	//operator_norm			(����ġ ���� : (2048)) �ٽ� ����(conv, attn)���� �����ϴ� ����ȭ
	/*convlayer
	(6144, 2048) -> conv ���� 2048 �Է� ���͸� 6144�� ����projection(2048*3 =6144)
	(2048, 1, 3) -> 1d depth convolution(ä�� ��, 1, Ŀ�� ũ��) ä�� ��*Ŀ�� ũ�� =6144
	(2048, 2048) -> conv ������ ���� ó���� ����� �ٽ� �ѹ� �����ϰ� �����Ͽ� ���� ���(2048)�� ����� ����
	*/
	//residual connection
	//ffn norm -> (2048) feed forward ���꿡 ���� ���� ����Ǵ� ����ȭ
	/*feed forward
	 w1 -> (10752, 2048) �Էº���(2048)�� �߰�����(10752)���� Ȯ��
	 w2 -> (2048, 10752) �߰� ���(10752)�� �ٽ� ���� hidden_size(2048)��
	 w3 -> (10752, 2048) swiglu ������ ���� ����(�Է��� w1�� ���ķ� ó��)
	*/

	//residual connectionSS
	//output -> next layer

	string name;
	bs::RMSNorm operation_norm;
	bs::Conv1DLayer conv;
	bs::RMSNorm ffn_norm;
	bs::FFN_SWiGLU feed_forward;



public:

	ConvLayer(const string& name_prefix,
		int hidden_size,
		int intermediate_size)
	{
		name = name_prefix;
		operation_norm = bs::RMSNorm(name_prefix + ".operator_norm", hidden_size);
		conv = bs::Conv1DLayer(name_prefix + ".conv", hidden_size, hidden_size, 3);  //��Ȯ�� �ʿ�
		ffn_norm = bs::RMSNorm(name_prefix + ".ffn_norm", hidden_size);
		feed_forward = bs::FFN_SWiGLU(name_prefix + ".feed_forward", hidden_size, intermediate_size);
	}

	datatype forward(datatype x) override {
		datatype residual = x;
		x = operation_norm.forward(x);
		x = conv.forward(x);
		x = add(residual, x);

		residual = x;
		x = feed_forward.forward(ffn_norm.forward(x));
		x = add(residual, x);

		return x;
	}

	/*
	void loadWeights(ifstream& file, MetadataMap metadata) override {
		operation_norm.loadWeights(file, metadata);
		conv.loadWeights(file, metadata);
		ffn_norm.loadWeights(file, metadata);
		feed_forward.loadWeights(file, metadata);
	}*/

};

class AttentionLayer : public Layer {

	/*
	�ϳ��� conv ���̾ �ʿ��� �͵� ����
	�Է�->*/
	//operator_norm  (2048)

	/*self - attention
	k_layernorm.weight  (64)
	q_layernorm.weight  (64)
	key�� query�� ���� ���� �� ��� ������ ����ȭ(RMSnorm) �� ����� ������ 64 (�� ��忡 ���������� ����ȭ)

	k_proj.weight (512, 2048)
	v_proj.weight (512, 2048)
	�Է� ����(2048)�� key ,value ���ͷ� ��ȯ key, value�� head�� 8���̹Ƿ� 64*8->512

	q_proj.weight (2048, 2048)
	�Է� ����(2048)�� query ���ͷ� ��ȯ query head�� 32�� �̹Ƿ� 64*32 -> 2048

	out_proj.weight	(2048, 2048)
	32���� attention head ��� ����� ��ħ

	*/
	//residual connection
	//ffn norm -> (2048) feed forward ���꿡 ���� ���� ����Ǵ� ����ȭ
	/*feed forward
	 w1 -> (10752, 2048) �Էº���(2048)�� �߰�����(10752)���� Ȯ��
	 w2 -> (2048, 10752) �߰� ���(10752)�� �ٽ� ���� hidden_size(2048)��
	 w3 -> (10752, 2048) swiglu ������ ���� ����(�Է��� w1�� ���ķ� ó��)
	*/
	//residual connection
	//output -> next layer


	//kvĳ�� -> ���� ������ ���� �������� ������	 ������ �Է� ���� �� �Է� ���� ���� ���� �߷п��� �̹� �ߴ� ����� �ٽ��ؾ���
	//-> kvĳ�ø� �ξ� kv��갪 ��Ȱ�� -> kvĳ�ô� �� ��ū�� �߷��� �� ���� ����� -> �߷��ϸ鼭 ������Ʈ

	string name;

	bs::RMSNorm operation_norm;
	bs::GQAAttention self_attn;
	bs::RMSNorm ffn_norm;
	bs::FFN_SWiGLU feed_forward;




public:



	AttentionLayer(const string& name_prefix,
		int hidden_size,
		int num_attention_heads,
		int num_key_value_heads,
		int intermediate_size)
	{
		int head_dim = hidden_size / num_attention_heads;
		name = name_prefix;
		operation_norm = bs::RMSNorm(name_prefix + ".operator_norm", hidden_size);
		self_attn = bs::GQAAttention(name_prefix + ".self_attn", hidden_size, num_attention_heads, num_key_value_heads, head_dim);
		ffn_norm = bs::RMSNorm(name_prefix + ".ffn_norm", hidden_size);
		feed_forward = bs::FFN_SWiGLU(name_prefix + ".feed_forward", hidden_size, intermediate_size);
	}

	datatype forward(datatype x) override {
		datatype residual = x;
		x = operation_norm.forward(x);
		x = self_attn.forward(x, x, x, x);
		x = add(residual, x);

		residual = x;
		x = feed_forward.forward(ffn_norm.forward(x));
		x = add(residual, x);

		return x;
	}

	/*
	void loadWeights(ifstream& file, MetadataMap metadata) override {
		operation_norm.loadWeights(file, metadata);
		self_attn.loadWeights(file, metadata);
		ffn_norm.loadWeights(file, metadata);
		feed_forward.loadWeights(file, metadata);
	}*/

};


class BongSeek {

	vector<unique_ptr<Layer>> layers;
	bs::Embedding embedding;
	bs::RMSNorm embednorm;
	Rope pe;
public:

	BongSeek(Config config) {
		//transformer������ �Է����� encoder�� decoder �ν��Ͻ��� ������, 
		//lfm2�� 30���� layer�� �������� �� �������־ �ϴ� ���������� ���������� ����

		embedding = bs::Embedding("model.embed_tokens", config.vocab_size, config.hidden_size);
		embednorm = bs::RMSNorm("model.embedding_norm", config.hidden_size);
		pe = Rope(config.hidden_size, config.max_position_embeddings);

		int i = 0;
		for (string type : config.layer_types) {
			string name = "model.layers." + to_string(i);
			if (type == "conv")
			{
				layers.push_back(make_unique<ConvLayer>(name, config.hidden_size,
					config.intermediate_size, config.norm_eps));
			}
			else if (type == "full_attention")
			{
				layers.push_back(make_unique<AttentionLayer>(name, config.hidden_size,
					config.num_attention_heads, config.num_key_value_heads, config.intermediate_size));
			}
			i++;
		}
	}

	Tensor forward(Tensor x) //input = (��ġũ��, ��ū����)
	{
		//embedding -> (��ġũ��, ��ū����, 2048) ���			W(65536, 2048)
		//(65536, 2048)���� ��ū idx�� �ش��ϴ� (2048)ũ���� ���� �̾ƿ� 
		Tensor embed = embedding.forward(x);

		//embedding norm(2048) -> (batch, token, 2048)			W(2048)
		embed = embednorm.forward(embed);

		//positional encoding -> (batch, token, 2048)			������ ����ġ ����

		Tensor current = pe.forward(embed);

		//layers ������� ������ || �Է� ����� (batch, token, 2048)
		//transformer������ N���� encoder decoder�� ���������, lfm2�� �ٸ� ����
		//-> encoder, decoder class ��ſ� attention, conv, feedforward���� ������ convlayer�� attention layer�� ����� ���

		cout << endl;

		for (auto& layer : layers) {
			current = layer->forward(current);
			cout << endl;
		}

		//embedding ����ġ ��Ȱ�� -> (batch, token, 65536)


		return current;
	}

	/*
	void load_weights(ifstream& file, MetadataMap metadata)
	{
		//safetensors �о�� ����ġ ����

		//���ǻ���: �� layer�� ����ġ�� ���� ������ ������ �ΰų� ����ġ�� ������ �� �ִ� �Լ��� �������־�� ��
		embedding.loadWeights(file, metadata);
		embednorm.loadWeights(file, metadata);
		for (auto& layer : layers)
		{
			layer->loadWeights(file, metadata);
		}
	}*/

};


BongSeek makeModel()
{
	//config �о���� (���� ���Ƿ� ����)
	MModelConfig config("model_dir");

	//model ��ü�� ���� return;
	BongSeek Model(config);

	ifstream file("text.txt");

	//safetensors �а� map ����� (���� �׽�Ʈ�� ����)

	MetadataMap weights;
	Metadatainfo x{ 10, 10, {1, 1} };


	weights.insert({ "model.embed_tokens.weight", x });
	weights.insert({ "model.embedding_norm.weight", x });

	weights.insert({ "model.layers.0.conv.conv.weight", x });
	weights.insert({ "model.layers.0.conv.in_proj.weight", x });
	weights.insert({ "model.layers.0.conv.out_proj.weight", x });
	weights.insert({ "model.layers.0.feed_forward.w1.weight", x });
	weights.insert({ "model.layers.0.feed_forward.w2.weight", x });
	weights.insert({ "model.layers.0.feed_forward.w3.weight", x });
	weights.insert({ "model.layers.0.ffn_norm.weight", x });
	weights.insert({ "model.layers.0.operator_norm.weight", x });

	weights.insert({ "model.layers.1.conv.conv.weight", x });
	weights.insert({ "model.layers.1.conv.in_proj.weight", x });
	weights.insert({ "model.layers.1.conv.out_proj.weight", x });
	weights.insert({ "model.layers.1.feed_forward.w1.weight", x });
	weights.insert({ "model.layers.1.feed_forward.w2.weight", x });
	weights.insert({ "model.layers.1.feed_forward.w3.weight", x });
	weights.insert({ "model.layers.1.ffn_norm.weight", x });
	weights.insert({ "model.layers.1.operator_norm.weight", x });

	weights.insert({ "model.layers.2.feed_forward.w1.weight", x });
	weights.insert({ "model.layers.2.feed_forward.w2.weight", x });
	weights.insert({ "model.layers.2.feed_forward.w3.weight", x });
	weights.insert({ "model.layers.2.ffn_norm.weight", x });
	weights.insert({ "model.layers.2.operator_norm.weight", x });
	weights.insert({ "model.layers.2.self_attn.k_layernorm.weight", x });
	weights.insert({ "model.layers.2.self_attn.k_proj.weight", x });
	weights.insert({ "model.layers.2.self_attn.out_proj.weight", x });
	weights.insert({ "model.layers.2.self_attn.q_layernorm.weight", x });
	weights.insert({ "model.layers.2.self_attn.q_proj.weight", x });
	weights.insert({ "model.layers.2.self_attn.v_proj.weight", x });

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
}