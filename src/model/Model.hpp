#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include "../BongTorch/Module.hpp"
#include "Functions.hpp"
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
	//AttentionLayer, ConvLayer 부모 클래스(상속)
public:
	virtual ~Layer() = default;
	virtual Tensor forward(Tensor x)
	{
		return x;
	};
	virtual void loadWeights(ifstream& file, MetadataMap metadata) = 0;

};



class ConvLayer : public Layer {  //Layer은 임의로 설정해놓은 부모 클래스(나중에 맞춰 수정)
	/*
	하나의 conv 레이어에 필요한 것들 조립
	입력->*/
	//operator_norm			(가중치 차원 : (2048)) 핵심 연산(conv, attn)전에 수행하는 정규화
	/*convlayer
	(6144, 2048) -> conv 전에 2048 입력 벡터를 6144로 투영projection(2048*3 =6144)
	(2048, 1, 3) -> 1d depth convolution(채널 수, 1, 커널 크기) 채널 수*커널 크기 =6144
	(2048, 2048) -> conv 연산을 통해 처리된 결과를 다시 한번 조합하고 정리하여 최종 출력(2048)로 만드는 투영
	*/
	//residual connection
	//ffn norm -> (2048) feed forward 연산에 들어가기 전에 적용되는 정규화
	/*feed forward
	 w1 -> (10752, 2048) 입력벡터(2048)를 중간차원(10752)으로 확장
	 w2 -> (2048, 10752) 중간 결과(10752)를 다시 모델의 hidden_size(2048)로
	 w3 -> (10752, 2048) swiglu 연산을 위해 쓰임(입력을 w1과 병렬로 처리)
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
		conv = bs::Conv1DLayer(name_prefix + ".conv", hidden_size, hidden_size, 3);  //재확인 필요
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
	하나의 conv 레이어에 필요한 것들 조립
	입력->*/
	//operator_norm  (2048)

	/*self - attention
	k_layernorm.weight  (64)
	q_layernorm.weight  (64)
	key와 query를 만든 직후 각 헤드 단위로 정규화(RMSnorm) 각 헤드의 차원은 64 (각 헤드에 독립적으로 정규화)

	k_proj.weight (512, 2048)
	v_proj.weight (512, 2048)
	입력 벡터(2048)를 key ,value 벡터로 변환 key, value의 head가 8개이므로 64*8->512

	q_proj.weight (2048, 2048)
	입력 벡터(2048)를 query 벡터로 변환 query head는 32개 이므로 64*32 -> 2048

	out_proj.weight	(2048, 2048)
	32개의 attention head 계산 결과를 합침

	*/
	//residual connection
	//ffn norm -> (2048) feed forward 연산에 들어가기 전에 적용되는 정규화
	/*feed forward
	 w1 -> (10752, 2048) 입력벡터(2048)를 중간차원(10752)으로 확장
	 w2 -> (2048, 10752) 중간 결과(10752)를 다시 모델의 hidden_size(2048)로
	 w3 -> (10752, 2048) swiglu 연산을 위해 쓰임(입력을 w1과 병렬로 처리)
	*/
	//residual connection
	//output -> next layer


	//kv캐시 -> 다음 예측을 위해 이전까지 예측한	 문장을 입력 받을 때 입력 값에 대해 이전 추론에서 이미 했던 계산을 다시해야함
	//-> kv캐시를 두어 kv계산값 재활용 -> kv캐시는 한 토큰을 추론할 때 마다 길어짐 -> 추론하면서 업데이트

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
		//transformer에서는 입력으로 encoder와 decoder 인스턴스를 받지만, 
		//lfm2는 30개의 layer가 순서까지 다 정해져있어서 일단 내부적으로 정해지도록 설정

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

	Tensor forward(Tensor x) //input = (배치크기, 토큰개수)
	{
		//embedding -> (배치크기, 토큰개수, 2048) 출력			W(65536, 2048)
		//(65536, 2048)에서 토큰 idx에 해당하는 (2048)크기의 열을 뽑아옴 
		Tensor embed = embedding.forward(x);

		//embedding norm(2048) -> (batch, token, 2048)			W(2048)
		embed = embednorm.forward(embed);

		//positional encoding -> (batch, token, 2048)			가져올 가중치 없음

		Tensor current = pe.forward(embed);

		//layers 순서대로 순전파 || 입력 출력은 (batch, token, 2048)
		//transformer에서는 N개의 encoder decoder를 사용했지만, lfm2는 다른 구조
		//-> encoder, decoder class 대신에 attention, conv, feedforward등을 조합한 convlayer와 attention layer를 만들어 사용

		cout << endl;

		for (auto& layer : layers) {
			current = layer->forward(current);
			cout << endl;
		}

		//embedding 가중치 재활용 -> (batch, token, 65536)


		return current;
	}

	/*
	void load_weights(ifstream& file, MetadataMap metadata)
	{
		//safetensors 읽어와 가중치 설정

		//유의사항: 각 layer에 가중치를 접근 가능한 변수로 두거나 가중치를 설정할 수 있는 함수를 정의해주어야 함
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
	//config 읽어오기 (대충 임의로 만듬)
	Config config;

	//model 객체를 만들어서 return;
	BongSeek Model(config);

	ifstream file("text.txt");

	//safetensors 읽고 map 만들기 (대충 테스트용 만듬)

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

	//만든 model객체 load_weights로 학습을 대신해 가중치 설정


	//Model.load_weights(file, weights);
	return Model;
}

//추론 방식(결과값중 가장 큰 값 선택)
Tensor greedy_Decode(BongSeek model, Tensor x, Tensor x_mask, int max_len, int start_symbol)
{

	//결과값 greedy_decode(65536의 확률 분포에서 가장 큰 값 뽑아
	// 옴)
	//->생성할 sequence의 최대 길이만큼 순환하면서 입력으로 이전까지의 예측값을 넣어주면서 
	//새로운 예측값을s 추가해준다. -> 입력 길이가 하나씩 늘어나는 구조

	return x;  //정답 예측 tensor (배치크기, 토큰 개수)
}