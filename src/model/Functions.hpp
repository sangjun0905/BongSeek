#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include "Config.hpp"
#include "Tensor.hpp"
#include "json.hpp"

using namespace Eigen;

typedef vector<MatrixXd> Tensor;
struct Metadatainfo {
	size_t offset;
	size_t size_in_bytes;
	nb::Shape shape;
	std::string dtype;
};

using MetadataMap = map<string, Metadatainfo>;

//test용 프린트 함수	
//void testprint(string str);

Tensor add(Tensor x, Tensor y);

//행렬곱(가중치)
class Linear {
	Tensor weight;
	string name;


public:
	Linear() {};

	Linear(const string& prefix)
	{
		name = prefix;
	}

	void loadWeights(ifstream& file, MetadataMap& metadata)
	{
		string key = name + ".weight";
		if (metadata.count(key)) {
			auto data = metadata.at(key);
			//testprint((name+".weight"));
			/*
			// 1. 메타데이터를 기반으로 텐서 메모리 할당
            this->weight.shape = meta.shape;
            this->weight.allocate(meta.size_in_bytes);

            // 2. 파일 포인터를 가중치 위치로 이동 (seek)
            file.seekg(meta.offset);

            // 3. 파일에서 데이터를 직접 텐서의 메모리로 읽어옴 (read)
            file.read(this->weight.data.data(), meta.size_in_bytes);
			
			*/
			cout << key << endl;
		}	
	}

	Tensor forward(Tensor x)
	{
		return x;
	}

};

//임베딩
class Embedding{

	Linear embed_tokens;
	string name;

public:
	Embedding() {};

	Embedding(const string& prefix) {
		name = prefix;
		embed_tokens = Linear(prefix);
	}

	void loadWeights(ifstream& file, MetadataMap& metadata) {
		embed_tokens.loadWeights(file, metadata);
	}

	Tensor forward(Tensor x) {
		return x;
	}

};

//positional encoding
class Rope{
	int size;
	int maxlen;

public:
	Rope() {};

	Rope(int hidden_size, int max)
	{
		size = hidden_size;
		maxlen = max;
	}

	Tensor forward(Tensor x)
	{
		//testprint("Rope Positional Encoding");
		cout << "Rope Positional Encoding" << endl;
		return x;
	}
};

//정규화
class RMSNormal
{
	string name;
	int hiddensize;
	double eps;
	Linear normweight; //Gain(γ) -> scale 파라미터
	
public:
	RMSNorm() {};

	RMSNorm(const string& prefix, int size, double eps)
	{
		this->name = prefix;
		this->hiddensize = size;
		this->eps = eps;
		normweight = Linear(prefix);
	};

	void loadWeights(ifstream& file, MetadataMap metadata) {
		normweight.loadWeights(file, metadata);
	};

	Tensor forward(Tensor x)
	{
		//testprint("Normalization");
		cout << "Normalization" << endl;
		/*정규화*/
		return x;
	};

	

};

//key, value를 묶어서 하는 GQA
class GroupedQueryAttention {
	//GroupedQueryAttention();
	string name;
	int hidden_size;
	int num_attention_heads;
	int num_key_value_heads;
	int d_k;

	
	Linear k_proj;
	Linear q_proj;
	Linear v_proj;
	RMSNorm k_layernorm;
	RMSNorm q_layernorm;
	Linear out_proj;

	//kv캐시
	Tensor Kcache;
	Tensor Vcache;
	void casheupdate(Tensor& Knew, Tensor& Vnew)
	{
		//기존 kv 캐시에 새로운 캐시 추가
	}


public:
	GroupedQueryAttention() {};

	GroupedQueryAttention(const string& prefix, int size, int attn_heads, int kv_heads, double eps){
		this->name = prefix;
		hidden_size = size;
		num_attention_heads = attn_heads;
		num_key_value_heads = kv_heads;
		d_k = size / attn_heads;
		
		k_proj = Linear(name + ".k_proj");
		q_proj = Linear(name + ".q_proj");
		v_proj = Linear(name + ".v_proj");
		k_layernorm = RMSNorm(name + ".k_layernorm", d_k, eps);
		q_layernorm = RMSNorm(name + ".q_layernorm", d_k, eps);
		out_proj = Linear(name + ".out_proj");
	}

	void loadWeights(ifstream& file, MetadataMap metadata)  {
		k_proj.loadWeights(file, metadata);
		q_proj.loadWeights(file, metadata);
		v_proj.loadWeights(file, metadata);
		k_layernorm.loadWeights(file, metadata);
		q_layernorm.loadWeights(file, metadata);
		out_proj.loadWeights(file, metadata);

	};

	Tensor forward(Tensor Q, Tensor K, Tensor V, Tensor mask) {
		//testprint("Grouped Query Attention");
		cout << "Grouped Query Attention" << endl;
		return Q;
	};

	
};

//residual connection
class sublayerconnection{ };

//Swiglu 활성화함수를 쓰는 feed forward
class SwigluFeedforward {
	/*feed forward
	 w1 -> (10752, 2048) 입력벡터(2048)를 중간차원(10752)으로 확장
	 w2 -> (2048, 10752) 중간 결과(10752)를 다시 모델의 hidden_size(2048)로
	 w3 -> (10752, 2048) swiglu 연산을 위해 쓰임(입력을 w1과 병렬로 처리)
	*/
	Linear w1;
	Linear w2;
	Linear w3;

	string name;
	int hidden_size;
	int intermediate_size;

public:
	SwigluFeedforward() {};

	SwigluFeedforward(const string& prefix, int hidden_size, int intermediate_size)
	{
		this->name = prefix;
		this->hidden_size = hidden_size;
		this->intermediate_size = intermediate_size;  //feedforward 중간 은닉층 크기
		w1 = Linear(prefix+ ".w1");
		w2 = Linear(prefix+ ".w2");
		w3 = Linear(prefix+ ".w3");
	}
	
	void loadWeights(ifstream& file, MetadataMap metadata) {
		//가중치
		w1.loadWeights(file, metadata);
		w2.loadWeights(file, metadata);
		w3.loadWeights(file, metadata);
	}

	Tensor forward(Tensor x)
	{
		//testprint("Swiglu Feed Forward");
		cout << "Swiglu Feed Forward" << endl;
		return x;
	}

	
};

//1d convolution 
class Convolution {

	/*convlayer
	(6144, 2048) -> conv 전에 2048 입력 벡터를 6144로 투영projection(2048*3 =6144)
	(2048, 1, 3) -> 1d depth convolution(채널 수, 1, 커널 크기) 채널 수*커널 크기 =6144
	(2048, 2048) -> conv 연산을 통해 처리된 결과를 다시 한번 조합하고 정리하여 최종 출력(2048)로 만드는 투영
	*/
	Linear in_proj;
	Linear conv;
	Linear out_proj;
	string name;
	int hidden_size;


public:
	Convolution() {};

	Convolution(const string& prefix, int size)
	{
		name = prefix;
		hidden_size = size;
		in_proj = Linear(name + ".in_proj");
		conv = Linear(name + ".conv");
		out_proj = Linear(name + ".out_proj");
	}

	void loadWeights(ifstream& file, MetadataMap metadata){
		in_proj.loadWeights(file, metadata);
		conv.loadWeights(file, metadata);
		out_proj.loadWeights(file, metadata);
	}

	Tensor forward(Tensor x) 
	{
		//testprint("Convolution");
		cout << "Convolution" << endl;
		return x;
	}
};


class Generator {

	//마지막 변환 2048 -> 65536
	//첫 embedding 가중치 재활용
};
