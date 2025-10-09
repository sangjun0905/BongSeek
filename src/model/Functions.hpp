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

//test�� ����Ʈ �Լ�	
//void testprint(string str);

Tensor add(Tensor x, Tensor y);

//��İ�(����ġ)
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
			// 1. ��Ÿ�����͸� ������� �ټ� �޸� �Ҵ�
            this->weight.shape = meta.shape;
            this->weight.allocate(meta.size_in_bytes);

            // 2. ���� �����͸� ����ġ ��ġ�� �̵� (seek)
            file.seekg(meta.offset);

            // 3. ���Ͽ��� �����͸� ���� �ټ��� �޸𸮷� �о�� (read)
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

//�Ӻ���
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

//����ȭ
class RMSNormal
{
	string name;
	int hiddensize;
	double eps;
	Linear normweight; //Gain(��) -> scale �Ķ����
	
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
		/*����ȭ*/
		return x;
	};

	

};

//key, value�� ��� �ϴ� GQA
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

	//kvĳ��
	Tensor Kcache;
	Tensor Vcache;
	void casheupdate(Tensor& Knew, Tensor& Vnew)
	{
		//���� kv ĳ�ÿ� ���ο� ĳ�� �߰�
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

//Swiglu Ȱ��ȭ�Լ��� ���� feed forward
class SwigluFeedforward {
	/*feed forward
	 w1 -> (10752, 2048) �Էº���(2048)�� �߰�����(10752)���� Ȯ��
	 w2 -> (2048, 10752) �߰� ���(10752)�� �ٽ� ���� hidden_size(2048)��
	 w3 -> (10752, 2048) swiglu ������ ���� ����(�Է��� w1�� ���ķ� ó��)
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
		this->intermediate_size = intermediate_size;  //feedforward �߰� ������ ũ��
		w1 = Linear(prefix+ ".w1");
		w2 = Linear(prefix+ ".w2");
		w3 = Linear(prefix+ ".w3");
	}
	
	void loadWeights(ifstream& file, MetadataMap metadata) {
		//����ġ
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
	(6144, 2048) -> conv ���� 2048 �Է� ���͸� 6144�� ����projection(2048*3 =6144)
	(2048, 1, 3) -> 1d depth convolution(ä�� ��, 1, Ŀ�� ũ��) ä�� ��*Ŀ�� ũ�� =6144
	(2048, 2048) -> conv ������ ���� ó���� ����� �ٽ� �ѹ� �����ϰ� �����Ͽ� ���� ���(2048)�� ����� ����
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

	//������ ��ȯ 2048 -> 65536
	//ù embedding ����ġ ��Ȱ��
};
