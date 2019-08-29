Welcome file
Welcome file
<h2 align="center">embedding-as-service</h2>
<p align="center">One-Stop Solution to encode sentence to fixed length vectors from various embedding techniques 
<br>• Inspired from <a href="[https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)"> bert-as-service</a> </p>
<p align="center">
  <a href="https://github.com/amansrivastava17/embedding-as-service/stargazers">
    <img src="https://img.shields.io/github/stars/amansrivastava17/embedding-as-service.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://pypi.org/project/embedding-as-service/">
      <img src="https://img.shields.io/pypi/v/embedding-as-service?colorB=brightgreen" alt="Pypi package">
    </a>
  <a href="https://pypi.org/project/embedding-as-service/">
      <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/embedding-as-service">
  </a>
  <a href="https://github.com/amansrivastava/embedding-as-service/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/amansrivastava17/embedding-as-service.svg"
             alt="GitHub license">
  </a>
</p>

<p align="center">
 <a href="#what-is-it">What is it</a> •
  <a href="#install">Install</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#supported-models">Supported Embeddings</a> •
  <a href="#server-and-client-api">API</a> •
  <a href="#book-tutorial">Tutorials</a>   
</p>

<h2 align="center">What is it ?</h3>

**Encoding/Embedding**  is a upstream task of encoding any inputs in the form of text, image, audio, video, transactional data to fixed length vector. Embeddings are quite popular in the field of NLP, there has been various Embeddings models being proposed in recent years by researchers, some of the famous one are bert, xlnet, word2vec etc. The goal of this repo is to build one stop solution for all embeddings techniques available, here we are starting with popular text embeddings for now and later on we aim  to add as much technique for image, audio, video input also.

**Finally**, **`embedding-as-service`** help you to encode any given text to fixed length vector from supported embeddings and models.

<h2 align="center">Installation</h2>

Install the embedding-as-servive via `pip`. 
```bash
pip install embedding-as-service 
```
Note that the code MUST be running on **Python >= 3.6** with **Tensorflow >= 1.10** (_one-point-ten_). Again, this module does not support Python 2!

<h2 align="center">Getting Started</h2>

### Using Embeddings

1. **Intialise embedding** method and models from supported models from <a href="#pooling strategy">here</a>
```python
from embedding_as_service.text.encode import Encoder
# download model and intialise the model
>>> en = Encoder(embedding='xlnet', model='xlnet_base_cased', download=True)
```
2. Call encode class with list of sentences to get **tokens embedding**
```python 
>>> vector = en.encode(texts=['hello aman', 'how are you?'])
array([[[ 1.7049843 ,  0.        ,  1.3486509 , ..., -1.3647075 ,
          0.6958289 ,  1.8013777 ],
        [ 1.6232326 ,  0.07614848, -0.16540718, ..., -0.28085896,
          0.        ,  0.23663947],
        [ 1.5703819 ,  0.15056366,  0.3749014 , ...,  0.5040764 ,
          1.5577477 ,  0.23442714],
        ...,
        [ 2.7379205 ,  0.128727  ,  0.6431941 , ...,  0.0298907 ,
          0.        ,  0.6528186 ],
        [ 0.49853373, -0.57566786, -0.8758549 , ...,  0.6236979 ,
          0.        ,  0.03040635],
        [ 0.4913215 ,  0.60877025,  0.73050433, ..., -0.64490885,
          0.8525057 ,  0.3080206 ]]], dtype=float32)
>>> vector.shape
(2, 128, 768)
```
3. To get **sentence embeddings** using pooled method using one of **pooling strategy** available <a href="#pooling strategy">here</a>
```python
>>> vector = en.encode(texts=['hello aman', 'how are you?'], pooling='mean')
array([[-0.33547154,  0.34566957,  1.1954105 , ...,  0.33702594,
         1.0317835 , -0.785943  ],
       [-0.3439088 ,  0.36881036,  1.0612687 , ...,  0.28851607,
         1.1107115 , -0.6253736 ]], dtype=float32)
>>> vector.shape
(2, 768)
```
4. Default `max_seq_length` is `128` , to assign **custom `max_seq_length`** as param
```python
>>> vectors = en.encode(texts=['hello aman', 'how are you?'], max_seq_length=256)
array([[ 0.48388457, -0.01327741, -0.76577514, ..., -0.54265064,
        -0.5564591 ,  0.6454179 ],
       [ 0.53209245,  0.00526248, -0.71091074, ..., -0.5171917 ,
        -0.40458363,  0.6779779 ]], dtype=float32)
>>> vectors.shape
(2, 256, 768)
```
### Using Tokenizer

### Check Embedding Meta



<h2 align="center" href="#supported-models">Supported Embeddings and Models</h2>

Here are the list of supported embeddings and their respective models.

|Embedding  | Model  | Embedding dimensions | 
|:--|:--:|:--:|
|`xlnet`  |`xlnet_large_cased`  | 1024| 
|`xlnet`  |`xlnet_base_cased`  | 768| 
|`bert`  |`bert_base_uncased`  | 768| 
|`bert`  |`bert_base_cased`  | 768| 
|`bert`  |`bert_multi_cased` | 768| 
|`bert`  |`bert_large_uncased`  | 1024| 
|`bert`  |`bert_large_cased`  | 1024| 

embedding-as-service
One-Stop Solution to encode sentence to fixed length vectors from various embedding techniques 
• Inspired from bert-as-service

 GitHub stars  Pypi package  PyPI - Downloads  GitHub license

What is it • Install • Getting Started • Supported Embeddings • API • Tutorials

What is it ?
Encoding/Embedding is a upstream task of encoding any inputs in the form of text, image, audio, video, transactional data to fixed length vector. Embeddings are quite popular in the field of NLP, there has been various Embeddings models being proposed in recent years by researchers, some of the famous one are bert, xlnet, word2vec etc. The goal of this repo is to build one stop solution for all embeddings techniques available, here we are starting with popular text embeddings for now and later on we aim to add as much technique for image, audio, video input also.

Finally, embedding-as-service help you to encode any given text to fixed length vector from supported embeddings and models.

Installation
Install the embedding-as-servive via pip.

pip install embedding-as-service 
Note that the code MUST be running on Python >= 3.6 with Tensorflow >= 1.10 (one-point-ten). Again, this module does not support Python 2!

Getting Started
Using Embeddings
Intialise embedding method and models from supported models from here
from embedding_as_service.text.encode import Encoder
# download model and intialise the model
>>> en = Encoder(embedding='xlnet', model='xlnet_base_cased', download=True)
Call encode class with list of sentences to get tokens embedding
>>> vector = en.encode(texts=['hello aman', 'how are you?'])
array([[[ 1.7049843 ,  0.        ,  1.3486509 , ..., -1.3647075 ,
          0.6958289 ,  1.8013777 ],
        [ 1.6232326 ,  0.07614848, -0.16540718, ..., -0.28085896,
          0.        ,  0.23663947],
        [ 1.5703819 ,  0.15056366,  0.3749014 , ...,  0.5040764 ,
          1.5577477 ,  0.23442714],
        ...,
        [ 2.7379205 ,  0.128727  ,  0.6431941 , ...,  0.0298907 ,
          0.        ,  0.6528186 ],
        [ 0.49853373, -0.57566786, -0.8758549 , ...,  0.6236979 ,
          0.        ,  0.03040635],
        [ 0.4913215 ,  0.60877025,  0.73050433, ..., -0.64490885,
          0.8525057 ,  0.3080206 ]]], dtype=float32)
>>> vector.shape
(2, 128, 768)
To get sentence embeddings using pooled method using one of pooling strategy available here
>>> vector = en.encode(texts=['hello aman', 'how are you?'], pooling='mean')
array([[-0.33547154,  0.34566957,  1.1954105 , ...,  0.33702594,
         1.0317835 , -0.785943  ],
       [-0.3439088 ,  0.36881036,  1.0612687 , ...,  0.28851607,
         1.1107115 , -0.6253736 ]], dtype=float32)
>>> vector.shape
(2, 768)
Default max_seq_length is 128 , to assign custom max_seq_length as param
>>> vectors = en.encode(texts=['hello aman', 'how are you?'], max_seq_length=256)
array([[ 0.48388457, -0.01327741, -0.76577514, ..., -0.54265064,
        -0.5564591 ,  0.6454179 ],
       [ 0.53209245,  0.00526248, -0.71091074, ..., -0.5171917 ,
        -0.40458363,  0.6779779 ]], dtype=float32)
>>> vectors.shape
(2, 256, 768)
Using Tokenizer
Check Embedding Meta
Supported Embeddings and Models
Here are the list of supported embeddings and their respective models.

Embedding	Model	Embedding dimensions
xlnet	xlnet_large_cased	1024
xlnet	xlnet_base_cased	768
bert	bert_base_uncased	768
bert	bert_base_cased	768
bert	bert_multi_cased	768
bert	bert_large_uncased	1024
bert	bert_large_cased	1024
Markdown 5087 bytes 537 words 112 lines Ln 104, Col 11 HTML 2750 characters 454 words 83 paragraphs