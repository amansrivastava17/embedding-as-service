<h1 align="center">embedding-as-service</h1>  
<p align="center">One-Stop Solution to encode sentence to fixed length vectors from various embedding techniques   
<br>• Inspired from <a href="https://github.com/hanxiao/bert-as-service"> bert-as-service</a> </p>  
<p align="center">  
  <a href="https://github.com/amansrivastava17/embedding-as-service/stargazers">  
    <img src="https://img.shields.io/github/stars/amansrivastava17/embedding-as-service.svg?colorA=orange&colorB=orange&logo=github"  
         alt="GitHub stars">  
  </a> 
  <a href="https://pepy.tech/project/embedding-as-service/">  
      <img src="https://pepy.tech/badge/embedding-as-service" alt="Downloads">  
  </a>   
  <a href="https://pypi.org/project/embedding-as-service/">  
      <img src="https://img.shields.io/pypi/v/embedding-as-service?colorB=brightgreen" alt="Pypi package">  
  </a>  
  <a href="https://github.com/amansrivastava17/embedding-as-service/issues">
        <img src="https://img.shields.io/github/issues/amansrivastava17/embedding-as-service.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/amansrivastava17/embedding-as-service/blob/master/LICENSE">  
        <img src="https://img.shields.io/github/license/amansrivastava17/embedding-as-service.svg"  
             alt="GitHub license">  
  </a>
  <a href="https://github.com/amansrivastava17/embedding-as-service/graphs/contributors">  
        <img src="https://img.shields.io/badge/all_contributors-5-blue.svg"  
             alt="Contributors">  
  </a>  
</p>  
  
<p align="center">  
 <a href="#what-is-it">What is it</a> •  
  <a href="#-installation">Installation</a> •  
  <a href="#-️getting-started">Getting Started</a> •  
  <a href="#-supported-embeddings-and-models">Supported Embeddings</a> •  
  <a href="#-api-">API</a> •   
</p>  
</p>

<p align="center">
    <img src=".github/demo.gif?raw=true" width="670", height="350">
</p>
  
<h2 align="center">What is it</h3>  
  
**Encoding/Embedding** is a upstream task of encoding any inputs in the form of text, image, audio, video, transactional data to fixed length vector. Embeddings are quite popular in the field of NLP, there has been various Embeddings models being proposed in recent years by researchers, some of the famous one are bert, xlnet, word2vec etc. The goal of this repo is to build one stop solution for all embeddings techniques available, here we are starting with popular text embeddings for now and later on we aim  to add as much technique for image, audio, video inputs also.  
  
**Finally**, **`embedding-as-service`** help you to encode any given text to fixed length vector from supported embeddings and models.  
  
<h2 align="center">💾 Installation</h2>  
<p align="right"><a href="#embedding-as-service"><sup>▴ Back to top</sup></a></p>
  
  
Install the embedding-as-servive via `pip`.   
```bash  
$ pip install embedding-as-service
```  
> Note that the code MUST be running on **Python >= 3.6**. Again module does not support Python 2!  
  
<h2 align="center">⚡ ️Getting Started</h2> 
<p align="right"><a href="#embedding-as-service"><sup>▴ Back to top</sup></a></p>
 
  
#### 1. **Intialise encoder using supported embedding** and models from <a href="#-supported-embeddings-and-models">here</a>  
```python  
>>> from embedding_as_service.text.encode import Encoder  
>>> en = Encoder(embedding='bert', model='bert_base_cased', download=True)  
```  
#### 2. Get sentences **tokens embedding**  
```python 
>>> vecs = en.encode(texts=['hello aman', 'how are you?'])  
>>> vecs  
array([[[ 1.7049843 ,  0.        ,  1.3486509 , ..., -1.3647075 ,  
 0.6958289 ,  1.8013777 ], ... [ 0.4913215 ,  0.60877025,  0.73050433, ..., -0.64490885, 0.8525057 ,  0.3080206 ]]], dtype=float32)  
>>> vecs.shape  
(2, 128, 768) # batch x max_sequence_length x embedding_size  
```  
#### 3. Using **pooling strategy**, click <a href="#-pooling-strategies-">here</a> for more.  
<details><summary><I>Supported Pooling Methods</I></summary>

|Strategy|Description|
|---|---|
| `None` | no pooling at all, useful when you want to use word embedding instead of sentence embedding. This will results in a `[max_seq_len, embedding_size]` encode matrix for a sequence.|
| `reduce_mean` | take the average of all token embeddings |
| `reduce_min` | take the minumun of all token embeddings|
| `reduce_max` | take the maximum of all token embeddings |
| `reduce_mean_max` | do `reduce_mean` and `reduce_max` separately and then concat them together |
| `first_token` | get the token embedding of first token of a sentence |
| `last_token` | get the token embedding of last token of a sentence |
</details>


```python  
>>> vecs = en.encode(texts=['hello aman', 'how are you?'], pooling='reduce_mean')  
>>> vecs  
array([[-0.33547154,  0.34566957,  1.1954105 , ...,  0.33702594,  
 1.0317835 , -0.785943  ], [-0.3439088 ,  0.36881036,  1.0612687 , ...,  0.28851607, 1.1107115 , -0.6253736 ]], dtype=float32)  
  
>>> vecs.shape  
(2, 768) # batch x embedding_size  
```  

#### 4. Use custom `max_seq_length`, default is 128  
```python  
>>> vecs = en.encode(texts=['hello aman', 'how are you?'], max_seq_length=256)  
>>> vecs  
array([[ 0.48388457, -0.01327741, -0.76577514, ..., -0.54265064,  
 -0.5564591 ,  0.6454179 ], [ 0.53209245,  0.00526248, -0.71091074, ..., -0.5171917 , -0.40458363,  0.6779779 ]], dtype=float32)  
  
>>> vecs.shape  
(2, 256, 768) # batch x max_sequence_length x embedding_size  
```  
#### 5. Show embedding Tokens  
```python  
>>> en.tokenize(texts=['hello aman', 'how are you?'])  
[['_hello', '_aman'], ['_how', '_are', '_you', '?']]  
```  
  
#### 6. Using your own tokenizer  
```python  
>>> texts = ['hello aman!', 'how are you']  
  
# a naive whitespace tokenizer  
>>> tokens = [s.split() for s in texts]  
>>> vecs = en.encode(tokens, is_tokenized=True)  
```  
<h2 align="center">📋 API </h2>  
<p align="right"><a href="#embedding-as-service"><sup>▴ Back to top</sup></a></p>

1. **class** <span style="color:blue">`embedding_as_service.text.encoder.Encoder`</span>

  | Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `embedding` | str | *Required* | embedding method to be used, check `Embedding` column <a href="#-supported-embeddings-and-models">here</a>|
| `model`| str |*Required*| Model to be used for mentioned embedding, check `Model` column <a href="#-supported-embeddings-and-models">here</a>|
| `download`| bool |`False`| Download model if model does not exists|

2. **def** <span style="color:blue">`embedding_as_service.text.encoder.Encoder.encode`</span>

  | Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Texts` | List[str] or List[List[str]] | *Required* | List of sentences or list of list of sentence tokens in case of `is_tokenized=True`
| `pooling`| str |(Optional)| Pooling methods to apply, <a href="#-pooling-strategies-">here</a> is available methods|
| `max_seq_length`| int | `128` | Maximum Sequence Length, default is 128|
| `is_tokenized` | bool | `False` | set as True in case of tokens are passed for encoding |  
| `batch_size` | int | `128` | maximum number of sequences handled by encoder, larger batch will be partitioned into small batches. |

 3. **def** <span style="color:blue">`embedding_as_service.text.encoder.Encoder.tokenize`</span>
 
  | Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Texts` | List[str] | *Required* | List of sentences  


<h2 align="center" href="#supported-models">✅ Supported Embeddings and Models</h2>  
<p align="right"><a href="#embedding-as-service"><sup>▴ Back to top</sup></a></p>

 
Here are the list of supported embeddings and their respective models.  
  
|  |Embedding  | Model  | Embedding dimensions | Paper |   
|:--|:--|:--:|:--:|--|  
|:one: |**albert**|`albert_base` | 768| <a href="https://arxiv.org/pdf/1909.11942.pdf"> Read Paper :bookmark:  </a>|  
||  |`albert_large` | 1024| |  
||  |`albert_xlarge` | 2048| |  
||  |`albert_xxlarge` | 4096| |  
|:two: |**xlnet** |`xlnet_large_cased` | 1024| <a href="https://arxiv.org/abs/1906.08237"> Read Paper :bookmark: </a>|  
||  |`xlnet_base_cased` | 768| |  
|:three: |**bert** |`bert_base_uncased` | 768| <a href="https://arxiv.org/abs/1810.04805"> Read Paper :bookmark:  </a>|  
|||`bert_base_cased` | 768| |  
||  |`bert_multi_cased` | 768||   
||  |`bert_large_uncased` | 1024||   
||  |`bert_large_cased` | 1024| |  
|:four: |**elmo** |`elmo_bi_lm` | 512| <a href="https://allennlp.org/elmo"> Read Paper :bookmark: </a>|  
|:five: |**ulmfit** |`ulmfit_forward` | 300|<a href="https://arxiv.org/abs/1801.06146"> Read Paper :bookmark: </a>|   
|||`ulmfit_backward` | 300| |  
|:six: |**use**|`use_dan` | 512| <a href="https://arxiv.org/abs/1803.11175"> Read Paper :bookmark: </a>|  
||  |`use_transformer_large` | 512| |  
||  |`use_transformer_lite` | 512| |  
|:seven: |**word2vec**|`google_news_300` | 300| <a href="https://arxiv.org/abs/1301.3781"> Read Paper :bookmark:  </a>|  
|:eight: |**fasttext**|`wiki_news_300` | 300| <a href="https://arxiv.org/abs/1607.01759"> Read Paper :bookmark: </a>|  
||  |`wiki_news_300_sub` | 300| |  
||  |`common_crawl_300` | 300| |  
||  |`common_crawl_300_sub` | 300| |  
|:nine: |**glove**|`twitter_200` | 200| <a href="https://nlp.stanford.edu/pubs/glove.pdf"> Read Paper :bookmark:  </a>|  
||  |`twitter_100` | 100| |  
||  |`twitter_50` | 50| |  
||  |`twitter_25` | 25| |  
||  |`wiki_300` | 300| |  
||  |`wiki_200` | 200| |  
||  |`wiki_100` | 100| |  
||  |`wiki_50` | 50| |  
||  |`crawl_42B_300` | 300| |  
||  |`crawl_840B_300` | 300| |



## Credits 

This software uses the following open source packages:

- [XLnet](https://github.com/zihangdai/xlnet)
- [tensorflow-hub](https://www.tensorflow.org/hub)


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://www.linkedin.com/in/aman-srivastava-a8bb1285/"><img src="https://avatars0.githubusercontent.com/u/5950398?v=4" width="100px;" alt="Aman Srivastava"/><br /><sub><b>Aman Srivastava</b></sub></a><br /><a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=amansrivastava17" title="Code">💻</a> <a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=amansrivastava17" title="Documentation">📖</a> <a href="#infra-amansrivastava17" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/ashutoshsingh0223"><img src="https://avatars3.githubusercontent.com/u/40604544?v=4" width="100px;" alt="Ashutosh Singh"/><br /><sub><b>Ashutosh Singh</b></sub></a><br /><a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=ashutoshsingh0223" title="Code">💻</a> <a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=ashutoshsingh0223" title="Documentation">📖</a> <a href="#infra-ashutoshsingh0223" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/MrPranav101"><img src="https://avatars0.githubusercontent.com/u/43914392?v=4" width="100px;" alt="MrPranav101"/><br /><sub><b>MrPranav101</b></sub></a><br /><a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=MrPranav101" title="Code">💻</a> <a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=MrPranav101" title="Documentation">📖</a> <a href="#infra-MrPranav101" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>    
    <td align="center"><a href="https://www.linkedin.com/in/dhavaltaunk08/"><img src="https://avatars0.githubusercontent.com/u/31320833?v=4" width="100px;" alt="Dhaval Taunk"/><br /><sub><b>Dhaval Taunk</b></sub></a><br /><a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=DhavalTaunk08" title="Code">💻</a> <a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=DhavalTaunk08" title="Documentation">📖</a> <a href="#infra-DhavalTaunk08" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://chiragjn.github.io"><img src="https://avatars2.githubusercontent.com/u/10295418?v=4" width="100px;" alt="Chirag Jain"/><br /><sub><b>Chirag Jain</b></sub></a><br /><a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=chiragjn" title="Code">💻</a> <a href="https://github.com/amansrivastava17/embedding-as-service/commits?author=chiragjn" title="Documentation">📖</a> <a href="#infra-chiragjn" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

Please read the [contribution guidelines](CONTRIBUTION.md) first.


<h2>Citing</h2>
<p align="right"><a href="#embedding-as-service"><sup>▴ Back to top</sup></a></p>

If you use embedding-as-service in a scientific publication, we would appreciate references to the following BibTex entry:

```latex
@misc{aman2019embeddingservice,
  title={embedding-as-service},
  author={Srivastava, Aman},
  howpublished={\url{https://github.com/amansrivastava17/embedding-as-service}},
  year={2019}
}
```
