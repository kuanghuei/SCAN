# Introduction

This is Stacked Cross Attention Network, source code of [Stacked Cross Attention for Image-Text Matching](https://arxiv.org/abs/1803.08024) ([project page](https://kuanghuei.github.io/SCANProject/)) from Microsoft AI and Research. The paper will appear in ECCV 2018. It is built on top of the [VSE++](https://github.com/fartashf/vsepp) in PyTorch.


## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 0.3
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The precomputed image features of MS-COCO are from [here](https://github.com/peteanderson80/bottom-up-attention). The precomputed image features of Flickr30K are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from:

```bash
wget https://iudata.blob.core.windows.net/scan/data.zip
wget https://iudata.blob.core.windows.net/scan/vocab.zip
```

We refer to the path of extracted files for `data.zip` as `$DATA_PATH` and files for `vocab.zip` to `./vocab` directory. Alternatively, you can also run vocab.py to produce vocabulary files. For example, 

```bash
python vocab.py --data_path data --data_name f30k_precomp
python vocab.py --data_path data --data_name coco_precomp
```

## Data pre-processing (Optional)

The image features of Flickr30K and MS-COCO are available in numpy array format, which can be used for training directly. However, if you wish to test on another dataset, you will need to start from scratch:

1. Use the `bottom-up-attention/tools/generate_tsv.py` and the bottom-up attention model to extract features of image regions. The output file format will be a tsv, where the columns are ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features'].
2. Use `util/convert_data.py` to convert the above output to a numpy array.


If downloading the whole data package containing bottom-up image features for Flickr30K and MS-COCO is too slow for you, you can download the following package with everything but image features and compute image features locally from raw images.

```bash
wget https://iudata.blob.core.windows.net/scan/data_no_feature.zip
```

## Training new models
Run `train.py`:

```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/coco_scan/log --model_name runs/coco_scan/log --max_violation --bi_gru
```

Arguments used to train Flickr30K models:

| Method    | Arguments |
| :-------: | :-------: |
| SCAN t-i LSE     | `--max_violation --bi_gru --agg_func=LogSumExp --cross_attn=t2i --lambda_lse=6 --lambda_softmax=9` |
| SCAN t-i AVG     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9` |
| SCAN i-t LSE     | `--max_violation --bi_gru --agg_func=LogSumExp --cross_attn=i2t --lambda_lse=5 --lambda_softmax=4` |
| SCAN i-t AVG     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4` |


Arguments used to train MS-COCO models:

| Method    | Arguments |
| :-------: | :-------: |
| SCAN t-i LSE     | `--max_violation --bi_gru --agg_func=LogSumExp --cross_attn=t2i --lambda_lse=6 --lambda_softmax=9 --num_epochs=20 --lr_update=10 --learning_rate=.0005` |
| SCAN t-i AVG     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epochs=20 --lr_update=10 --learning_rate=.0005` |
| SCAN i-t LSE     | `--max_violation --bi_gru --agg_func=LogSumExp --cross_attn=i2t --lambda_lse=20 --lambda_softmax=4 --num_epochs=20 --lr_update=10 --learning_rate=.0005` |
| SCAN i-t AVG     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4 --num_epochs=20 --lr_update=10 --learning_rate=.0005` |

## Evaluate trained models

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/coco_scan/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.

## Reference

If you found this code useful, please cite the following paper:

```
@article{lee2018stacked,
  title={Stacked Cross Attention for Image-Text Matching},
  author={Lee, Kuang-Huei and Chen, Xi and Hua, Gang and Hu, Houdong and He, Xiaodong},
  journal={arXiv preprint arXiv:1803.08024},
  year={2018}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)


## Acknowledgments

The authors would like to thank [Po-Sen Huang](https://posenhuang.github.io/) and Yokesh Kumar for helping the manuscript. We also thank Li Huang, Arun Sacheti, and Bing Multimedia team for supporting this work.
