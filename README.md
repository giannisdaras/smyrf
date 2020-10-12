![](https://img.shields.io/badge/smyrf-smyrf-brightgreen) ![](https://img.shields.io/badge/-pytorch-red) ![](https://img.shields.io/badge/-tensorflow-yellow)

## SMYRF: Efficient attention using asymmetric clustering

Get started: [![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/19fIDq7LrRBRz0grM1Frs9Irg3lVh0dXN?usp=sharing)

### Abstract
> We propose a novel type of balanced clustering algorithm to approximate attention. Attention complexity is reduced from O(N^2) to O(NlogN), where N is the sequence length. Our algorithm, SMYRF, uses Locality Sensitive Hashing (LSH) in a novel way by defining new Asymmetric transformations and an adaptive scheme that produces balanced clusters. The biggest advantage of SMYRF is that it can be used as a drop-in replacement for dense attention layers *without any retraining*.
On the contrary, prior fast attention methods impose constraints (e.g. tight queries and keys) and require re-training from scratch. We apply our method to pre-trained state-of-the-art Natural Language Processing and Computer Vision models and we report significant memory and speed benefits. Notably, SMYRF-BERT outperforms (slightly) BERT on GLUE, while using $50\%$ less memory. We also show that SMYRF can be used interchangeably with dense attention before and after training. Finally, we use SMYRF to train GANs with attention in high resolutions. Using a single TPU, we train BigGAN on Celeba-HQ, with attention at resolution 128x128 and 256x256, capable of generating realistic human faces.

### Results

#### Memory-quality trade-off
![](visuals/quality_degradation.png)


#### GLUE benchmark
<table>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th style="text-align: left;">Avg.</th>
<th style="text-align: left;"><span class="math inline">#</span></th>
<th style="text-align: left;"><span class="math inline"><em>C</em></span></th>
<th style="text-align: left;">CoLA</th>
<th style="text-align: left;">MNLI-m/mm</th>
<th style="text-align: left;">MRPC</th>
<th style="text-align: left;">QNLI</th>
<th style="text-align: left;">QQP</th>
<th style="text-align: left;">RTE</th>
<th style="text-align: left;">SST-2</th>
<th style="text-align: left;">STS-B</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">BERT<span class="math inline"><em></em><sub>128</sub></span></td>
<td style="text-align: left;"><span class="math inline">82.69</span></td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">57.83</td>
<td style="text-align: left;"><span class="math inline"><strong>84.43</strong><strong>/</strong><strong>84.68</strong></span></td>
<td style="text-align: left;"><span class="math inline"><strong>88.41</strong></span></td>
<td style="text-align: left;"><span class="math inline"><strong>91.31</strong></span></td>
<td style="text-align: left;">89.70</td>
<td style="text-align: left;">65.70</td>
<td style="text-align: left;"><span class="math inline"><strong>93.46</strong></span></td>
<td style="text-align: left;"><span class="math inline">88.73</span></td>
</tr>
<tr class="even">
<td style="text-align: left;">SMYRF-BERT<span class="math inline"><em></em><sub>2x32</sub></span></td></td>
<td style="text-align: left;"><span class="math inline"><strong>82.98</strong></span></td>
<td style="text-align: left;">2</td>
<td style="text-align: left;">32</td>
<td style="text-align: left;"><span class="math inline">58.79</span></td>
<td style="text-align: left;">83.76/84.27</td>
<td style="text-align: left;">87.69</td>
<td style="text-align: left;">91.14</td>
<td style="text-align: left;"><span class="math inline"><strong>89.72</strong></span></td>
<td style="text-align: left;"><span class="math inline"><strong>68.59</strong></span></td>
<td style="text-align: left;">93.23</td>
<td style="text-align: left;"><span class="math inline"><strong>89.65</strong></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">SMYRF-BERT<span class="math inline"><em></em><sub>2x16</sub></span></td></td>
<td style="text-align: left;"><span class="math inline">81.74</span></td>
<td style="text-align: left;">2</td>
<td style="text-align: left;">16</td>
<td style="text-align: left;"><span class="math inline"><strong>58.90</strong></span></td>
<td style="text-align: left;"><span class="math inline">82.86/83.49</span></td>
<td style="text-align: left;"><span class="math inline">85.72</span></td>
<td style="text-align: left;"><span class="math inline">89.53</span></td>
<td style="text-align: left;"><span class="math inline">89.33</span></td>
<td style="text-align: left;"><span class="math inline">64.98</span></td>
<td style="text-align: left;"><span class="math inline">93.12</span></td>
<td style="text-align: left;"><span class="math inline">87.75</span></td>
</tr>
<tr class="even">
<td style="text-align: left;">BERT<span class="math inline"><em></em><sub>64</sub></span></td>
<td style="text-align: left;"><span class="math inline">81.57</span></td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">64</td>
<td style="text-align: left;">58.80</td>
<td style="text-align: left;">82.34/82.47</td>
<td style="text-align: left;">87.02</td>
<td style="text-align: left;">90.48</td>
<td style="text-align: left;">89.69</td>
<td style="text-align: left;">61.73</td>
<td style="text-align: left;">93.00</td>
<td style="text-align: left;">88.64</td>
</tr>
<tr class="odd">
<td style="text-align: left;">BERT<span class="math inline"><em></em><sub>32</sub></span></td>
<td style="text-align: left;"><span class="math inline">73.56</span></td>
<td style="text-align: left;">1</td>
<td style="text-align: left;">32</td>
<td style="text-align: left;"><span class="math inline">56.40</span></td>
<td style="text-align: left;"><span class="math inline">64.51/63.41</span></td>
<td style="text-align: left;"><span class="math inline">77.89</span></td>
<td style="text-align: left;">79.81</td>
<td style="text-align: left;">88.59</td>
<td style="text-align: left;">55.23</td>
<td style="text-align: left;">92.66</td>
<td style="text-align: left;">83.53</td>
</tr>
</tbody>
</table>

#### Interchangeability of SMYRF and dense attention
Results on IMDB dataset. Using dense attention on inference consistently improves results, nearly matching dense attention perf.

|               | Memory  | SMYRF Inference | Accuracy        |
|---------------|--------:|----------------:|----------------:|
| RoBERTa       |  100%   | &#9746;         | 94.96%          |
| SMYRF-RoBERTa |  50%    | &#x2612;        | 93.72%          |
| SMYRF-RoBERTa |  50%    | &#x2611;        | 94.62%          |
| BERT          |  100%   | &#x2612;        | 94.12%          |
| SMYRF-BERT    |  50%    | &#x2612;        | 92.64%          |
| SMYRF-BERT    |  50%    | &#x2611;        | 93.54%          |


#### Smyrf-BigGAN training on Celeba-HQ-128
Generated faces by a Smyrf-BigGAN trained on 128x128 resolution with attention at 128x128, using 50% of dense memory.
![](visuals/smyrf_128res.jpg)

Results after 120k iterations:

|              | Resolution | Attention | # | C    | FID   |
|--------------|------------|-----------|---|------|-------|
| BigGAN       | 128x128    | 64x64     | 1 | 4096 | 26.06 |
| Smyrf-BigGAN | 128x128    | 128x128   | 4 | 2048 | **25.03** |

where \# denotes number of hashes and C number of queries per cluster.


### What's here
The code hosted in this repository is the one we used to run all the experiments in the paper.
Get started: [![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/19fIDq7LrRBRz0grM1Frs9Irg3lVh0dXN?usp=sharing)

For a deeper dive, look at the `examples/` folder where we have code for pre-training SMYRF-BigGAN, sampling from a pre-trained BigGAN with SMYRF, finetuning state-of-the-art NLP models with SMYRF and a lot more.
