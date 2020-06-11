\small
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
<td style="text-align: left;"></td>
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
<td style="text-align: left;"></td>
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

\centering
+---------+---------+---------+-----------------+----------------+
|         | Dataset | Memory  | SMYRF Inference | Accuracy       |
+:========+:========+:========+:===============:+:===============+
| RoBERTa |         | $100\%$ | \xmark          | $\bm{94.96\%}$ |
+---------+---------+---------+-----------------+----------------+
|         |         |         |                 | $93.72\%$      |
+---------+---------+---------+-----------------+----------------+
|         |         |         |                 | $\bm{94.62\%}$ |
+---------+---------+---------+-----------------+----------------+
| BERT    |         | $100\%$ |                 | $94.12\%$      |
+---------+---------+---------+-----------------+----------------+
|         |         |         |                 | $92.64\%$      |
+---------+---------+---------+-----------------+----------------+
|         |         |         |                 | $\bm{93.54\%}$ |
+---------+---------+---------+-----------------+----------------+
