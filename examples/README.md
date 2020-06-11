## Installation
Please begin by installing the requirements: `pip install -r requirements.txt`

After that, please install the fork of HuggingFace's transformer that lives under the `transformers/` folder. To do that, run: `cd transformers && pip install -e .`
We fork the HuggingFace's repo in order to integrate SMYRF to pre-trained models. After the release of the code, we plan to open a Pull Request for the integration of
SMYRF to the library.


## What's here
The easiest way to start is with our Colab notebooks:
- [Finetuning SMYRF on downstream NLP tasks](https://colab.research.google.com/drive/16_DTy7-jHKHZc9PJ0RVMmLmagzPPm2hP?usp=sharing)
- [Using SMYRF on a pre-trained BigGAN on ImageNet](https://colab.research.google.com/drive/1D_UYVtPz3yEHkACzztwSZM9NLlZZxNjT?usp=sharing)
- [Using SMYRF on a pre-trained BigGAN on Celeba-HQ](https://colab.research.google.com/drive/1kJmNXCz-uiEgiHWKFtJ-tlD-TMj345aN?usp=sharing)
