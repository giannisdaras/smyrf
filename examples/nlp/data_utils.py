from transformers import InputExample
import nlp
import warnings
import random


class ImdbProcessor:
    def __init__(self):
        self.dataset = nlp.load_dataset('imdb')
        self.counter = 0

    def get_labels(self):
        return [0, 1]

    def get_dev_examples(self, *args, **kwargs):
        examples = []
        for example in self.dataset['test']:
            examples.append(InputExample(guid=self.counter, text_a=example['text'], text_b=None, label=example['label']))
            self.counter += 1
        return examples

    def get_train_examples(self, *args, **kwargs):
        examples = []
        for example in self.dataset['train']:
            examples.append(InputExample(guid=self.counter, text_a=example['text'], text_b=None, label=example['label']))
            self.counter += 1
        return examples



class HyperpartisanProcessor:
    def __init__(self):
        warnings.warn('Hyperpartisan requires manual installation. Make sure you download the files from Zenodo first.')
        dataset = nlp.load_dataset('hyperpartisan_news_detection', 'byarticle', data_dir='.')
        examples = [x for x in dataset['train']]
        random.shuffle(examples)
        last_train = int(0.9 * len(examples))
        self.train_examples = examples[:last_train]
        self.dev_examples = examples[last_train:]
        self.counter = 0

    def get_labels(self):
        return [True, False]

    def get_dev_examples(self, *args, **kwargs):
        examples = []
        for example in self.dev_examples:
            examples.append(InputExample(guid=self.counter,
                                         text_a=self.remove_html_markup(example['text']),
                                         text_b=None,
                                         label=example['hyperpartisan']))
        return examples


    def get_train_examples(self, *args, **kwargs):
        examples = []
        for example in self.train_examples:
            examples.append(InputExample(guid=self.counter,
                                         text_a=self.remove_html_markup(example['text']),
                                         text_b=None,
                                         label=example['hyperpartisan']))
        return examples

    def remove_html_markup(self, s):
        # code from: https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
        tag = False
        quote = False
        out = ""

        for c in s:
                if c == '<' and not quote:
                    tag = True
                elif c == '>' and not quote:
                    tag = False
                elif (c == '"' or c == "'") and tag:
                    quote = not quote
                elif not tag:
                    out = out + c

        return out


class BoolQProcessor:
    def __init__(self):
        self.dataset = nlp.load_dataset('boolq')
        self.counter = 0

    def get_labels(self):
        return [True, False]

    def get_dev_examples(self, *args, **kwargs):
        examples = []
        for example in self.dataset['validation']:
            examples.append(InputExample(guid=self.counter, text_a=example['question'], text_b=example['passage'], label=example['answer']))
            self.counter += 1
        return examples

    def get_train_examples(self, *args, **kwargs):
        examples = []
        for example in self.dataset['train']:
            examples.append(InputExample(guid=self.counter, text_a=example['question'], text_b=example['passage'], label=example['answer']))
            self.counter += 1
        return examples
