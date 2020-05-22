from transformers import InputExample
import nlp

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
