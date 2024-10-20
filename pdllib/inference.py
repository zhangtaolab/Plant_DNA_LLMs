import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import time
from .models import MambaSequenceClassification, MambaSequenceRegression
from .utils import load_seqfile
from .task_map import task_map

class ModelInference:
    def __init__(self, model_path, source='huggingface', device='auto', max_token=512):
        self.model_path = model_path
        self.source = source
        self.device = self._get_device(device)
        self.max_token = max_token
        self._load_model_and_tokenizer()

    def _get_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _load_model_and_tokenizer(self):
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model_name = config._name_or_path
        self.id2label = config.id2label
        self.num_labels = len(self.id2label)

        if self.id2label[0] == "LABEL_0":
            for task in task_map:
                if task in self.model_name:
                    self.num_labels = len(task_map[task]['labels'])
                    self.id2label = {i: task_map[task]['labels'][i] for i in range(self.num_labels)}
            if self.id2label[0] == "LABEL_0":
                if self.num_labels == 2:
                    self.id2label = {0: 'False', 1: 'True'}
                elif self.num_labels == 3:
                    self.id2label = {0: 'None', 1: 'Full', 2: 'Partial'}
                else:
                    self.id2label = {i: str(i) for i in range(self.num_labels)}

        if "dnamamba" in self.model_name.lower():
            if self.num_labels > 1:
                self.model = MambaSequenceClassification.from_pretrained(self.model_path, num_classes=self.num_labels)
            else:
                self.model = MambaSequenceRegression.from_pretrained(self.model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.num_labels,
                trust_remote_code=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)

    def predict(self, sequences, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_token).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits

        if self.num_labels > 1:
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            results = []
            for i, pred in enumerate(predictions):
                label = self.id2label[pred.item()]
                prob = probs[i].tolist()
                results.append({'label': label, 'probability': {self.id2label[j]: p for j, p in enumerate(prob)}})
        else:
            predictions = logits.squeeze()
            results = [{'label': pred.item(), 'probability': pred.item()} for pred in predictions]

        return results

    def predict_file(self, file_path, threshold=0.5, batch_size=1):
        seqs = load_seqfile(file_path, batch_size=batch_size)
        all_results = []
        for batch in seqs:
            results = self.predict(batch, threshold)
            all_results.extend(results)
        return all_results
