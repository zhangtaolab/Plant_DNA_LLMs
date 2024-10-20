import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from dataclasses import dataclass, field, asdict
import json

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    mamba_available = True
except:
    mamba_available = False

if mamba_available:
    @dataclass
    class MambaConfig:
        d_model: int = 768
        n_layer: int = 24
        vocab_size: int = 8000
        ssm_cfg: dict = field(default_factory=dict)
        rms_norm: bool = True
        residual_in_fp32: bool = True
        fused_add_norm: bool = True
        pad_vocab_size_multiple: int = 8
        tie_embeddings: bool = True

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_json_string(self):
            return json.dumps(asdict(self))

        def to_dict(self):
            return asdict(self)

    class MambaClassificationHead(nn.Module):
        def __init__(self, d_model, num_classes, **kwargs):
            super(MambaClassificationHead, self).__init__()
            self.classification_head = nn.Linear(d_model, num_classes, **kwargs)

        def forward(self, hidden_states):
            return self.classification_head(hidden_states)

    class MambaSequenceClassification(MambaLMHeadModel):
        def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
            num_classes=2,
        ) -> None:
            super().__init__(config, initializer_cfg, device, dtype)

            self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=num_classes)

            del self.lm_head

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden_states = self.backbone(input_ids)

            mean_hidden_states = hidden_states.mean(dim=1)

            logits = self.classification_head(mean_hidden_states)

            if labels is None:
                return ModelOutput(logits=logits)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return ModelOutput(loss=loss, logits=logits)

        def can_generate(self):
            return False

        def predict(self, text, tokenizer, id2label=None):
            input_ids = torch.tensor(tokenizer(text)['input_ids'], device='cuda')[None]
            with torch.no_grad():
                logits = self.forward(input_ids).logits[0]
                label = torch.argmax(logits).item()

            if id2label is not None:
                return id2label[label]
            else:
                return label

        @classmethod
        def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, num_classes=2, **kwargs):
            config_data = load_config_hf(pretrained_model_name)
            config = MambaConfig(**config_data)

            model = cls(config, device=device, dtype=dtype, num_classes=num_classes, **kwargs)

            model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
            model.load_state_dict(model_state_dict, strict=False)

            return model

    class MambaSequenceRegression(MambaLMHeadModel):
        def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__(config, initializer_cfg, device, dtype)

            self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=1)

            del self.lm_head

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden_states = self.backbone(input_ids)

            mean_hidden_states = hidden_states.mean(dim=1)

            logits = self.classification_head(mean_hidden_states)

            if labels is None:
                return ModelOutput(logits=logits)

            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

            return ModelOutput(loss=loss, logits=logits)

        def can_generate(self):
            return False

        def predict(self, text, tokenizer, id2label=None):
            input_ids = torch.tensor(tokenizer(text)['input_ids'], device='cuda')[None]
            with torch.no_grad():
                logits = self.forward(input_ids).logits[0]
                label = logits.cpu().numpy()

            if id2label is not None:
                return id2label[label]
            else:
                return label

        @classmethod
        def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
            config_data = load_config_hf(pretrained_model_name)
            config = MambaConfig(**config_data)

            model = cls(config, device=device, dtype=dtype, **kwargs)

            model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
            model.load_state_dict(model_state_dict, strict=False)

            return model
