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

####################################################################################################

if mamba_available:
    @dataclass
    class MambaConfig:
        # 模型配置类
        # Model configuration class
        d_model: int = 768  # 模型维度
        # Model dimension
        n_layer: int = 24  # 层数
        # Number of layers
        vocab_size: int = 8000  # 词汇表大小
        # Vocabulary size
        ssm_cfg: dict = field(default_factory=dict)  # SSM配置
        # SSM configuration
        rms_norm: bool = True  # 是否使用RMS归一化
        # Whether to use RMS normalization
        residual_in_fp32: bool = True  # 残差是否使用FP32
        # Whether to use FP32 for residuals
        fused_add_norm: bool = True  # 是否使用融合的加法归一化
        # Whether to use fused add normalization
        pad_vocab_size_multiple: int = 8  # 词汇表大小的倍数
        # Multiple for vocabulary size
        tie_embeddings: bool = True  # 是否共享嵌入
        # Whether to tie embeddings

        def __init__(self, **kwargs):
            # 初始化配置
            # Initialize configuration
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_json_string(self):
            # 转换为JSON字符串
            # Convert to JSON string
            return json.dumps(asdict(self))

        def to_dict(self):
            # 转换为字典
            # Convert to dictionary
            return asdict(self)

    class MambaClassificationHead(nn.Module):
        # 分类头类
        # Classification head class
        def __init__(self, d_model, num_classes, **kwargs):
            super(MambaClassificationHead, self).__init__()
            # 初始化线性层
            # Initialize linear layer
            self.classification_head = nn.Linear(d_model, num_classes, **kwargs)

        def forward(self, hidden_states):
            # 前向传播
            # Forward pass
            return self.classification_head(hidden_states)

    class MambaSequenceClassification(MambaLMHeadModel):
        # 序列分类模型类
        # Sequence classification model class
        def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
            num_classes=2,
        ) -> None:
            super().__init__(config, initializer_cfg, device, dtype)
            # 初始化分类头
            # Initialize classification head
            self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=num_classes)

            del self.lm_head  # 删除语言模型头

        def forward(self, input_ids, attention_mask=None, labels=None):
            # 前向传播
            # Forward pass
            hidden_states = self.backbone(input_ids)  # 获取隐藏状态

            mean_hidden_states = hidden_states.mean(dim=1)  # 计算平均隐藏状态

            logits = self.classification_head(mean_hidden_states)  # 计算logits

            if labels is None:
                return ModelOutput(logits=logits)  # 如果没有标签，返回logits

            loss_fct = nn.CrossEntropyLoss()  # 定义损失函数
            loss = loss_fct(logits, labels)  # 计算损失

            return ModelOutput(loss=loss, logits=logits)  # 返回损失和logits

        def can_generate(self):
            # 检查模型是否可以生成
            # Check if the model can generate
            return False

        def predict(self, text, tokenizer, id2label=None):
            # 进行预测
            # Make predictions
            input_ids = torch.tensor(tokenizer(text)['input_ids'], device='cuda')[None]  # 获取输入ID
            with torch.no_grad():
                logits = self.forward(input_ids).logits[0]  # 前向传播获取logits
                label = torch.argmax(logits).item()  # 获取预测标签

            if id2label is not None:
                return id2label[label]  # 返回标签名称
            else:
                return label  # 返回标签ID

        @classmethod
        def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, num_classes=2, **kwargs):
            # 从预训练模型加载
            # Load from pretrained model
            config_data = load_config_hf(pretrained_model_name)  # 加载配置
            config = MambaConfig(**config_data)  # 创建配置对象

            model = cls(config, device=device, dtype=dtype, num_classes=num_classes, **kwargs)  # 初始化模型

            model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)  # 加载模型状态字典
            model.load_state_dict(model_state_dict, strict=False)  # 加载状态字典

            return model  # 返回模型

    class MambaSequenceRegression(MambaLMHeadModel):
        # 序列回归模型类
        # Sequence regression model class
        def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__(config, initializer_cfg, device, dtype)
            # 初始化回归头
            # Initialize regression head
            self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=1)

            del self.lm_head  # 删除语言模型头

        def forward(self, input_ids, attention_mask=None, labels=None):
            # 前向传播
            # Forward pass
            hidden_states = self.backbone(input_ids)  # 获取隐藏状态

            mean_hidden_states = hidden_states.mean(dim=1)  # 计算平均隐藏状态

            logits = self.classification_head(mean_hidden_states)  # 计算logits

            if labels is None:
                return ModelOutput(logits=logits)  # 如果没有标签，返回logits

            loss_fct = nn.MSELoss()  # 定义均方误差损失函数
            loss = loss_fct(logits.squeeze(), labels.squeeze())  # 计算损失

            return ModelOutput(loss=loss, logits=logits)  # 返回损失和logits

        def can_generate(self):
            # 检查模型是否可以生成
            # Check if the model can generate
            return False

        def predict(self, text, tokenizer, id2label=None):
            # 进行预测
            # Make predictions
            input_ids = torch.tensor(tokenizer(text)['input_ids'], device='cuda')[None]  # 获取输入ID
            with torch.no_grad():
                logits = self.forward(input_ids).logits[0]  # 前向传播获取logits
                label = logits.cpu().numpy()  # 获取预测值

            if id2label is not None:
                return id2label[label]  # 返回标签名称
            else:
                return label  # 返回预测值

        @classmethod
        def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
            # 从预训练模型加载
            # Load from pretrained model
            config_data = load_config_hf(pretrained_model_name)  # 加载配置
            config = MambaConfig(**config_data)  # 创建配置对象

            model = cls(config, device=device, dtype=dtype, **kwargs)  # 初始化模型

            model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)  # 加载模型状态字典
            model.load_state_dict(model_state_dict, strict=False)  # 加载状态字典

            return model  # 返回模型

####################################################################################################

# 指定模型列表、分词器类型列表和数据源列表
base_model = ['plant-dnabert', 'plant-nucleotide-transformer',
              'plant-dnagpt', 'plant-dnagemma', 'plant-dnamamba',
              'dnabert2', 'nucleotide-transformer-v2-100m', 'agront-1b']
tokenizer_type = ['BPE', '6mer', 'singlebase']
data_source = ['huggingface','modelscope']
