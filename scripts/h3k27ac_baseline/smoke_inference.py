"""Exercise local pdllib CPU inference; random weights imply no accuracy claim."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pdllib import ModelInference, load_seqfile

    with TemporaryDirectory(prefix="pdllms-smoke-") as directory:
        root = Path(directory) / "model"
        root.mkdir()
        torch.manual_seed(7)
        vocabulary = {
            token: index for index, token in enumerate([
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "C", "G", "T", "N"
            ])
        }
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
            cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
            model_max_length=128,
        )
        tokenizer.save_pretrained(root)
        config = BertConfig(
            vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )
        BertForSequenceClassification(config).save_pretrained(root)
        sequences = ["ACGT" * 8, "GGCC" * 8]
        model = ModelInference(str(root), source="local", device="cpu", max_token=64)
        if len(model.tokenizer(sequences[0])["input_ids"]) != 34:
            raise RuntimeError("Tokenizer did not preserve single-base tokens")
        predictions = model.predict(sequences)
        if len(predictions) != 2:
            raise RuntimeError("Expected two batch predictions")
        for prediction in predictions:
            probabilities = prediction["probability"]
            if (set(probabilities) != {"negative", "positive"}
                    or abs(sum(probabilities.values()) - 1) >= 1e-6):
                raise RuntimeError("Invalid classification probabilities")
        fasta = Path(directory) / "smoke.fa"
        fasta.write_text("".join(
            f">seq{index}\n{sequence}\n" for index, sequence in enumerate(sequences)
        ))
        if len(model.predict_file(str(fasta), batch_size=2)) != 2:
            raise RuntimeError("Expected two FASTA predictions")
        if load_seqfile(str(fasta), batch_size=2) != [sequences]:
            raise RuntimeError("FASTA sequences changed while loading")
        print(json.dumps({
            "status": "passed", "batch_predictions": 2, "fasta_predictions": 2,
            "model": "random tiny BERT; environment check only",
        }))


if __name__ == "__main__":
    main()
