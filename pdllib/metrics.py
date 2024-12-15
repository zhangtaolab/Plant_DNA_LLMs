import os
import numpy as np
from scipy.special import softmax
import sklearn
import evaluate


# Define evaluation metrics
def calculate_metric_with_sklearn(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    print(valid_labels.shape, valid_predictions.shape)
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

## Load evaluate metrics locally to avoid downloading from Hugging Face

def classification_metrics():
    clf_metrics = evaluate.combine(["evaluate/metrics/accuracy/accuracy.py",
                                    "evaluate/metrics/f1/f1.py",
                                    "evaluate/metrics/precision/precision.py",
                                    "evaluate/metrics/recall/recall.py",
                                    "evaluate/metrics/matthews_correlation/matthews_correlation.py"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        predictions = np.argmax(logits, axis=-1)
        return clf_metrics.compute(predictions=predictions, references=labels)

    return compute_metrics


def regression_metrics():
    mse_metric = evaluate.load("evaluate/metrics/mse/mse.py")
    mae_metric = evaluate.load("evaluate/metrics/mae/mae.py")
    r2_metric = evaluate.load("evaluate/metrics/r_squared/r_squared.py")
    spm_metric = evaluate.load("evaluate/metrics/spearmanr/spearmanr.py")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        mse = mse_metric.compute(references=labels, predictions=logits)
        mae = mae_metric.compute(references=labels, predictions=logits)
        r2 = r2_metric.compute(references=labels, predictions=logits)
        spearmanr = spm_metric.compute(references=labels, predictions=logits)

        return {**mse, **mae, "r2": r2, **spearmanr}

    return compute_metrics


def multi_classification_metrics():
    metric0 = evaluate.load("evaluate/metrics/accuracy/accuracy.py")
    metric1 = evaluate.load("evaluate/metrics/precision/precision.py")
    metric2 = evaluate.load("evaluate/metrics/recall/recall.py")
    metric3 = evaluate.load("evaluate/metrics/f1/f1.py")
    metric4 = evaluate.load("evaluate/metrics/matthews_correlation/matthews_correlation.py")
    roc_metric = evaluate.load("evaluate/metrics/roc_auc/roc_auc.py", "multiclass")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        # predictions = np.argmax(logits, axis=-1)
        pred_probs = softmax(logits, axis=1)
        predictions = [x.tolist().index(max(x)) for x in pred_probs]

        accuracy = metric0.compute(predictions=predictions, references=labels)
        precision = metric1.compute(predictions=predictions, references=labels, average="micro")
        recall = metric2.compute(predictions=predictions, references=labels, average="micro")
        f1 = metric3.compute(predictions=predictions, references=labels, average="micro")
        mcc = metric4.compute(predictions=predictions, references=labels)
        roc_auc_ovr = roc_metric.compute(references=labels,
                                        prediction_scores=pred_probs,
                                        multi_class='ovr')
        roc_auc_ovo = roc_metric.compute(references=labels,
                                        prediction_scores=pred_probs,
                                        multi_class='ovo')

        return {**accuracy, **precision, **recall, **f1, **mcc,
                "AUROC_ovr": roc_auc_ovr['roc_auc'], "AUROC_ovo": roc_auc_ovo['roc_auc']}

    return compute_metrics


def multi_labels_metrics():
    metric0 = evaluate.load("evaluate/metrics/accuracy/accuracy.py")
    metric1 = evaluate.load("evaluate/metrics/precision/precision.py")
    metric2 = evaluate.load("evaluate/metrics/recall/recall.py")
    metric3 = evaluate.load("evaluate/metrics/f1/f1.py")
    metric4 = evaluate.load("evaluate/metrics/matthews_correlation/matthews_correlation.py")
    roc_metric = evaluate.load("evaluate/metrics/roc_auc/roc_auc.py", "multilabel")

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = sigmoid(logits)
        predictions = (predictions > 0.5).astype(int).reshape(-1)    
        labels = labels.astype(int).reshape(-1)

        accuracy = metric0.compute(predictions=predictions, references=labels)
        precision = metric1.compute(predictions=predictions, references=labels, average="micro")
        recall = metric2.compute(predictions=predictions, references=labels, average="micro")
        f1 = metric3.compute(predictions=predictions, references=labels, average="micro")
        mcc = metric4.compute(predictions=predictions, references=labels)
        roc_auc = roc_metric.compute(references=labels,
                                        prediction_scores=predictions,
                                        average='micro')

        return {**accuracy, **precision, **recall, **f1, **mcc,
                "AUROC": roc_auc['roc_auc']}

    return compute_metrics


def token_classification_metrics(label_list):
    seqeval = evaluate.load("evaluate/metrics/seqeval/seqeval.py")

    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=-1)

        # 将id转换为原始的字符串类型的标签
        true_predictions = [
            [label_list[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels) 
        ]

        true_labels = [
            [label_list[l] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels) 
        ]

        result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")

        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"]
        }

    return compute_metrics


def metrics_for_dnabert2(task):
    import torch

    r2_metric = evaluate.load("evaluate/metrics/r_squared/r_squared.py")
    spm_metric = evaluate.load("evaluate/metrics/spearmanr/spearmanr.py")
    clf_metrics = evaluate.combine(["e../valuate/metrics/accuracy/accuracy.py",
                                    "evaluate/metrics/f1/f1.py",
                                    "evaluate/metrics/precision/precision.py",
                                    "evaluate/metrics/recall/recall.py",
                                    "evaluate/metrics/matthews_correlation/matthews_correlation.py"])
    metric1 = evaluate.load("evaluate/metrics/precision/precision.py")
    metric2 = evaluate.load("evaluate/metrics/recall/recall.py")
    metric3 = evaluate.load("evaluate/metrics/f1/f1.py")
    metric4 = evaluate.load("evaluate/metrics/matthews_correlation/matthews_correlation.py")
    roc_metric = evaluate.load("evaluate/metrics/roc_auc/roc_auc.py", "multiclass")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task.lower() == "regression":
            r2 = r2_metric.compute(references=labels, predictions=logits[0])
            spearman = spm_metric.compute(references=labels, predictions=logits[0])
            return {"r2": r2, "spearmanr": spearman['spearmanr']}
        else:
            if task.lower() == "classification":
                predictions = torch.argmax(torch.from_numpy(logits[0]), dim=-1)
                return clf_metrics.compute(predictions=predictions, references=labels)
            else:
                pred_probs = softmax(logits[0], axis=1)
                predictions = [x.tolist().index(max(x)) for x in pred_probs]
                precision = metric1.compute(predictions=predictions, references=labels, average="micro")
                recall = metric2.compute(predictions=predictions, references=labels, average="micro")
                f1 = metric3.compute(predictions=predictions, references=labels, average="micro")
                mcc = metric4.compute(predictions=predictions, references=labels)
                roc_auc_ovr = roc_metric.compute(references=labels,
                                        prediction_scores=pred_probs,
                                        multi_class='ovr')
                roc_auc_ovo = roc_metric.compute(references=labels,
                                        prediction_scores=pred_probs,
                                        multi_class='ovo')
                return {**precision, **recall, **f1, **mcc, "AUROC_ovr": roc_auc_ovr['roc_auc'], "AUROC_ovo": roc_auc_ovo['roc_auc']}

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        logits = logits[0] if isinstance(logits, tuple) else logits
        # pred_ids = torch.argmax(logits, dim=-1)
        return logits, labels
    
    return compute_metrics, preprocess_logits_for_metrics
