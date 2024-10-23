import os
from .inference import ModelInference

def plant_llms_inference(model_path, seq_or_file, source='huggingface', device='auto', max_length=512):
    """
    Perform inference on a DNA sequence using a specified model.
    使用指定的模型对DNA序列进行推理。

    Args:
    model_path (str): Path or name of the model to use.
                      要使用的模型的路径或名称。
    seq_or_file (str): The DNA sequence or file contains sequences to analyze.
                    要分析的DNA序列或文件。
    source (str): Source of the model (default: 'huggingface').
                  模型的来源（默认：'huggingface'）。
    device (str): Device to run inference on (default: 'auto').
                  运行推理的设备（默认：'auto'）。
    max_length (int): Maximum length of the input sequence (default: 512).
                      输入序列的最大长度（默认：512）。

    Returns:
    dict: A dictionary containing the prediction results.
          包含预测结果的字典。
    """
    # 检查输入序列的有效性
    # Check the validity of the input sequence
    if not seq_or_file or not isinstance(seq_or_file, str):
        return {"error": "Invalid sequence provided. Please provide a non-empty string."}

    try:
        model = ModelInference(
            model_path=model_path,
            source=source,
            device=device,
            max_token=max_length
        )
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    try:
        if os.path.exists(seq_or_file):
            results = model.predict_file(seq_or_file)
        else:
            results = model.predict([seq_or_file])
            if results:
                results = results[0]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    if results:
        return results
    else:
        return {"error": "No prediction result"}
