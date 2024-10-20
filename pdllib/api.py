from .inference import ModelInference

def plant_llms_inference(model_path, sequence, source='huggingface', device='auto', max_length=512):
    """
    Perform inference on a DNA sequence using a specified model.
    使用指定的模型对DNA序列进行推理。

    Args:
    model_path (str): Path or name of the model to use.
                      要使用的模型的路径或名称。
    sequence (str): The DNA sequence to analyze.
                    要分析的DNA序列。
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
    model = ModelInference(
        model_path=model_path,
        source=source,
        device=device,
        max_token=max_length
    )

    results = model.predict([sequence])

    if results:
        return results[0]
    else:
        return {"error": "No prediction result"}
