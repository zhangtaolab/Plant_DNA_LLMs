from pdllib import ModelInference

def run_inference(model_path, sequence, source='huggingface', device='auto', max_length=512):
    # 初始化ModelInference对象
    model = ModelInference(
        model_path=model_path,
        source=source,
        device=device,
        max_token=max_length
    )

    # 进行预测
    results = model.predict([sequence])

    # 打印结果
    for result in results:
        print(f"Sequence: {sequence}")
        print(f"Length: {len(sequence)}")
        print(f"Label: {result['label']}")
        print(f"Probability: {result['probability']}")
        print()

if __name__ == "__main__":
    # 设置参数
    model_path = "zhangtaolab/plant-dnabert-BPE-promoter"
    sequence = "GGGAAAAAGTGAACTCCATTGTTTTTTCACGCTAAGCAGACCACAATTGCTGCTTGGTACGAAAAGAAAACCGAACCCTTTCACCCACGCACAACTCCATCTCCATTAGCATGGACAGAACACCGTAGATTGAACGCGGGAGGCAACAGGCTAAATCGTCCGTTCAGCCAAAACGGAATCATGGGCTGTTTTTCCAGAAGGCTCCGTGTCGTGTGGTTGTGGTCCAAAAACGAAAAAGAAAGAAAAAAGAAAACCCTTCCCAAGACGTGAAGAAAAGCAATGCGATGCTGATGCACGTTA"

    # 运行推理
    run_inference(model_path, sequence)