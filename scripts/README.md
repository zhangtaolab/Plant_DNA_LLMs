* `run_finetune_test.sh` is the test script for finetuning

* `run_inference_test.sh` is the test script for inference

* `hfd.sh` is download from [here](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f), which is used for fast download models and datasets

    Usage:
    ```bash
    # Download model
    hfd.sh zhangtaolab/plant-dnagpt-BPE --tool wget --local-dir plant-dnagpt-BPE
    # Download dataset
    hfd.sh zhangtaolab/plant-multi-species-core-promoters --dataset --tool wget --local-dir plant-multi-species-core-promoters
    ```