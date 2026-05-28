import pandas as pd
import argparse
import os

def txt_to_parquet(input_txt, output_parquet):
    print(f"正在读取: {input_txt}")

    df = pd.read_csv(
        input_txt,
        sep='\t',
        header=None,
        names=['text'],
        quoting=3
    )

    print(df.head())

    print(f"正在转换并保存至: {output_parquet}")
    df.to_parquet(output_parquet, engine='pyarrow', index=False)

    print("转换完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="txt 转 parquet 工具")

    parser.add_argument("-i", "--input", required=True, help="输入 txt 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 parquet 文件路径")

    args = parser.parse_args()

    txt_to_parquet(args.input, args.output)