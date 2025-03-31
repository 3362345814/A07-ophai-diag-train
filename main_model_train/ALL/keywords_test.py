from pathlib import Path

from config import ROOT_DIR
from gen_keywords import InferenceEngine, prepare_datasets


def load_vocab():
    """加载训练时生成的词汇表"""
    _, word2idx = prepare_datasets(
        ROOT_DIR / "dataset/Archive/full_df.csv",
        ROOT_DIR / "dataset/Archive/preprocessed_images"
    )
    return word2idx


def test_single_image(model_path, image_path, word2idx):
    """执行单图测试"""
    try:
        # 初始化推理引擎
        engine = InferenceEngine(model_path, word2idx)

        # 执行预测
        prediction, _ = engine.predict(image_path)

        print("\n" + "=" * 50)
        print(f"测试图片: {image_path}")
        print("生成关键词:", prediction)
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"测试失败: {str(e)}")
        if not Path(model_path).exists():
            print(f"模型文件不存在：{model_path}")
        if not Path(image_path).exists():
            print(f"图片文件不存在：{image_path}")


if __name__ == "__main__":
    # 加载词汇表
    word2idx = load_vocab()

    model = "models/keywords_model.pth"
    image = ROOT_DIR / "dataset/Archive/optic_disk/61_left.jpg"

    # 执行测试
    test_single_image(model, image, word2idx)
