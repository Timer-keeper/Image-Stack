from PIL import Image
import numpy as np
import os
import time


def compute_average_image(input_folder, output_path):
    """
    从指定文件夹加载图像，计算平均值堆栈并保存结果（优化版本）

    :param input_folder: 包含输入图像的文件夹路径
    :param output_path: 输出文件的完整路径（包含文件名）
    """
    start_time = time.time()

    # 输入验证
    if not os.path.isdir(input_folder):
        raise ValueError(f"输入文件夹 '{input_folder}' 不存在")

    # 准备文件列表
    valid_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    file_list = [
        f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)
    ]
    total_files = len(file_list)
    if total_files == 0:
        raise ValueError(f"未找到支持格式的图像文件，支持的格式为：{valid_extensions}")

    print(f"█ 发现 {total_files} 个图像文件，开始处理...")

    # 初始化处理变量
    sum_array = None
    target_size = None
    count = 0
    resize_count = 0
    errors = []

    # 主处理循环
    for current, filename in enumerate(sorted(file_list), 1):
        filepath = os.path.join(input_folder, filename)
        print(f"▌ 处理进度 ({current}/{total_files}): {filename}")

        try:
            with Image.open(filepath) as img:
                # 转换为RGB并验证有效性
                rgb_img = img.convert("RGB")

                # 设置目标尺寸（以第一个有效图像为准）
                if target_size is None:
                    target_size = rgb_img.size
                    sum_array = np.zeros((*target_size[::-1], 3), dtype=np.float64)
                    print(f"█ 基准尺寸设置为：{target_size}")

                # 尺寸调整
                if rgb_img.size != target_size:
                    rgb_img = rgb_img.resize(target_size, Image.Resampling.LANCZOS)
                    resize_count += 1

                # 累加像素值
                sum_array += np.array(rgb_img, dtype=np.float64)
                count += 1

        except Exception as e:
            errors.append((filename, str(e)))
            print(f"⚠ 跳过 {filename}: {str(e)}")
            continue

    # 后处理验证
    if count == 0:
        raise ValueError("没有成功加载任何图像文件")

    # 计算平均值并转换格式
    average_array = np.clip(sum_array / count, 0, 255).astype(np.uint8)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 智能保存（根据格式优化参数）
    output_ext = os.path.splitext(output_path)[1].lower()
    result_img = Image.fromarray(average_array)

    if output_ext in (".jpg", ".jpeg"):
        result_img.save(output_path, quality=100, subsampling=0)  # 最高质量JPEG
    elif output_ext == ".png":
        result_img.save(output_path, optimize=True)  # 优化PNG
    else:
        result_img.save(output_path)

    # 生成报告
    processing_time = time.time() - start_time
    print("\n█ 处理完成!")
    print(f"├ 成功处理图像: {count} 张")
    print(f"├ 尺寸调整数量: {resize_count} 张")
    print(f"├ 输出文件尺寸: {target_size}")
    print(f"├ 处理耗时: {processing_time:.2f}秒")

    if errors:
        print(f"└ 跳过文件 ({len(errors)} 个):")
        for filename, error in errors:
            print(f"  ├ {filename}: {error}")
    print(f"输出文件保存至: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    input_folder = "./images"
    output_path = "./output/average_result.png"  # 推荐使用无损格式

    try:
        compute_average_image(input_folder, output_path)
    except Exception as e:
        print(f"处理失败: {str(e)}")
