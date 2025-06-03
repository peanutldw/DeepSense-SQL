import json
import re
import os
from typing import List, Tuple, Any, Optional


def clean_and_process_sql(
        input_path: str,
        output_path: Optional[str] = None,
        pattern: str = r'(WITH\b|SELECT\b).*',
        flags: int = re.IGNORECASE | re.DOTALL
) -> Tuple[List[List[str]], str]:
    """
    合并版SQL清理处理器：读取JSON文件，清理SQL内容，并保存结果

    参数:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径（None则不保存）
        pattern: 正则表达式模式（默认匹配SELECT/WITH开头的SQL）
        flags: 正则表达式标志

    返回:
        Tuple[清理后的数据, 状态消息]

    示例:
        data, msg = clean_and_process_sql("input.json", "output.json")
    """

    def clean_sql(sql_str: str) -> str:
        """内部清理函数"""
        match = re.search(pattern, sql_str, flags=flags)
        return match.group(0) if match else sql_str

    try:
        # 验证输入路径
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 读取并处理数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data: List[List[str]] = json.load(f)

        cleaned_data = []
        for item in data:
            if len(item) >= 2:  # 基本格式检查
                question, raw_sql = item[0], str(item[1])
                cleaned_data.append([question, clean_sql(raw_sql)])

        # 保存结果（如果指定了输出路径）
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
            msg = f"成功处理并保存到 {output_path}"
        else:
            msg = "成功处理数据（未保存文件）"

        return cleaned_data, msg

    except json.JSONDecodeError as e:
        return [], f"JSON解析错误: {str(e)}"
    except Exception as e:
        return [], f"处理失败: {str(e)}"


# 使用示例
if __name__ == "__main__":
    input_file = "../outputs/bird/predict_dev.json"
    output_file = "../bird/cleaned_sql.json"

    data, message = clean_and_process_sql(input_file, output_file)
    print(message)