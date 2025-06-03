import random
import json
import time
from copy import deepcopy


def shuffle_schema(schema_info, seed_offset=0):
    """
    随机打乱schema中表和字段的顺序
    通过seed_offset确保每次调用生成不同的随机序列
    """
    # 使用时间戳 + 偏移量作为随机种子
    random.seed(int(time.time() * 1000) + seed_offset)

    # 深拷贝避免修改原始数据
    shuffled_schema = deepcopy(schema_info)
    # 打乱表顺序
    random.shuffle(shuffled_schema['tables'])

    # 打乱每个表的字段顺序
    for table in shuffled_schema['tables']:
        random.shuffle(table['fields'])

    return shuffled_schema


def generate_shuffled_schemas(original_schema, num_variations=16):
    """
    从JSON格式的字典生成多个随机打乱的schema版本

    参数:
        original_schema: 输入的JSON格式字典
        num_variations: 要生成的打乱版本数量

    返回:
        list: 包含所有打乱后schema的列表
    """
    # 生成指定数量的打乱版本
    shuffled_versions = []
    for i in range(num_variations):
        # 每次调用使用不同的seed_offset
        shuffled_version = shuffle_schema(original_schema, seed_offset=i)
        shuffled_versions.append(shuffled_version)

    return shuffled_versions


def save_shuffled_schemas(output_file, shuffled_schemas):
    """
    将打乱后的schema保存到JSON文件

    参数:
        output_file: 输出文件路径
        shuffled_schemas: 打乱后的schema列表
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shuffled_schemas, f, indent=2)















if __name__ == "__main__":
    # 示例用法
    input_json = 'california_schools.json'

    # 生成5个打乱版本
    shuffled_schemas = generate_shuffled_schemas(input_json, num_variations=16)

    # 打印结果
    for i, schema in enumerate(shuffled_schemas, 1):
        print(f"版本 {i}:")
        print(json.dumps(schema, indent=2))
        print("-" * 50)

