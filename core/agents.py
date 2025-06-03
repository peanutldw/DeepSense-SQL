# -*- coding: utf-8 -*-
import concurrent
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from func_timeout import func_set_timeout, FunctionTimedOut
from core.random_schema import *
from core.utils import  parse_sql_from_string, add_prefix, load_json_file, extract_world_info, is_email, \
    is_valid_date_column
from core.const import *
from copy import deepcopy
from typing import List, Dict, Union, Tuple
import itertools
import sqlite3
import time
import abc
import sys
import os
from tqdm import trange
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
LLM_API_FUC = None
from core.api_config import MODEL_NAME, client
# try import core.api, if error then import core.llm
try:
    from core import api
    LLM_API_FUC = api.safe_call_llm
    print(f"Use func from core.api in agents.py")
except:
    from core import llm
    LLM_API_FUC = llm.safe_call_llm
    print(f"Use func from core.llm in agents.py")

# 检查CUDA是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在使用: {device}")

#加载模型并移动到CUDA（指向您的文件夹路径）
rank_model = SentenceTransformer('./first-stage-ranker').to(device)


def call_llm(prompt, model=MODEL_NAME, temperature=0):
    global MODEL_NAME
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    text = response.choices[0].message.content
    return text
def parse_json(text: str) -> dict:
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            try:
                return json.loads(text[start:end].strip())
            except: pass

    try:
        return json.loads(text)
    except: pass

    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except: pass

    return {}

def get_topk_sql_candidates(
        query: str,
        sql_candidates: List[str],
        rank_model: SentenceTransformer,
        device: str = "cuda",
        top_k: int = 5
) -> List[Dict[str, Union[str, float]]]:
    """
    保留Top-K相似度最高的候选SQL

    参数:
        query: 自然语言查询 (str)
        sql_candidates: 候选SQL列表 (list[str])
        rank_model: 加载好的SBERT模型
        device: 计算设备 (cuda/cpu)
        top_k: 保留的候选数量

    返回:
        list[dict]: 包含SQL和得分的字典列表，按得分降序排列
    """
    if not sql_candidates:
        return []

    # 统一设备
    rank_model = rank_model.to(device)

    # 编码查询和候选SQL
    nl_embedding = rank_model.encode(query, convert_to_tensor=True, device=device)
    sql_embeddings = rank_model.encode(sql_candidates, convert_to_tensor=True, device=device)

    # 计算余弦相似度
    similarities = cos_sim(nl_embedding, sql_embeddings)[0]

    # 组合结果并排序
    scored_sql = [
        {"sql": sql, "score": score.item()}
        for sql, score in zip(sql_candidates, similarities)
    ]
    scored_sql.sort(key=lambda x: -x["score"])

    # 直接返回Top-K（无需阈值过滤）
    return scored_sql[:top_k]

def llm_compare_sql(
        question: str,
        schema: str,
        sql_pair: tuple,
        **word_info
) -> int:
    """
    使用LLM比较两个SQL，返回更优的编号（1或2）
    """
    prompt = f"""# 任务说明
    请根据用户提问，比较以下两个SQL查询的质量，选择更符合用户需求的SQL。   
    # 数据库Schema
    {schema}  
    # 用户提问
    {question}  
    # 候选SQL 1
    {sql_pair[0]}
    # 候选SQL 2
    {sql_pair[1]}  
    请按以下规则比较：
    1. 语法正确性
    2. 语义匹配度（是否准确回答问题）
    3. 结果完整性（是否遗漏关键字段）
    只需返回数字1或2，不要包含其他任何内容："""

    text = LLM_API_FUC(prompt, **word_info, temperature = 0).strip()
    matches = re.finditer(r'(?<!\d)([12])(?!\d)', text)
    last_match = None

    # 遍历所有匹配项，保留最后一个
    for match in matches:
        last_match = match.group(1)

    return int(last_match)


def select_best_sql_via_llm_parallel(
        question: str,
        schema: str,
        candidate_sqls: List[str],
        max_comparisons: int = 28,
        max_workers: int = 15  # 控制并发度
) -> Dict:
    if not candidate_sqls:
        return {"best_sql": "", "comparison_log": ["警告：候选SQL列表为空"]}
    if len(candidate_sqls) == 1:
        return {"best_sql": candidate_sqls[0], "comparison_log": []}

    scores = {i: 0 for i in range(len(candidate_sqls))}
    comparison_log = []
    pairs = list(itertools.combinations(range(len(candidate_sqls)), 2))[:max_comparisons]

    # 并行执行比较任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {
            executor.submit(
                llm_compare_sql,
                question,
                schema,
                (candidate_sqls[idx1], candidate_sqls[idx2])
            ): (idx1, idx2)
            for idx1, idx2 in pairs
        }

        for future in concurrent.futures.as_completed(future_to_pair):
            idx1, idx2 = future_to_pair[future]
            try:
                winner = future.result()
                scores[winner - 1] += 1
                comparison_log.append(f"对比 {idx1 + 1} vs {idx2 + 1}: 胜者 {winner}")
            except Exception as e:
                comparison_log.append(f"对比 {idx1 + 1} vs {idx2 + 1}: 失败 - {str(e)}")

    best_idx = max(scores, key=lambda k: scores[k])
    return {
        "best_sql": candidate_sqls[best_idx],
        "score": scores[best_idx],
        "comparison_log": comparison_log
    }

class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def talk(self, message: dict):
        pass


class Preprocessor(BaseAgent):


    # 定义当前 Agent 的名称，用于在消息机制中标识
    name = PREPROCESSOR_NAME
    # 定义当前 Agent 的描述
    description = "Get database description"

    def __init__(
        self,
        data_path: str,
        tables_json_path: str,
        model_name: str,
        dataset_name: str,
        lazy: bool = False
    ):
        # 调用父类 BaseAgent 的构造函数
        super().__init__()
        # 处理并存储数据路径，去除路径末尾的斜杠或反斜杠
        self.data_path = data_path.strip('/').strip('\\')
        # 表示存储所有数据库对应的表结构信息的 JSON 文件路径
        self.tables_json_path = tables_json_path
        # 使用的模型名称
        self.model_name = model_name
        # 数据集名称，一般是 "spider" 或其他
        self.dataset_name = dataset_name
        # 用来缓存所有数据库的摘要信息，key 是 db_id，值是数据库的描述
        self.db2infos = {}
        # 用来缓存所有数据库对应的 tables.json 中的信息，key 是 db_id，值是 JSON 解析之后的 dict
        self.db2dbjsons = {}
        # 调用方法初始化 db2dbjsons
        self.init_db2jsons()
        # 如果不设置 lazy，则一次性读取并缓存所有数据库的信息
        if not lazy:
            self._load_all_db_info()
        # 用于保存当前 Agent 处理的消息
        self._message = {}


    def init_db2jsons(self):
        # 如果 tables_json_path 不存在，则抛出错误
        if not os.path.exists(self.tables_json_path):
            raise FileNotFoundError(f"tables.json not found in {self.tables_json_path}")
        # 加载 tables.json 文件的内容
        data = load_json_file(self.tables_json_path)
        # 对每个数据库条目进行处理
        for item in data:
            # 取出数据库的 ID
            db_id = item['db_id']

            # 取出表名列表，并在原来的 item 中统计表的数量
            table_names = item['table_names']
            item['table_count'] = len(table_names)

            # 创建一个与 table_names 等长的列表，用于后续统计每张表的列数量
            column_count_lst = [0] * len(table_names)
            # 这里 columns 的结构一般是类似 [ (table_index, column_name), ... ]
            for tb_idx, col in item['column_names']:
                # 如果 tb_idx 有效，则为该表计数 +1
                if tb_idx >= 0:
                    column_count_lst[tb_idx] += 1

            # 找出表中最大列数、总列数、平均列数等信息，并写入 item
            item['max_column_count'] = max(column_count_lst)
            item['total_column_count'] = sum(column_count_lst)
            item['avg_column_count'] = sum(column_count_lst) // len(table_names)

            # 将当前数据库的 item（含表和列等结构化信息）以 db_id 为 key 缓存起来
            self.db2dbjsons[db_id] = item

    def _get_column_attributes(self, cursor, table):
        # 使用 PRAGMA table_info(table_name) 获取表格的列信息
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        columns = cursor.fetchall()

        # 用于构建列信息的列表
        columns_info = []
        # 主键列表
        primary_keys = []
        # 列名列表
        column_names = []
        # 列类型列表
        column_types = []

        # 遍历从数据库中获取到的 columns 结果
        for column in columns:
            # column 一般形如 (cid, name, type, notnull, dflt_value, pk)
            column_names.append(column[1])
            column_types.append(column[2])
            is_pk = bool(column[5])
            # 如果是主键，则加入 primary_keys
            if is_pk:
                primary_keys.append(column[1])
            # 构建每列的元信息字典
            column_info = {
                'name': column[1],        # 列名
                'type': column[2],        # 数据类型
                'not_null': bool(column[3]),   # 是否允许为空
                'primary_key': bool(column[5]) # 是否为主键
            }
            columns_info.append(column_info)

        # 返回列名和列类型这两个最关键的信息
        return column_names, column_types

    def _get_unique_column_values_str(
        self,
        cursor,
        table,
        column_names,
        column_types,
        json_column_names,
        is_key_column_lst
    ):
        """
        根据列名，查询数据库中的去重值，并获取每列部分示例值字符串。
        如果列是主键/外键或其他无意义值，则可能直接忽略或返回空字符串。
        """
        # col_to_values_str_lst: 最终要和 json_column_names 对齐的 [ [col_name, values_str], ... ]
        col_to_values_str_lst = []
        # col_to_values_str_dict: 用于先行查询并存储 (key = 列名, value = 值示例字符串)
        col_to_values_str_dict = {}

        # 收集所有 key 列（主键或外键）
        key_col_list = [json_column_names[i] for i, flag in enumerate(is_key_column_lst) if flag]

        # 列的总数
        len_column_names = len(column_names)

        # 逐列处理
        for idx, column_name in enumerate(column_names):
            # 如果该列在主键/外键中，则跳过（一般不需要示例值）
            if column_name in key_col_list:
                continue

            # 小写列名，若该列名后缀是 id / email / url 则不做取值示例
            lower_column_name = column_name.lower()
            if lower_column_name.endswith('id') or \
               lower_column_name.endswith('email') or \
               lower_column_name.endswith('url'):
                values_str = ''
                col_to_values_str_dict[column_name] = values_str
                continue

            # 拼装 SQL 语句，通过 group by 统计不同值
            sql = f"SELECT `{column_name}` FROM `{table}` GROUP BY `{column_name}` ORDER BY COUNT(*) DESC"
            cursor.execute(sql)
            values = cursor.fetchall()
            # 取出真正的值（fetchall() 返回的是 tuple）
            values = [value[0] for value in values]

            # 默认值为空字符串，如果下面出错就维持为空字符串
            values_str = ''
            try:
                # 尝试获取值示例字符串
                values_str = self._get_value_examples_str(values, column_types[idx])
            except Exception as e:
                print(f"\nerror: get_value_examples_str failed, Exception:\n{e}\n")

            # 存进字典
            col_to_values_str_dict[column_name] = values_str

        # 对 json_column_names 做一次循环，使返回结果与原来列的顺序对应
        for k, column_name in enumerate(json_column_names):
            values_str = ''
            is_key = is_key_column_lst[k]
            # 如果是主键或外键，就无需值示例
            if is_key:
                values_str = ''
            # 如果该列不在外键中，尝试在字典里取
            elif column_name in col_to_values_str_dict:
                values_str = col_to_values_str_dict[column_name]
            else:
                # 如果列名在字典找不到，那可能是异常情况，先做输出确认
                print(col_to_values_str_dict)
                time.sleep(3)
                print(f"error: column_name: {column_name} not found in col_to_values_str_dict")

            # 按照 [列名, 列值示例字符串] 的结构加入列表
            col_to_values_str_lst.append([column_name, values_str])

        # 返回和 json_column_names 对齐后的列名-值示例对
        return col_to_values_str_lst

    def _get_value_examples_str(self, values: List[object], col_type: str):
        """
        传入某一列去重后的 values，以及该列在 sqlite 中的类型，
        返回一个字符串形式的示例值(例如前几个非空值)。如果有 URL、Email 等会直接排除。
        """
        # 如果没有任何值，直接返回空
        if not values:
            return ''

        # 如果值的数量超过 10 且列类型是数值类型，则直接返回空
        if len(values) > 10 and col_type in ['INTEGER', 'REAL', 'NUMERIC', 'FLOAT', 'INT']:
            return ''

        vals = []
        has_null = False
        # 遍历所有值
        for v in values:
            # 如果为空，则标记一下
            if v is None:
                has_null = True
            else:
                tmp_v = str(v).strip()
                # 跳过空字符串
                if tmp_v == '':
                    continue
                else:
                    vals.append(v)

        # 如果 vals 里还是空，返回空字符串
        if not vals:
            return ''

        # 针对文本列，需要对一些无意义或过长的值做排除
        if col_type in ['TEXT', 'VARCHAR']:
            new_values = []
            for v in vals:
                if not isinstance(v, str):
                    new_values.append(v)
                else:
                    # 如果数据集是 spider，先做一下 strip
                    if self.dataset_name == 'spider':
                        v = v.strip()
                    if v == '':
                        continue
                    elif ('https://' in v) or ('http://' in v):
                        # 包含 URL，则不使用
                        return ''
                    elif is_email(v):
                        # 如果是 Email 格式，也不使用
                        return ''
                    else:
                        new_values.append(v)
            vals = new_values

            # 检查最大字符串长度，若超过一定长度也不返回
            tmp_vals = [len(str(a)) for a in vals]
            if not tmp_vals:
                return ''
            max_len = max(tmp_vals)
            if max_len > 50:
                return ''

        # 重新检查，可能再次过滤为空
        if not vals:
            return ''

        # 仅取前 6 个值
        vals = vals[:6]

        # 检查该列是否可能是日期类型，如果是的话，只取一个值示例
        is_date_column = is_valid_date_column(vals)
        if is_date_column:
            vals = vals[:1]

        # 如果出现过 None，则把 None 插在最前面
        if has_null:
            vals.insert(0, None)

        # 组装成字符串输出
        val_str = str(vals)
        return val_str

    def _load_single_db_info(self, db_id: str) -> dict:
        """
        加载单个数据库的详细信息，包括：
        - 每个表的所有列描述（desc_dict）
        - 每个表的列值示例信息（value_dict）
        - 每个表的主键信息（pk_dict）
        - 每个表的外键信息（fk_dict）
        """
        # 用来存储表 -> 列描述信息
        table2coldescription = {}
        # 用来存储表 -> 主键列
        table2primary_keys = {}
        # 用来存储表 -> 外键信息
        table_foreign_keys = {}
        # 用来存储表 -> 列值示例信息
        table_unique_column_values = {}

        # 从缓存的 db2dbjsons 中获取该数据库的所有结构化信息
        db_dict = self.db2dbjsons[db_id]

        # 收集所有（主键+外键）索引
        important_key_id_lst = []
        keys = db_dict['primary_keys'] + db_dict['foreign_keys']
        for col_id in keys:
            # 主键或外键信息可能是单个 int，也可能是多个 int 组成的 list
            if isinstance(col_id, list):
                important_key_id_lst.extend(col_id)
            else:
                important_key_id_lst.append(col_id)

        # 拼装 SQLite 数据库路径
        db_path = f"{self.data_path}/{db_id}/{db_id}.sqlite"
        # 建立连接并获得 cursor
        conn = sqlite3.connect(db_path)
        # 避免编码问题，忽略部分字符
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()

        # 原始的表名列表
        table_names_original_lst = db_dict['table_names_original']

        # 遍历每个表
        for tb_idx, tb_name in enumerate(table_names_original_lst):
            # 取出所有原始列名
            all_column_names_original_lst = db_dict['column_names_original']
            # 取出所有处理后的列名(去掉下划线等)
            all_column_names_full_lst = db_dict['column_names']

            # col2dec_lst 用来存储该表每一列的信息: [orig_col_name, full_col_name, extra_desc]
            col2dec_lst = []

            pure_column_names_original_lst = []
            is_key_column_lst = []

            # 遍历所有列，筛选属于 tb_idx 这张表的列
            for col_idx, (root_tb_idx, orig_col_name) in enumerate(all_column_names_original_lst):
                if root_tb_idx != tb_idx:
                    continue
                # 记录下原始列名
                pure_column_names_original_lst.append(orig_col_name)
                # 判断该列是否在所有主键/外键之中
                if col_idx in important_key_id_lst:
                    is_key_column_lst.append(True)
                else:
                    is_key_column_lst.append(False)
                # 取出该列对应的 “全名”，并进行一定处理
                full_col_name: str = all_column_names_full_lst[col_idx][1]
                full_col_name = full_col_name.replace('_', ' ')
                # 先占位 extra_desc，这里设为空字符串
                cur_desc_obj = [orig_col_name, full_col_name, '']
                col2dec_lst.append(cur_desc_obj)

            # 初始化各信息的存储
            table2coldescription[tb_name] = col2dec_lst
            table_foreign_keys[tb_name] = []
            table_unique_column_values[tb_name] = []
            table2primary_keys[tb_name] = []

            # 调用 _get_column_attributes 得到所有 SQLite 实际列名和类型
            all_sqlite_column_names_lst, all_sqlite_column_types_lst = \
                self._get_column_attributes(cursor, tb_name)

            # 调用 _get_unique_column_values_str 获取列对应的值示例
            col_to_values_str_lst = self._get_unique_column_values_str(
                cursor,
                tb_name,
                all_sqlite_column_names_lst,
                all_sqlite_column_types_lst,
                pure_column_names_original_lst,
                is_key_column_lst
            )
            # 将列值示例信息存入 table_unique_column_values
            table_unique_column_values[tb_name] = col_to_values_str_lst

        # 开始处理该数据库的所有外键
        foreign_keys_lst = db_dict['foreign_keys']
        for from_col_idx, to_col_idx in foreign_keys_lst:
            from_col_name = all_column_names_original_lst[from_col_idx][1]
            from_tb_idx = all_column_names_original_lst[from_col_idx][0]
            from_tb_name = table_names_original_lst[from_tb_idx]

            to_col_name = all_column_names_original_lst[to_col_idx][1]
            to_tb_idx = all_column_names_original_lst[to_col_idx][0]
            to_tb_name = table_names_original_lst[to_tb_idx]

            # 在源表的外键列表中，记录 (源列, 目标表, 目标列)
            table_foreign_keys[from_tb_name].append((from_col_name, to_tb_name, to_col_name))

        # 接着处理所有主键
        for pk_idx in db_dict['primary_keys']:
            # 主键可能是单个 int，也可能是多个 int
            pk_idx_lst = []
            if isinstance(pk_idx, int):
                pk_idx_lst.append(pk_idx)
            elif isinstance(pk_idx, list):
                pk_idx_lst = pk_idx
            else:
                err_message = f"pk_idx: {pk_idx} is not int or list"
                print(err_message)
                raise Exception(err_message)

            # 根据 col_idx 找到表和列名
            for cur_pk_idx in pk_idx_lst:
                tb_idx = all_column_names_original_lst[cur_pk_idx][0]
                col_name = all_column_names_original_lst[cur_pk_idx][1]
                tb_name = table_names_original_lst[tb_idx]
                table2primary_keys[tb_name].append(col_name)

        # 关闭游标
        cursor.close()
        # 做一个简单的延时，可根据需要去掉
        time.sleep(3)

        # 整理返回结果
        result = {
            "desc_dict": table2coldescription,
            "value_dict": table_unique_column_values,
            "pk_dict": table2primary_keys,
            "fk_dict": table_foreign_keys
        }
        return result

    def _load_all_db_info(self):
        """
        一次性地将 data_path 下所有数据库都加载到 self.db2infos 中，
        其中每个数据库的详细信息由 _load_single_db_info(db_id) 提供。
        """
        print("\nLoading all database info...", file=sys.stdout, flush=True)
        # 列出 data_path 下所有文件/文件夹，假设这些都是数据库 ID
        db_ids = [item for item in os.listdir(self.data_path)]
        # 使用 trange 做一个进度条输出
        for i in trange(len(db_ids)):
            db_id = db_ids[i]
            # 加载单个数据库并放入缓存
            db_info = self._load_single_db_info(db_id)
            self.db2infos[db_id] = db_info

    def _build_bird_table_schema_sqlite_str(self, table_name, new_columns_desc, new_columns_val):
        """
        构建类似 SQL DDL 的文本，用于描述表的结构和部分列示例值。
        """
        schema_desc_str = ''
        # 先加上 CREATE TABLE table_name
        schema_desc_str += f"CREATE TABLE {table_name}\n"
        extracted_column_infos = []
        # new_columns_desc: [(orig_col_name, full_col_name, extra_desc), ...]
        # new_columns_val:  [(col_name, col_values_str), ...]
        for (col_name, full_col_name, col_extra_desc), (_, col_values_str) in \
            zip(new_columns_desc, new_columns_val):

            # 构建列描述行，如 " district_id INTEGER PRIMARY KEY, -- location of branch"
            col_line_text = ''
            # 如果有额外描述，则加上 "And ..." 前缀
            col_extra_desc = 'And ' + str(col_extra_desc) if col_extra_desc != '' and str(col_extra_desc) != 'nan' else ''
            # 只保留前 100 个字符
            col_extra_desc = col_extra_desc[:100]

            # 拼接列行
            col_line_text += f"  {col_name},  --"
            if full_col_name != '':
                full_col_name = full_col_name.strip()
                col_line_text += f" {full_col_name},"
            if col_values_str != '':
                col_line_text += f" Value examples: {col_values_str}."
            if col_extra_desc != '':
                col_line_text += f" {col_extra_desc}"
            extracted_column_infos.append(col_line_text)
        # 用花括号包裹每一列信息
        schema_desc_str += '{\n' + '\n'.join(extracted_column_infos) + '\n}' + '\n'
        return schema_desc_str

    def _build_bird_table_schema_list_str(self, table_name, new_columns_desc, new_columns_val):
        """
        以更偏向列表的方式构建表结构描述。
        """
        schema_desc_str = ''
        # 先加上 Table 标识
        schema_desc_str += f"# Table: {table_name}\n"
        extracted_column_infos = []
        # 逐列拼接
        for (col_name, full_col_name, col_extra_desc), (_, col_values_str) in \
            zip(new_columns_desc, new_columns_val):

            # 如果有额外列描述不为空，添加 "And ..."
            col_extra_desc = 'And ' + str(col_extra_desc) if col_extra_desc != '' and str(col_extra_desc) != 'nan' else ''
            col_extra_desc = col_extra_desc[:100]

            col_line_text = ''
            col_line_text += f'  ('
            col_line_text += f"{col_name},"

            if full_col_name != '':
                full_col_name = full_col_name.strip()
                col_line_text += f" {full_col_name}."
            if col_values_str != '':
                col_line_text += f" Value examples: {col_values_str}."
            if col_extra_desc != '':
                col_line_text += f" {col_extra_desc}"
            col_line_text += '),'
            extracted_column_infos.append(col_line_text)

        # 用中括号包围，拼成列表形式
        schema_desc_str += '[\n' + '\n'.join(extracted_column_infos).strip(',') + '\n]' + '\n'
        return schema_desc_str

    def _get_db_desc_str(
            self,
            db_id: str,
            extracted_schema: dict,
            use_gold_schema: bool = False
    ) -> Tuple[dict, str, dict]:
        """
        返回数据库的完整（或筛选后）表结构信息（JSON格式），以及外键信息。
        如果 extracted_schema 不为空，则按照其指定的列进行保留或过滤。

        返回: (schema_json, fk_desc_str, chosen_db_schem_dict)
        """
        # 如果该 db_id 未被加载过则进行懒加载
        if self.db2infos.get(db_id, {}) == {}:
            self.db2infos[db_id] = self._load_single_db_info(db_id)

        db_info = self.db2infos[db_id]
        desc_info = db_info['desc_dict']  # 表 -> 列描述
        value_info = db_info['value_dict']  # 表 -> 列值示例
        pk_info = db_info['pk_dict']  # 表 -> 主键列
        fk_info = db_info['fk_dict']  # 表 -> 外键列(带关联信息)
        type_info = db_info.get('type_dict', {})  # 表 -> 列类型信息

        # 断言这几个结构对同一数据库下的表是一致的
        tables_1, tables_2, tables_3 = desc_info.keys(), value_info.keys(), fk_info.keys()
        assert set(tables_1) == set(tables_2)
        assert set(tables_2) == set(tables_3)

        # 用于构建最终的 JSON schema
        schema_json = {"tables": []}
        db_fk_infos = []  # 收集所有外键关系的字符串
        chosen_db_schem_dict = {}

        for (table_name, columns_desc), \
                (_, columns_val), \
                (fk_table_name, fk_info), \
                (pk_table_name, pk_list) in \
                zip(desc_info.items(), value_info.items(), fk_info.items(), pk_info.items()):

            # 如果 extracted_schema 里没有该表，且 use_gold_schema = True，则跳过
            table_decision = extracted_schema.get(table_name, '')
            if table_decision == '' and use_gold_schema:
                continue

            all_columns = [name for name, _, _ in columns_desc]
            primary_key_columns = [name for name in pk_list]
            foreign_key_columns = [name for name, _, _ in fk_info]
            important_keys = primary_key_columns + foreign_key_columns

            new_columns_desc = []
            new_columns_val = []

            # 根据 table_decision 的不同类型，做不同处理
            if table_decision == "drop_all":
                new_columns_desc = deepcopy(columns_desc[:6])
                new_columns_val = deepcopy(columns_val[:6])
            elif table_decision == "keep_all" or table_decision == '':
                new_columns_desc = deepcopy(columns_desc)
                new_columns_val = deepcopy(columns_val)
            else:
                llm_chosen_columns = table_decision
                append_col_names = []
                for idx, col in enumerate(all_columns):
                    if col in important_keys:
                        new_columns_desc.append(columns_desc[idx])
                        new_columns_val.append(columns_val[idx])
                        append_col_names.append(col)
                    elif col in llm_chosen_columns:
                        new_columns_desc.append(columns_desc[idx])
                        new_columns_val.append(columns_val[idx])
                        append_col_names.append(col)

                if len(all_columns) > 6 and len(new_columns_val) < 6:
                    for idx, col in enumerate(all_columns):
                        if len(append_col_names) >= 6:
                            break
                        if col not in append_col_names:
                            new_columns_desc.append(columns_desc[idx])
                            new_columns_val.append(columns_val[idx])
                            append_col_names.append(col)

            # 构建表结构JSON
            table_json = {
                "table_name": table_name,
                "description": "",  # 可以在这里添加表级别的描述
                "fields": []
            }

            for (col_name, _, col_desc), (_, val_examples) in zip(new_columns_desc, new_columns_val):
                # 获取列类型信息，如果没有则默认为"string"
                col_type = type_info.get(table_name, {}).get(col_name, "string")

                field_json = {
                    "name": col_name,
                    "type": col_type,
                    "description": col_desc
                }

                if val_examples:  # 如果有值示例则添加
                    field_json["examples"] = val_examples

                table_json["fields"].append(field_json)

            schema_json["tables"].append(table_json)
            chosen_db_schem_dict[table_name] = [col_name for col_name, _, _ in new_columns_desc]

            # 收集外键描述信息
            for col_name, to_table, to_col in fk_info:
                from_table = table_name
                if '`' not in str(col_name):
                    col_name = f"`{col_name}`"
                if '`' not in str(to_col):
                    to_col = f"`{to_col}`"
                fk_link_str = f"{from_table}.{col_name} = {to_table}.{to_col}"
                if fk_link_str not in db_fk_infos:
                    db_fk_infos.append(fk_link_str)

        # 将所有外键关系拼接成字符串
        fk_desc_str = '\n'.join(db_fk_infos).strip()
        return schema_json, fk_desc_str, chosen_db_schem_dict

    def talk(self, message: dict):
        """
        当前 Agent 的主要对话接口，根据输入消息（包含 db_id, query, evidence 等），
        返回经过（或未经过）精简的数据库结构描述，以及外键信息。
        :param message: {
            "db_id": database_name,       # 数据库名称
            "query": user_query,          # 用户自然语言查询
            "evidence": extra_info,       # 额外信息
            "extracted_schema": None      # 或者已有的 db->tables 信息
        }
        :return: 修改并返回给下一个 Agent 的 message
        """
        # 如果消息的目标不是当前 Agent，就直接返回
        if message['send_to'] != self.name:
            return

        # 保存当前要处理的消息
        self._message = message
        # 解析出 db_id, 已有的抽取 schema, 用户 query, 以及 evidence
        db_id = message.get('db_id')
        ext_sch = message.get('extracted_schema', {})

        # 获取数据库的描述字符串、外键字符串，以及本次选中的表结构
        db_schema, db_fk, chosen_db_schem_dict = self._get_db_desc_str(
            db_id=db_id,
            extracted_schema=ext_sch
        )

        message['desc_str'] = db_schema
        message['fk_str'] = db_fk
        message['pruned'] = False
        message['send_to'] = GENERATOR_NAME


# class SchemaSorter(BaseAgent):
#     name = SCHEMA_SORTER_NAME
#     description = "Sort tables and columns by relevance to the user query"
#
#     def __init__(self):
#         super().__init__()
#         self._message = {}
#
#     def talk(self, message: dict):
#         if message['send_to'] != self.name:
#             return
#         self._message = message
#
#         query = message.get('query')
#         original_schema = message.get('desc_str')
#         fk_info = message.get('fk_str')
#
#         print("排序前的schema\n",json.dumps(original_schema, ensure_ascii=False, indent=2))
#         prompt = f"""你是一个SQL专家，当前任务是根据用户的自然语言查询，对数据库中的表和字段进行相关性排序。请优先将可能被用于生成SQL的表和字段放在前面。请仅输出符合示例的 ```json 代码块```，不要写任何额外文字。
#
#         # 用户查询
#         {query}
#
#         # 数据库结构描述
#         {json.dumps(original_schema, ensure_ascii=False, indent=2)}
#         外键：
#         {fk_info}
#         请返回一个 JSON 格式，包含表排序顺序（table_order）和每个表下字段的排序顺序（column_order），示例如下：
#         ```json
#         {{
#           "table_order": ["tableA", "tableB"],
#           "column_order": {{
#             "tableA": ["field1", "field2"],
#             "tableB": ["fieldX"]
#           }}
#         }}
#         ```"""
#
#         word_info = extract_world_info(message)
#         reply = LLM_API_FUC(prompt, **word_info)
#
#         try:
#             sort_result = parse_json(reply)
#             schema = original_schema
#             table_order = sort_result.get("table_order", [])
#             column_order = sort_result.get("column_order", {})
#             table_map = {t["table_name"].lower(): t for t in schema.get("tables", [])}
#
#             sorted_tables = []
#             for table_name in table_order:
#                 canonical_name = table_name.lower()
#                 if canonical_name in table_map:
#                     table = table_map[canonical_name]
#                     field_map = {f["name"]: f for f in table["fields"]}
#                     ordered_fields = []
#                     used_fields = set()
#
#                     for col in column_order.get(table_name, []):
#                         if col in field_map:
#                             ordered_fields.append(field_map[col])
#                             used_fields.add(col)
#
#                     for f in table["fields"]:
#                         if f["name"] not in used_fields:
#                             ordered_fields.append(f)
#
#                     table["fields"] = ordered_fields
#                     sorted_tables.append(table)
#                 else:
#                     print(f"[WARNING] table_name '{table_name}' not found in schema")
#
#             schema["tables"] = sorted_tables
#             message["sorted_suggestion"] = sort_result
#             message["sorted_schema"] = schema
#         except Exception as e:
#             print("[ERROR] JSON解析或表匹配失败")
#             print("模型返回内容:", reply)
#             print("异常信息:", str(e))
#             message["sorted_schema"] = {"error": f"解析失败: {str(e)}", "raw_reply": reply}
#         message['send_to'] = GENERATOR_NAME

class Generator(BaseAgent):
    """
    Decompose the question and solve them using CoT
    """
    name = GENERATOR_NAME
    description = "Decompose the question and solve them using three different CoTs, generate 3*3 SQLs, filter them with 2 stages and return the best one"

    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self._message = {}


    def talk(self, message: dict):
        """
        :param self:
        :param message: {"query": user_query,
                        "evidence": extra_info,
                        "desc_str": description of db schema,
                        "fk_str": foreign keys of database}
        :return: final SQL
        """
        if message['send_to'] != self.name: return
        self._message = message
        #ssss
        query, fk_info, original_schema =  message.get('query'), \
                                    message.get('fk_str'), \
                                    message.get('desc_str')
        key_points = message.get(KEY_POINTS_FIELD, "")
        # 生成3个不同顺序的schema
        # schema_infos = generate_shuffled_schemas(original_schema, num_variations=3)
        #schema_info = json.dumps(original_schema, ensure_ascii=False, indent=2)
        schema_info = original_schema
        #print("original_schema\n",original_schema)
        # print("排序后的schema\n",schema_info)
        template1 = decompose_template
        template2 = query_plan_template
        template3 = sql_like_template

        prompts = []

        prompt1 = template1.format(
                query=query,
                desc_str=schema_info,
                fk_str=fk_info,  # 使用当前循环中的 fk_info
                key_points=key_points
            )
        prompts.append(prompt1)

          # 遍历 fk_infos 列表中的每一个外键信息
        prompt2 = template2.format(
                query=query,
                desc_str=schema_info,
                fk_str=fk_info,  # 使用当前循环中的 fk_info
                key_points=key_points
            )
        prompts.append(prompt2)

         # 遍历 fk_infos 列表中的每一个外键信息
        prompt3 = template3.format(
                query=query,
                desc_str=schema_info,
                fk_str=fk_info,  # 使用当前循环中的 fk_info
                key_points=key_points
            )
        prompts.append(prompt3)
        # for schema_info in schema_infos:  # 遍历 fk_infos 列表中的每一个外键信息
        #     prompt1 = template1.format(
        #         query=query,
        #         desc_str=schema_info,
        #         fk_str=fk_info,  # 使用当前循环中的 fk_info
        #         evidence=evidence,
        #         key_points=''
        #     )
        #     prompts.append(prompt1)
        #
        # for schema_info in schema_infos:  # 遍历 fk_infos 列表中的每一个外键信息
        #     prompt2 = template2.format(
        #         query=query,
        #         desc_str=schema_info,
        #         fk_str=fk_info,  # 使用当前循环中的 fk_info
        #         evidence=evidence,
        #         key_points=''
        #     )
        #     prompts.append(prompt2)
        #
        # for schema_info in schema_infos:  # 遍历 fk_infos 列表中的每一个外键信息
        #     prompt3 = template3.format(
        #         query=query,
        #         desc_str=schema_info,
        #         fk_str=fk_info,  # 使用当前循环中的 fk_info
        #         evidence=evidence,
        #         key_points=''
        #     )
        #     prompts.append(prompt3)
        word_info = extract_world_info(self._message)

        def call_llm_api(prompt, word_info, temperature=0.7):
            try:
                reply = LLM_API_FUC(prompt, **word_info, temperature = temperature).strip()
                return reply
            except Exception as e:
                print(f"Error processing prompt '{prompt[:20]}...': {str(e)}")
                return None  # 返回 None 表示失败

        # 并发调用函数
        def batch_call_llm_api(prompts, word_info, max_workers=28):
            # 绑定固定参数（word_info 和 temperature）
            func = partial(call_llm_api, word_info=word_info, temperature=0.7)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                replies1 = list(executor.map(func, prompts))
            return replies1

        replies = batch_call_llm_api(prompts, word_info)
        sql_candidates = [parse_sql_from_string(reply) for reply in replies]

        #print('************sql_candidates',sql_candidates)
        if not sql_candidates:
            print("警告：未获取到任何有效的SQL候选")
            final_sql = parse_sql_from_string(LLM_API_FUC(str(prompts[0]), ** word_info,temperature=0))
            message['final_sql'] = final_sql
            message['send_to'] = REFINER_NAME
            return
        else:
            top_sql = get_topk_sql_candidates(query, sql_candidates, rank_model, device, top_k=6)
            # for j, res in enumerate(top_sql):
            #     print(f"***TOP K个候选SQL:\nRank {j + 1}: Score={res['score']:.4f}\nSQL: {res['sql']}\n")
            sql_list = [item['sql'] for item in top_sql]
            #schema_uni = schema_info + "\nforeign_key information:\n" + fk_info
            schema_uni =  json.dumps(schema_info)+"\nforeign_key information:\n"+fk_info
            result = select_best_sql_via_llm_parallel(query, schema_uni, sql_list)
            best_sql = result.get('best_sql')

            message['final_sql'] = best_sql
            print('***最优结果为：', result)
            message['fixed'] = False
            message['send_to'] = REFINER_NAME


class Refiner(BaseAgent):
    """
    Round 0 : 执行 -> 若失败/空 => 反思 -> 回 Generator
    Round 1~MAX_RETRY_ROUND :
        先调用 LLM 校正新 SQL -> 执行
        若仍失败/空 => 再反思 -> 回 Generator
    任一轮成功且结果非空即结束
    """

    name = REFINER_NAME
    description = "Execute; on failure/empty reflect; after Generator regen, LLM-fix then execute."

    # ---------------- 初始化 ---------------- #
    def __init__(self, data_path: str, exec_timeout: int = 60):
        super().__init__()
        self.data_path = data_path
        self.exec_timeout = exec_timeout
        self._message = {}

    # ---------------- 工具函数 ---------------- #
    @func_set_timeout(120)
    def _execute_sql(self, sql: str, db_id: str):
        db_path = f"{self.data_path}/{db_id}/{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        try:
            cur.execute(sql)
            data = cur.fetchall()
            return {"data": data[:5], "sqlite_error": "", "exception_class": ""}
        except sqlite3.Error as e:
            print("执行出错：\n",e)
            return {"data": [], "sqlite_error": " ".join(e.args), "exception_class": e.__class__.__name__}
        except Exception as e:
            return {"data": [], "sqlite_error": str(e), "exception_class": type(e).__name__}

    def _need_reflection(self, res: dict) -> bool:
        """
        返回 True → 需要反思 / 重新生成
        额外：空结果时塞一个标记 is_empty_result，方便 talk() 做次数控制
        """
        is_empty = len(res["data"]) == 0
        if is_empty:
            res["is_empty_result"] = True
        return res["sqlite_error"] or is_empty

    # ---------- LLM 反思 ---------- #
    def _reflect(self, query: str, schema: str, fk: str, sql: str, res: dict) -> str:
        prompt = f"""你是 SQL 调试专家。以下 SQL 运行出错或结果为空，请反思，并给出 bullet-list 改进要点：
        # 用户查询
        {query}

        # 数据库结构
        {schema}

        # 外键信息
        {fk}

        # 当前 SQL
        {sql}

        # 执行结果 / 错误
        {res}
        #生成的反思（改进要点）的格式：
        <key_points>
        1. ...
        2. ...
        </key_points>
        """

        def extract_key_points_block(text):
            pattern = r"<key_points>.*?</key_points>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0).strip()
            return text
        print("生成的反思：\n",extract_key_points_block(call_llm(prompt)))
        return extract_key_points_block(call_llm(prompt))

    # ---------- LLM SQL 校正 ---------- #
    def _llm_fix(
            self,
            query: str,
            schema: str,
            sql: str,
            fk_str: str,
            sqlite_error: str = "",
            exception_class: str = "",
    ) -> str:

        fix_prompt = f"""
        【Instruction】
        When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
        Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
        When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
        【Constraints】
        - In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
        - In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
        - If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
        - If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
        - If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
        【Query】
        -- {query}
        【Database info】
        {schema}
        【Foreign keys】
        {fk_str}
        【old SQL】
        ```sql
        {sql}
        ```
        【SQLite error】 
        {sqlite_error}
        【Exception class】
        {exception_class}
        
        Now please fixup old SQL and generate new SQL again.
        ```sql
        [correct SQL]
        ```
        """
        fixed_sql = call_llm(fix_prompt)
        print("refiner修正后的SQL：\n", parse_sql_from_string(fixed_sql))
        return parse_sql_from_string(fixed_sql)

    def _safe_exec(self, sql: str, db_id: str):
        try:
            return self._execute_sql(sql, db_id)
        except FunctionTimedOut:
            return {"data": [],
                    "sqlite_error": "FunctionTimedOut",
                    "exception_class": "FunctionTimedOut"}

    # ---------------- 主入口 ---------------- #
    def talk(self, message: dict):
        if message.get("send_to") != self.name:
            return
        self._message = message

        db_id = message["db_id"]
        query = message["query"]
        schema_info = message.get("desc_str")
        fk_info = message.get("fk_str", "")
        sql_in = message.get("pred", message.get("final_sql"))
        round_idx = message.get("try_times", 0)

        if round_idx > MAX_RETRY_ROUND:
            message.update({"send_to": SYSTEM_NAME,
                            "stop_reason": "exceed_max_retry"})
            return

        # ---------- Step 1 · 先执行原 SQL ----------
        exec_res = self._safe_exec(sql_in, db_id)


        # ---------- Step 2 · 失败 / 空结果 → 尝试 fix ----------
        sql_to_exec = sql_in
        if self._need_reflection(exec_res):
            fixed_sql = self._llm_fix(query, schema_info, sql_in, fk_info)
            if fixed_sql.strip():
                sql_to_exec = fixed_sql
                exec_res = self._safe_exec(sql_to_exec, db_id)

        # 这里需要修改为根据用户反馈
        # ---------- Step 3 · 仍失败 / 空 → 反思 ----------
        need_reflect = self._need_reflection(exec_res)

        # 处理空结果重试上限（同旧逻辑）
        if exec_res.get("is_empty_result"):
            empty_retry_cnt = message.get("empty_retry_cnt", 0)
            if empty_retry_cnt >= MAX_EMPTY_RETRY:
                need_reflect = False
            else:
                message["empty_retry_cnt"] = empty_retry_cnt + 1

        if need_reflect:
            key_points = self._reflect(query, schema_info, fk_info,
                                       sql_to_exec, exec_res)
            message.update({
                "pred": sql_to_exec,
                KEY_POINTS_FIELD: key_points,
                "try_times": round_idx + 1,
                "send_to": GENERATOR_NAME,
            })
        else:
            message.update({
                "pred": sql_to_exec,
                "answer_data": exec_res["data"],
                "try_times": round_idx + 1,
                "send_to": SYSTEM_NAME,
            })

if __name__ == "__main__":
    m = 0