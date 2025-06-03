MAX_ROUND = 3  # max try times of one agent talk
PREPROCESSOR_NAME = 'Preprocessor'
GENERATOR_NAME = 'Generator'
REFINER_NAME = 'Refiner'
SYSTEM_NAME = 'System'
KEY_POINTS_FIELD = 'key_points'
subq_pattern = r"Sub question\s*\d+\s*:"
MAX_RETRY_ROUND = 2
MAX_EMPTY_RETRY = 1          # 空结果只允许额外重试一次
decompose_template = """
Given a 【Database schema】 description, a knowledge 【Evidence】 and the 【Question】, you need to use valid SQLite and understand the database and knowledge, and then decompose the question into subquestions for text-to-SQL generation.
When generating SQL, we should always consider constraints:
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values

{key_points}
==========

【Database schema】
# Table: frpm
[
  (CDSCode, CDSCode. Value examples: ['01100170109835', '01100170112607'].),
  (Charter School (Y/N), Charter School (Y/N). Value examples: [1, 0, None]. And 0: N;. 1: Y),
  (Enrollment (Ages 5-17), Enrollment (Ages 5-17). Value examples: [5271.0, 4734.0].),
  (Free Meal Count (Ages 5-17), Free Meal Count (Ages 5-17). Value examples: [3864.0, 2637.0]. And eligible free rate = Free Meal Count / Enrollment)
]
# Table: satscores
[
  (cds, California Department Schools. Value examples: ['10101080000000', '10101080109991'].),
  (sname, school name. Value examples: ['None', 'Middle College High', 'John F. Kennedy High', 'Independence High', 'Foothill High'].),
  (NumTstTakr, Number of Test Takers in this school. Value examples: [24305, 4942, 1, 0, 280]. And number of test takers in each school),
  (AvgScrMath, average scores in Math. Value examples: [699, 698, 289, None, 492]. And average scores in Math),
  (NumGE1500, Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500. Value examples: [5837, 2125, 0, None, 191]. And Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500. . commonsense evidence:. . Excellence Rate = NumGE1500 / NumTstTakr)
]
【Foreign keys】
frpm.`CDSCode` = satscores.`cds`
【Question】
List school names of charter schools with an SAT excellence rate over the average.
【Evidence】
Charter schools refers to `Charter School (Y/N)` = 1 in the table frpm; Excellence rate = NumGE1500 / NumTstTakr


Decompose the question into sub questions, considering 【Constraints】, and generate the SQL after thinking step by step:
Sub question 1: Get the average value of SAT excellence rate of charter schools.
SQL
```sql
SELECT AVG(CAST(T2.`NumGE1500` AS REAL) / T2.`NumTstTakr`)
    FROM frpm AS T1
    INNER JOIN satscores AS T2
    ON T1.`CDSCode` = T2.`cds`
    WHERE T1.`Charter School (Y/N)` = 1
```

Sub question 2: List out school names of charter schools with an SAT excellence rate over the average.
SQL
```sql
SELECT T2.`sname`
  FROM frpm AS T1
  INNER JOIN satscores AS T2
  ON T1.`CDSCode` = T2.`cds`
  WHERE T2.`sname` IS NOT NULL
  AND T1.`Charter School (Y/N)` = 1
  AND CAST(T2.`NumGE1500` AS REAL) / T2.`NumTstTakr` > (
    SELECT AVG(CAST(T4.`NumGE1500` AS REAL) / T4.`NumTstTakr`)
    FROM frpm AS T3
    INNER JOIN satscores AS T4
    ON T3.`CDSCode` = T4.`cds`
    WHERE T3.`Charter School (Y/N)` = 1
  )
```

Question Solved.

==========

【Database schema】
# Table: account
[
  (account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
# Table: client
[
  (client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
  (gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),
  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].)
]
# Table: district
[
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (A4, number of inhabitants . Value examples: ['95907', '95616', '94812'].),
  (A11, average salary. Value examples: [12541, 11277, 8114].)
]
【Foreign keys】
account.`district_id` = district.`district_id`
client.`district_id` = district.`district_id`
【Question】
What is the gender of the youngest client who opened account in the lowest average salary branch?
【Evidence】
Later birthdate refers to younger age; A11 refers to average salary

Decompose the question into sub questions, considering 【Constraints】, and generate the SQL after thinking step by step:
Sub question 1: What is the district_id of the branch with the lowest average salary?
SQL
```sql
SELECT `district_id`
  FROM district
  ORDER BY `A11` ASC
  LIMIT 1
```

Sub question 2: What is the youngest client who opened account in the lowest average salary branch?
SQL
```sql
SELECT T1.`client_id`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1
```

Sub question 3: What is the gender of the youngest client who opened account in the lowest average salary branch?
SQL
```sql
SELECT T1.`gender`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1 
```
Question Solved.

==========

【Database schema】
{desc_str}
【Foreign keys】
{fk_str}
【Question】
{query}

Decompose the question into sub questions, considering 【Constraints】and 【Key points】, and generate the SQL after thinking step by step:
"""
query_plan_template = """
**Role**: You are a SQL query planner. Given a database schema and a natural language question, generate a step-by-step execution plan mimicking how a database engine processes the query, then output the final optimized SQL.

{key_points}

**Example
Database Info:
### Table: account
[
  (account_id, PRIMARY KEY, the id of the account. Value examples: [11382, 11362, 2, 1, 2367]),
  (district_id, FOREIGN KEY, location of branch. Value examples: [77, 76, 2, 1, 39]),
  (frequency, VARCHAR(20), frequency of the account. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU']),
  (date, DATE, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'])
]
### Table: client
[
  (client_id, PRIMARY KEY, the unique identifier. Value examples: [13998, 13971, 2, 1, 2839]),
  (gender, CHAR(1), gender (F: female, M: male). Value examples: ['M', 'F']),
  (birth_date, DATE, birth date. Value examples: ['1987-09-27', '1986-08-13']),
  (district_id, FOREIGN KEY, location of branch. Value examples: [77, 76, 2, 1, 39])
]
### Table: district
[
  (district_id, PRIMARY KEY, location of branch. Value examples: [77, 76, 2, 1, 39]),
  (A4, VARCHAR(10), number of inhabitants. Value examples: ['95907', '95616', '94812']),
  (A11, INT, average salary. Value examples: [12541, 11277, 8114, 8110, 8814])
]

## Foreign Keys
account.district_id → district.district_id
client.district_id → district.district_id
 **************************
 Answer Repeating the question, and generating the SQL with a query plan.
 **Question**: How many Thai restaurants can be found in San Pablo Ave, Albany?
 **Evidence**: Thai restaurant refers to food_type = ’thai’; San Pablo Ave Albany refers to street_name
 = ’san pablo ave’ AND T1.city = ’albany’
 **Query Plan**:
 ** Preparation Steps:**
 1. Initialize the process: Start preparing to execute the query.
 2. Prepare storage: Set up storage space (registers) to hold temporary results, initializing them to NULL.
 3. Open the location table: Open the location table so we can read from it.
 4. Open the generalinfo table: Open the generalinfo table so we can read from it.
 ** Matching Restaurants:**
 1. Start reading the location table: Move to the first row in the location table.
 2. Check if the street matches: Look at the street_name column of the current row in location. If it’s not
 "san pablo ave," skip this row.
 3. Identify the matching row: Store the identifier (row ID) of this location entry.
 4. Find the corresponding row in generalinfo: Use the row ID from location to directly find the matching
 row in generalinfo.
 5. Check if the food type matches: Look at the food_type column in generalinfo. If it’s not "thai," skip
 this row.
 6. Check if the city matches: Look at the city column in generalinfo. If it’s not "albany," skip this row.
 ** Counting Restaurants:**
 1. Prepare to count this match: If all checks pass, prepare to include this row in the final count.
 2. Count this match: Increment the count for each row that meets all the criteria.
 3. Move to the next row in location: Go back to the location table and move to the next row, repeating
 the process until all rows are checked.
 4. Finalize the count: Once all rows have been checked, finalize the count of matching rows.
 5. Prepare the result: Copy the final count to prepare it for output.
 ** Delivering the Result:**
 1. Output the result: Output the final count, which is the number of restaurants that match all the
 specified criteria.
 2. End the process: Stop the query execution process.
 3. Setup phase: Before starting the actual query execution, the system prepares the specific values it will
 be looking for, like "san pablo ave," "thai," and "albany."
 **Final Optimized SQL Query:**
```sql
 SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2
 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = ’thai’ AND T1.city = ’albany’ AND
 T2.street_name = ’san pablo ave’
```
question solved!
---
Database Info:
{desc_str}
foreign keys:
{fk_str}
**************************
Answer Repeating the question, and generating the SQL with a query plan.
 **Question: {query}
"""

sql_like_template = """
# 任务说明
你是一个专业的数据工程师，需要将自然语言问题转换为SQL-Like中间语言，最终生成符合语法的SQL。
# 生成规则
1. 首先生成SQL-Like中间语言（忽略JOIN语法和函数格式）
2. 最终SQL必须严格遵循sqlite3语法
3. 所有列名必须来自上述数据库结构

{key_points}

# 示例
#user's question:What is the gender of the youngest client who opened account in the lowest average salary branch?
#evidence：Later birthdate refers to younger age; A11 refers to average salary
#Database schema:
Table: account
[
(account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
(district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
(frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
(date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
Table: client
[
(client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
(gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),
(birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
(district_id, location of branch. Value examples: [77, 76, 2, 1, 39].)
]
Table: district
[
(district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
(A4, number of inhabitants . Value examples: ['95907', '95616', '94812'].),
(A11, average salary. Value examples: [12541, 11277, 8114].)
]
【Foreign keys】
account.district_id = district.district_id
client.district_id = district.district_id
#回答如下：
#reason: We need to find the gender of the youngest client who opened an account in the branch with the lowest average salary.
#columns: client.gender, client.birth_date, account.date, district.A11
#values: None (all values are derived from the database)
#SELECT: client.gender
#SQL-like: SHOW client.gender WHERE account.date = MIN(account.date) AND district.A11 = MIN(district.A11) ORDER BY client.birth_date DESC LIMIT 1
#SQL:
```sql
SELECT client.gender
FROM client
INNER JOIN account ON client.client_id = account.account_id
INNER JOIN district ON client.district_id = district.district_id
WHERE district.A11 = (SELECT MIN(A11) FROM district)
ORDER BY client.birth_date DESC
LIMIT 1
```
#user's question:{query}
#Database schema:
{desc_str}
Foreign keys: 
{fk_str}
"""

refiner_template = """
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
{desc_str}
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