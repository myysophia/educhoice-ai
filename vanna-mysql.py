from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat
from vanna.qianwen import QianWenAI_Chat

from vanna.flask import VannaFlaskApp
import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
#     def __init__(self, config=None):
#         ChromaDB_VectorStore.__init__(self, config=config)
#         OpenAI_Chat.__init__(self, config=config)

class MyVanna(ChromaDB_VectorStore, QianWenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': config['openai']['api_key'], 'model': config['openai']['model']})

# class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
#     def __init__(self, config=None):
#         ChromaDB_VectorStore.__init__(self, config=config)
#         GoogleGeminiChat.__init__(self, config={'api_key': GEMINI_API_KEY, 'model': GEMINI_MODEL})

# vn = MyVanna()

db_config = config['database']
vn.connect_to_mysql(
    host=db_config['host'],
    dbname=db_config['dbname'],
    user=db_config['user'],
    password=db_config['password'],
    port=db_config['port']
)

# The information schema query may need some tweaking depending on your database. This is a good starting point.
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS where TABLE_SCHEMA='school'")

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
plan

# If you like the plan, then uncomment this and run it to train
# vn.train(plan=plan)
# The following are methods for adding training data. Make sure you modify the examples to match your database.



# Import Vanna (assuming you have it initialized as 'vn')
# from vanna.remote import VannaDefault
# vn = VannaDefault(model='your_model_name', api_key='your_api_key') # Example initialization

# --- Add DDL Statements ---
# DDL statements help Vanna understand the database schema (tables, columns, types, relationships).

print("Training DDL for schools table...")
vn.train(ddl="""
    create table schools (
        id                             int auto_increment primary key,
        name                           varchar(255) not null unique comment '学校名称',
        brief_introduction             text         null comment '学校简介',
        school_code                    varchar(100) null unique comment '学校代码 (官方)',
        master_point                   int          null comment '硕士点数量',
        phd_point                      int          null comment '博士点数量',
        research_project               int          null comment '重点研究项目数量 (if applicable)',
        title_double_first_class       tinyint(1)   null comment '是否双一流: 1是, 0否',
        title_985                      tinyint(1)   null comment '是否985: 1是, 0否',
        title_211                      tinyint(1)   null comment '是否211: 1是, 0否',
        title_college                  tinyint(1)   null comment '是否专科: 1是, 0否',
        title_undergraduate            tinyint(1)   null comment '是否本科: 1是, 0否',
        region                         varchar(255) null comment '学校所在地区或省份',
        website                        varchar(255) null comment '学校官网地址',
        recruitment_phone              varchar(100) charset utf8mb4 null comment '招生咨询电话',
        email                          varchar(100) null comment '招生咨询邮箱',
        promotion_rate                 varchar(50)  null comment '升学率',
        abroad_rate                    varchar(50)  null comment '出国率',
        employment_rate                varchar(50)  null comment '就业率',
        double_first_class_disciplines text         null comment '双一流建设学科列表 (逗号分隔)'
    );
""")

print("Training DDL for scores table...")
vn.train(ddl="""
    create table scores (
        id          int auto_increment primary key,
        school_id   int         null comment '关联的学校ID (FK to schools.id)',
        location    int         null comment '生源地代码或标识 (需要文档解释具体含义)',
        year        int         null comment '分数对应的年份 (e.g., 2023)',
        type_id     int         null comment '科类ID (e.g., 1代表理科, 2代表文科, 需要文档确认)',
        tag         varchar(50) null comment '招生类型或标签 (e.g., 普通批, 国家专项, 提前批)',
        lowest      int         null comment '该年份该学校该批次最低录取分数 (投档线)',
        lowest_rank int         null comment '该年份该学校该批次最低录取分数对应的全省排名/位次',
        sg_name     varchar(50) null comment '选科组名称 (新高考模式下, e.g., 物理组)',
        batch_name  varchar(50) null comment '录取批次名称 (e.g., 本科一批, 本科二批)',
        constraint scores_school_id_IDX unique (school_id, location, type_id, year, tag, sg_name, batch_name),
        constraint scores_ibfk_1 foreign key (school_id) references schools (id)
    );
""")
# Note: Indices are usually not needed for Vanna training but kept here for completeness if desired.
# create index idx_scores_lowest on scores (lowest);
# create index idx_scores_lowest_rank on scores (lowest_rank);
# create index idx_scores_year on scores (year);


print("Training DDL for major_score_his table...")
vn.train(ddl="""
    create table major_score_his (
        id                  varchar(50)  not null comment '专业分数记录的唯一ID' primary key,
        school_id           int          not null comment '关联的学校ID (FK to schools.id)',
        special_id          int          null comment '专业ID (可能来自其他专业基础表)',
        spe_id              int          null comment '专业ID (冗余或不同来源, 建议统一)',
        year                int          null comment '分数对应的年份',
        sp_name             varchar(255) null comment '专业标准名称 (e.g., 计算机科学与技术)',
        spname              varchar(500) null comment '专业详细名称 (可能包含方向, e.g., 计算机科学与技术(人工智能方向))',
        info                varchar(255) null comment '专业其他信息备注',
        local_province_name varchar(50)  null comment '招生省份 (e.g., 陕西省)',
        local_type_name     varchar(50)  null comment '招生科类 (e.g., 理科, 文科, 物理类, 历史类)',
        local_batch_name    varchar(50)  null comment '招生批次 (e.g., 本科提前批, 本科一批)',
        level2_name         varchar(255) null comment '专业所属的二级学科名称 (学科门类)',
        level3_name         varchar(255) null comment '专业所属的三级学科名称 (具体学科)',
        average             int          null comment '该专业当年录取平均分',
        max                 int          null comment '该专业当年录取最高分',
        min                 int          null comment '该专业当年录取最低分',
        min_section         varchar(50)  null comment '该专业当年录取最低位次/排名',
        proscore            int          null comment '专业投档线 (可能与min含义相同或略有不同)',
        is_top              int          null comment '是否重点专业: 1是, 0否',
        is_score_range      int          null comment '分数是否为区间形式: 1是, 0否',
        min_range           varchar(50)  null comment '最低分区间 (当is_score_range=1时)',
        min_rank_range      varchar(50)  null comment '最低位次区间 (当is_score_range=1时)',
        remark              varchar(255) null comment '备注信息'
    ) comment '专业历年分数表' charset = utf8mb4;
""")
# create index idx_major_scores_his_school_id on major_score_his (school_id);

print("Training DDL for school_plan_his table...")
vn.train(ddl="""
    create table school_plan_his (
        id               bigint unsigned auto_increment primary key,
        school_id        int               not null comment '关联的学校ID (FK to schools.id)',
        year             int               not null comment '计划对应的年份',
        sp_name          varchar(255)      not null comment '专业名称',
        spname           text              null comment '专业详细名称 (可能包含方向)',
        num              int     default 0 not null comment '该专业当年计划招生人数',
        length           varchar(50)       null comment '学制 (e.g., 4年, 5年)',
        tuition          varchar(50)       null comment '学费 (e.g., 5000元/年, 或具体数字)',
        province_name    varchar(50)       null comment '招生计划对应的省份',
        special_group    tinyint default 0 null comment '特殊类型 (e.g., 0普通, 1艺术类, 2体育类)',
        local_batch_name varchar(50)       null comment '招生批次名称',
        local_type_name  varchar(50)       null comment '招生科类名称'
    ) comment '学校历年招生计划表' charset = utf8mb4;
""")
# create index idx_school_year on school_plan_his (school_id, year);


# --- Add Documentation ---
# Documentation helps Vanna understand business logic, column definitions, and terminology.

print("Training documentation...")
vn.train(documentation="""
- `schools` 表包含学校的基本静态信息，如名称、代码、是否985/211/双一流 (`title_985`, `title_211`, `title_double_first_class` 值为1表示是)、硕士点 (`master_point`) 和博士点 (`phd_point`) 数量、所在地区 (`region`)、官网 (`website`) 和联系方式 (`recruitment_phone`, `email`)。
- `scores` 表记录学校**整体**的**最低录取分数线 (`lowest`)** 和**最低位次 (`lowest_rank`)**。每年、每个学校、每个生源地 (`location`)、每种科类 (`type_id`)、每个批次 (`batch_name`) 可能有多条记录，通过 `tag` 区分招生类型（如普通批）。`type_id=1` 通常指理科，`type_id=2` 通常指文科，具体需要根据数据确认。位次 (`lowest_rank`) 数字越小表示排名越靠前。
- `major_score_his` 表记录**具体专业**的录取分数详情，包括最低分 (`min`)、最高分 (`max`)、平均分 (`average`) 和最低位次 (`min_section`)。`sp_name` 是专业名称。`local_province_name` 是招生省份，`local_type_name` 是招生科类（如文科、理科、物理类），`local_batch_name` 是招生批次。
- `school_plan_his` 表记录**具体专业**的**招生计划**信息，核心是计划招生人数 (`num`)、学制 (`length`) 和学费 (`tuition`)。
- 查询专业分数和计划时，需要将 `major_score_his` 和 `school_plan_his` 通过 `school_id`, `year`, `sp_name`,`spname`, `local_batch_name`, `local_type_name` 进行关联。
- '985大学' 指 `schools.title_985 = 1` 的学校。
- '211大学' 指 `schools.title_211 = 1` 的学校。
- '双一流大学' 指 `schools.title_double_first_class = 1` 的学校。
- '分数线' 通常指最低录取分数，对应 `scores.lowest` (校线) 或 `major_score_his.min` (专业线)。
- '位次' 通常指最低录取位次，对应 `scores.lowest_rank` (校线) 或 `major_score_his.min_section` (专业线)。
- '科别' 或 '科类' 指文科/理科或新高考选科组合，对应 `scores.type_id` 或 `major_score_his.local_type_name` 或 `school_plan_his.local_type_name`。
""")


# --- Add Question-SQL Pairs ---
# These pairs teach Vanna how to translate natural language questions into specific SQL queries.
# Include variations of questions that map to the same SQL.

print("Training Question-SQL pairs...")

# Pair 1 (Based on your SQL query 1)
vn.train(
    question="查询西安航空学院2023年本科二批的录取分数和学校的详细信息",
    sql="""
SELECT
    sch.name AS 学校名称, sch.brief_introduction AS 学校简介, sch.school_code AS 学校代码,
    sch.master_point AS 硕士点, sch.phd_point AS 博士点, sch.title_985, sch.title_211,
    sch.region AS 地区, sch.website AS 学校官网, sch.recruitment_phone AS 学校招生电话,
    sch.email AS 学校招生邮箱, sch.double_first_class_disciplines AS 学校一级学科,
    scores.year AS 年份,
    CASE WHEN scores.type_id = 1 THEN '理科' WHEN scores.type_id = 2 THEN '文科' ELSE '未知科类' END AS 科别,
    scores.lowest AS 最低校分, scores.lowest_rank AS 最低校位次, scores.batch_name AS 批次,
    scores.tag AS 招生类型
FROM schools sch
INNER JOIN scores ON sch.id = scores.school_id
WHERE sch.name = '西安航空学院' AND scores.batch_name = '本科二批' AND scores.year = 2023;
"""
)

vn.train(
    question="西安交大2020-2024年计算机专业录取分数?",
    sql="""
SELECT
    sch.name AS 学校名称,
    msh.year AS 年份,
    msh.spname AS 专业名称,
    msh.min AS 专业最低分,
    msh.min_section AS 专业最低位次,
    msh.average AS 专业平均分,
    msh.max AS 专业最高分
FROM schools sch
INNER JOIN major_score_his msh ON sch.id = msh.school_id
WHERE sch.name like '%西安交通大学%' AND msh.spname like  '%计算机%' AND msh.year BETWEEN 2020 AND 2024
ORDER BY msh.year;
"""
)
vn.train(
    question="西安航空学院2023年本科二批的最低分和最低位次是多少？学校官网和电话呢？",
    sql="""
SELECT
    sch.name AS 学校名称, sch.website AS 学校官网, sch.recruitment_phone AS 学校招生电话,
    scores.year AS 年份,
    CASE WHEN scores.type_id = 1 THEN '理科' WHEN scores.type_id = 2 THEN '文科' ELSE '未知科类' END AS 科别,
    scores.lowest AS 最低校分, scores.lowest_rank AS 最低校位次, scores.batch_name AS 批次
FROM schools sch
INNER JOIN scores ON sch.id = scores.school_id
WHERE sch.name = '西安航空学院' AND scores.batch_name = '本科二批' AND scores.year = 2023;
"""
) # Example of a simpler SQL for a simpler question, Vanna can learn both

# Pair 2 (Based on your SQL query 2)
vn.train(
    question="查询西安交通大学2023年各个专业的录取分数、最低位次、平均分、计划录取人数、学制和学费",
    sql="""
SELECT
    sch.name AS 学校名称,
    scohis.sp_name AS 专业名称, scohis.local_province_name AS 招生省份,
    scohis.local_batch_name AS 招生批次, scohis.local_type_name AS 招生类别,
    scohis.year AS 年份, scohis.min AS 专业最低分, scohis.min_section AS 专业最低位次,
    scohis.max AS 专业最高分, scohis.average AS 专业平均分, scohis.proscore AS 投档线,
    plan.num AS 计划录取人数, plan.length AS 学制, plan.province_name AS 学校所在省份,
    plan.tuition AS 学费
FROM schools sch
INNER JOIN major_score_his scohis ON sch.id = scohis.school_id
INNER JOIN school_plan_his plan ON plan.school_id = sch.id AND plan.sp_name = scohis.sp_name AND scohis.year = plan.year AND scohis.local_batch_name = plan.local_batch_name AND scohis.local_type_name = plan.local_type_name
WHERE sch.name = '西安交通大学' AND scohis.year = 2023
ORDER BY scohis.min DESC;
"""
)
vn.train(
    question="西安交通大学2023年各专业招多少人？分数线怎么样？学费多少？",
    sql="""
SELECT
    sch.name AS 学校名称,
    scohis.sp_name AS 专业名称,
    scohis.local_type_name AS 招生类别,
    scohis.year AS 年份, scohis.min AS 专业最低分, scohis.min_section AS 专业最低位次,
    plan.num AS 计划录取人数, plan.tuition AS 学费
FROM schools sch
INNER JOIN major_score_his scohis ON sch.id = scohis.school_id
INNER JOIN school_plan_his plan ON plan.school_id = sch.id AND plan.sp_name = scohis.sp_name AND scohis.year = plan.year AND scohis.local_batch_name = plan.local_batch_name AND scohis.local_type_name = plan.local_type_name
WHERE sch.name = '西安交通大学' AND scohis.year = 2023
ORDER BY scohis.min DESC;
"""
) # Another example of simpler SQL for a slightly different question

vn.train(
    question="西安交通大学2023年文科专业的录取分数和计划人数是多少？",
    sql="""
SELECT
    sch.name AS 学校名称,
    scohis.sp_name AS 专业名称, scohis.local_province_name AS 招生省份,
    scohis.local_batch_name AS 招生批次, scohis.local_type_name AS 招生类别,
    scohis.year AS 年份, scohis.min AS 专业最低分, scohis.min_section AS 专业最低位次,
    plan.num AS 计划录取人数
FROM schools sch
INNER JOIN major_score_his scohis ON sch.id = scohis.school_id
INNER JOIN school_plan_his plan ON plan.school_id = sch.id AND plan.sp_name = scohis.sp_name AND scohis.year = plan.year AND scohis.local_batch_name = plan.local_batch_name AND scohis.local_type_name = plan.local_type_name
WHERE sch.name = '西安交通大学' AND scohis.year = 2023 AND scohis.local_type_name = '文科'
ORDER BY scohis.min DESC;
"""
)

# Pair 3 (Based on your SQL query 3)
vn.train(
    question="我理科预估考699分，位次大约是5名，请根据2022到2024年的数据，推荐一下分数或位次比较接近的学校",
    sql="""
SELECT DISTINCT
    sch.name AS 学校名称, sch.brief_introduction AS 学校简介, sch.school_code AS 学校代码,
    sch.master_point AS 硕士点, sch.phd_point AS 博士点, sch.title_985, sch.title_211,
    sch.region AS 所属省份, sch.website AS 学校官网, sch.recruitment_phone AS 学校招生电话,
    sch.email AS 学校招生邮箱, sch.double_first_class_disciplines AS 学校一级学科,
    scores.year AS 年份, scores.lowest AS 最低校分, scores.lowest_rank AS 最低校位次,
    scores.batch_name AS 批次,
    CASE WHEN scores.type_id = 1 THEN '理科' WHEN scores.type_id = 2 THEN '文科' ELSE '未知科类' END AS 科别
FROM schools sch
INNER JOIN scores ON sch.id = scores.school_id
WHERE scores.year BETWEEN 2022 AND 2024
  AND scores.type_id = 1 -- Assuming 1 is 理科
  AND (scores.lowest BETWEEN 699 - 70 AND 699 + 70 OR scores.lowest_rank BETWEEN 5 - 2 AND 5 + 4) -- Example range based on input
ORDER BY ABS(scores.lowest - 699) ASC, ABS(scores.lowest_rank - 5) ASC;
"""
) # Note: The range logic in SQL might need adjustment based on how Vanna interprets "接近" (close to). This is one interpretation.

vn.train(
    question="理科生，分数699，位次3到9，找找近三年哪些大学比较合适？",
    sql="""
SELECT DISTINCT
    sch.name AS 学校名称,
    scores.year AS 年份, scores.lowest AS 最低校分, scores.lowest_rank AS 最低校位次,
    scores.batch_name AS 批次,
    CASE WHEN scores.type_id = 1 THEN '理科' WHEN scores.type_id = 2 THEN '文科' ELSE '未知科类' END AS 科别
FROM schools sch
INNER JOIN scores ON sch.id = scores.school_id
WHERE scores.year BETWEEN YEAR(CURDATE()) - 2 AND YEAR(CURDATE()) -- Example for '近三年'
  AND scores.type_id = 1 -- Assuming 1 is 理科
  AND (scores.lowest >= 699 OR scores.lowest_rank <= 9) -- Simplified logic, might need refinement
ORDER BY scores.lowest_rank ASC, scores.lowest DESC;
"""
) # Another interpretation focusing on rank

# --- Additional General Use Cases ---

print("Training additional general Q-SQL pairs...")

vn.train(
    question="北京大学的学校代码和官网是什么？",
    sql="SELECT school_code, website FROM schools WHERE name = '北京大学';"
)
vn.train(
    question="列出所有985大学的名称和所在地区",
    sql="SELECT name, region FROM schools WHERE title_985 = 1;"
)
vn.train(
    question="哪些学校既是985又是211？",
    sql="SELECT name FROM schools WHERE title_985 = 1 AND title_211 = 1;"
)
vn.train(
    question="清华大学2022年理科本科一批的最低录取分数和位次是多少？",
    sql="""
SELECT lowest, lowest_rank
FROM scores s
JOIN schools sch ON s.school_id = sch.id
WHERE sch.name = '清华大学' AND s.year = 2022 AND s.type_id = 1 AND s.batch_name = '本科一批';
"""
)
vn.train(
    question="查询复旦大学'计算机科学与技术'专业2023年在上海招生的最低分是多少？",
    sql="""
SELECT min
FROM major_score_his msh
JOIN schools sch ON msh.school_id = sch.id
WHERE sch.name = '复旦大学' AND msh.sp_name = '计算机科学与技术' AND msh.year = 2023 AND msh.local_province_name = '上海';
"""
)
vn.train(
    question="浙江大学2023年计划招生总人数是多少？",
    sql="""
SELECT SUM(num) AS total_enrollment
FROM school_plan_his plan
JOIN schools sch ON plan.school_id = sch.id
WHERE sch.name = '浙江大学' AND plan.year = 2023;
"""
)
vn.train(
    question="查询所有学校的名称及其博士点数量",
    sql="SELECT name, phd_point FROM schools ORDER BY phd_point DESC;"
)

print("Training complete.")

# After running these, you can ask Vanna questions like:
# vn.ask("What is the website for Tsinghua University?")
# vn.ask("List 985 schools in Beijing")
# vn.ask("Compare the 2023 lowest score for Peking University and Tsinghua University for science students in the first batch")
# vn.ask("Show me the enrollment plan for Computer Science at Fudan University in 2023")


training_data = vn.get_training_data()
training_data

# You can remove training data if there's obsolete/incorrect information. 
# vn.remove_training_data(id='1-ddl')

vn.ask(question="西安交通大学2024年理科录取分数?")
app = VannaFlaskApp(vn)
app.run()