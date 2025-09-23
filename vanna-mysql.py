import logging
import os
from datetime import datetime
from flask import request
import json
from openai import OpenAI
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat
from vanna.qianwen import QianWenAI_Chat

from vanna.flask import VannaFlaskApp
import yaml
from auth import SimplePassword

logging.basicConfig(level=logging.INFO)  


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

vector_store_config = config.get('vector_store', {})
mode = config.get('mode', 'openai-compatible')

if mode == 'qwen':
    llm_config = config.get('qwen', {})
    if not llm_config:
        raise ValueError('模式设置为 qwen，但未找到 qwen 配置')
else:
    llm_config = config.get('llm')
    if llm_config is None:
        llm_config = config.get('openai', {})
        if llm_config:
            logging.info('检测到 openai 配置，默认映射到本地/兼容 LLM 配置')
    if not llm_config:
        raise ValueError('未找到 llm/openai 配置，请检查 config.yaml')

myvanna_cfg = {'vector_store': vector_store_config}
if mode == 'qwen':
    myvanna_cfg['qwen'] = llm_config
else:
    myvanna_cfg['llm'] = llm_config

class MyVanna(ChromaDB_VectorStore, (QianWenAI_Chat if mode == 'qwen' else OpenAI_Chat)):
    def __init__(self, config=None):
        if config is None:
            config = {}

        vector_cfg = dict(config.get('vector_store', {}) or {})
        backend_cfg = dict(config.get('llm', {})) if mode != 'qwen' else dict(config.get('qwen', {}))

        if 'path' not in vector_cfg:
            vector_cfg['path'] = '.'

        if mode != 'qwen' and not backend_cfg.get('model'):
            raise ValueError('未配置 llm.model，无法确定使用的本地模型名称')

        # 初始化向量检索
        ChromaDB_VectorStore.__init__(self, config=vector_cfg)

        if mode == 'qwen':
            QianWenAI_Chat.__init__(self, config=backend_cfg)
        else:
            client_kwargs = {}
            api_key = backend_cfg.get('api_key') or os.getenv('OPENAI_API_KEY') or 'EMPTY'
            client_kwargs['api_key'] = api_key

            base_url = backend_cfg.get('base_url') or os.getenv('OPENAI_BASE_URL')
            if base_url:
                client_kwargs['base_url'] = base_url

            headers = backend_cfg.get('headers')
            if headers:
                client_kwargs['default_headers'] = headers

            client = OpenAI(**client_kwargs)

            chat_cfg = {
                'model': backend_cfg.get('model'),
                'temperature': backend_cfg.get('temperature', 0.2),
            }
            if backend_cfg.get('max_tokens') is not None:
                chat_cfg['max_tokens'] = backend_cfg['max_tokens']

            OpenAI_Chat.__init__(self, client=client, config=chat_cfg)

    def add_documentation_to_prompt(self, initial_prompt, documentation_list, max_tokens=14000):
        """在拼装提示词前过滤掉空文档或异常对象"""
        cleaned_docs = []
        for doc in documentation_list or []:
            if isinstance(doc, str) and doc.strip():
                cleaned_docs.append(doc)
            else:
                logging.warning("忽略无效文档片段：%s", type(doc).__name__)

        if not cleaned_docs:
            return initial_prompt

        return super().add_documentation_to_prompt(initial_prompt, cleaned_docs, max_tokens=max_tokens)

    def str_to_approx_token_count(self, string):
        """兼容 None 或非字符串输入，避免长度计算抛出异常"""
        if string is None:
            logging.warning("检测到 None 文档片段，按 0 token 处理")
            return 0

        if not isinstance(string, str):
            logging.warning("文档类型非字符串(%s)，转换后计数", type(string).__name__)
            string = str(string)

        return len(string) / 4

vn = MyVanna(config=myvanna_cfg)

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

# vn.remove_collection("sql")  

# vn.remove_collection("documentation")

# 设置环境变量避免tokenizers警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 优化ChromaDB性能
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

print("开始训练数据库schema...")
try:
    # The information schema query may need some tweaking depending on your database. This is a good starting point.
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS where TABLE_SCHEMA='test_env'")
    print(f"获取到 {len(df_information_schema)} 个字段信息")

    # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
    plan = vn.get_training_plan_generic(df_information_schema)
    print(f"生成训练计划，包含 {len(plan)} 个训练项")

    # If you like the plan, then uncomment this and run it to train
    print("开始训练数据库schema...")
    vn.train(plan=plan)
    print("数据库schema训练完成")
except Exception as e:
    print(f"数据库schema训练失败: {e}")
    logging.error(f"数据库schema训练失败: {e}")
# The following are methods for adding training data. Make sure you modify the examples to match your database.



# --- Add Documentation ---
# Documentation helps Vanna understand business logic, column definitions, and terminology.

print("Training documentation...")
try:
    vn.train(documentation="""
这是一个公安人员管理信息系统，包含以下核心业务模块：

1. 重点人员管理（t_fl_zdry表）：
   - 人员基本信息：身份证号、姓名、民族、籍贯、国籍、性别、出生日期等
   - 人员标签分类：通过rybq字段进行人员分类管理
   - 人员级别：ryjb字段标识人员重要性级别
   - 管控状态：各种管控相关信息

2. 人员管控信息（t_fl_zdry_gkxx表）：
   - 管控单位信息：管控单位名称、代码
   - 管控级别：fxdj（风险等级）、gkjb（管控级别）
   - 责任人员：zrld（责任领导）、zrmj（责任民警）
   - 联系方式：责任人和民警的联系电话

3. 人员相关指令信息（t_fl_zlxx相关表）：
   - 指令信息主表：t_fl_zlxx（指令类型、申请单位、接收时间等）
   - 指令反馈：t_fl_zlxx_feedback_ssry（实上人员反馈信息）
   - 云控指令反馈：t_fl_zlxx_feedback_byk（布控指令反馈）

4. 人员详细信息扩展表：
   - 车辆信息：t_fl_zdry_clxx（车辆基本信息）
   - 通讯信息：t_fl_zdry_txxx（手机号码、MAC地址、IMEI等）
   - 虚拟身份：t_fl_zdry_xnsf（微信、QQ等网络身份）
   - 社会关系：t_fl_zdry_shgx（家庭关系、社会关系）
   - 民警走访记录：t_fl_zdry_mjzfjl（民警走访情况）

5. 字典管理：
   - 人员标签字典：t_fl_rybq（人员分类标签）
   - 要素字典：t_fl_label（各类要素字典，如民族等）

常用字段说明：
- rybh：人员编号
- sfzh：身份证号
- xm：姓名
- glbh：关联编号（通常为身份证号）
- gkdwmc：管控单位名称
- zrld：责任领导
- zrmj：责任民警
- fknr：反馈内容
- fksj：反馈时间
- zl_type：指令类型
""")
    print("业务文档训练完成")
except Exception as e:
    print(f"业务文档训练失败: {e}")
    logging.error(f"业务文档训练失败: {e}")

# 添加数据库表结构DDL训练
print("Training DDL documentation...")
try:
    vn.train(documentation="""
-- 核心业务表结构定义

-- 1. 重点人员基本信息表 (t_fl_zdry)
-- 这是重点人员管理的主表，存储人员的基本信息、标签、风险等级等
DROP TABLE IF EXISTS `t_fl_zdry`;
CREATE TABLE `t_fl_zdry` (
  `rybh` varchar(100) NOT NULL COMMENT '人员编号，主键',
  `sfzh` varchar(255) DEFAULT '' COMMENT '身份证号码',
  `xm` varchar(255) DEFAULT '' COMMENT '姓名',
  `mz` varchar(255) DEFAULT '' COMMENT '民族',
  `jg` varchar(255) DEFAULT '' COMMENT '籍贯',
  `gj` varchar(255) DEFAULT '' COMMENT '国籍',
  `xb` varchar(255) DEFAULT '' COMMENT '性别',
  `csrq` varchar(255) DEFAULT NULL COMMENT '出生日期',
  `zw` varchar(255) DEFAULT '' COMMENT '职务',
  `zzmm` varchar(255) DEFAULT '' COMMENT '政治面貌',
  `hyzt` varchar(255) DEFAULT '' COMMENT '婚姻状况',
  `whcd` varchar(255) DEFAULT '' COMMENT '文化程度',
  `hjdz` varchar(255) DEFAULT '' COMMENT '户籍地址',
  `xzdz` varchar(255) DEFAULT '' COMMENT '现住地址',
  `rybq` varchar(500) DEFAULT '' COMMENT '人员标签，多个标签用逗号分隔',
  `jzbq` varchar(255) DEFAULT '' COMMENT '警种标签',
  `hjdpcs` varchar(255) DEFAULT '' COMMENT '户籍地派出所',
  `yxx` varchar(255) DEFAULT '0' COMMENT '有效性（0有效，1无效）',
  `ryjb` varchar(255) DEFAULT '' COMMENT '人员级别',
  `ssd` varchar(255) DEFAULT '' COMMENT '涉事地',
  `bjxwbq` varchar(255) DEFAULT '' COMMENT '背景和行为特征标签',
  `wffzjl` varchar(255) DEFAULT '' COMMENT '违法犯罪记录',
  `zysq` longtext COMMENT '主要诉求',
  `fxpg` varchar(255) DEFAULT '' COMMENT '风险评估',
  `fxqk` varchar(255) DEFAULT '' COMMENT '风险情况',
  PRIMARY KEY (`rybh`)
) COMMENT='风铃新风险人员库';

-- 2. 人员管控信息表 (t_fl_zdry_gkxx)
-- 记录重点人员的管控单位、责任人、风险等级等信息
DROP TABLE IF EXISTS `t_fl_zdry_gkxx`;
CREATE TABLE `t_fl_zdry_gkxx` (
  `id` int(13) NOT NULL AUTO_INCREMENT,
  `glbh` varchar(255) NOT NULL COMMENT '关联重点人员编号(身份证号)',
  `gkdwmc` varchar(255) DEFAULT '' COMMENT '管控单位名称',
  `gkdwdm` varchar(255) DEFAULT '' COMMENT '管控单位代码',
  `lgsj` varchar(255) DEFAULT '' COMMENT '列管时间',
  `sfcg` varchar(255) DEFAULT '' COMMENT '是否撤管',
  `cgly` varchar(255) DEFAULT '' COMMENT '撤管理由',
  `zrld` varchar(255) DEFAULT '' COMMENT '责任领导',
  `zrldlxdh` varchar(255) DEFAULT '' COMMENT '责任领导联系电话',
  `zrmj` varchar(255) DEFAULT '' COMMENT '责任民警',
  `zrmjlxdh` varchar(255) DEFAULT '' COMMENT '责任民警联系电话',
  `fxdj` varchar(255) DEFAULT '' COMMENT '风险等级',
  `gkjb` varchar(255) DEFAULT '' COMMENT '管控级别',
  `lgyj` varchar(255) DEFAULT '' COMMENT '列管依据',
  PRIMARY KEY (`id`),
  UNIQUE KEY `glbh` (`glbh`)
) COMMENT='风铃风险人员-管控信息表';

-- 3. 人员通讯信息表 (t_fl_zdry_txxx)
-- 存储重点人员的通讯设备信息：手机、MAC地址、IMSI、IMEI等
DROP TABLE IF EXISTS `t_fl_zdry_txxx`;
CREATE TABLE `t_fl_zdry_txxx` (
  `id` int(13) NOT NULL AUTO_INCREMENT,
  `glbh` varchar(100) NOT NULL COMMENT '关联重点人员编号',
  `sjhm` varchar(255) DEFAULT NULL COMMENT '手机号码',
  `mac_address` varchar(255) DEFAULT NULL COMMENT 'MAC地址',
  `imsi` varchar(255) DEFAULT NULL COMMENT 'IMSI',
  `imei` varchar(255) DEFAULT NULL COMMENT 'IMEI',
  `wifi_name` varchar(255) DEFAULT NULL COMMENT 'WIFI名称',
  `wifi_address` varchar(255) DEFAULT NULL COMMENT 'WIFI位置',
  PRIMARY KEY (`id`)
) COMMENT='风铃风险人员-通讯信息';

-- 4. 人员车辆信息表 (t_fl_zdry_clxx)
-- 记录重点人员相关的车辆信息
DROP TABLE IF EXISTS `t_fl_zdry_clxx`;
CREATE TABLE `t_fl_zdry_clxx` (
  `id` int(13) NOT NULL AUTO_INCREMENT,
  `glbh` varchar(30) NOT NULL COMMENT '关联重点人员编号',
  `czgx` varchar(255) DEFAULT NULL COMMENT '车主关系',
  `cllx` varchar(255) DEFAULT NULL COMMENT '车辆类型',
  `clzl` varchar(255) DEFAULT NULL COMMENT '车辆种类',
  `cphm` varchar(255) DEFAULT NULL COMMENT '车牌号码',
  `clpp` varchar(255) DEFAULT NULL COMMENT '车辆品牌',
  `clxh` varchar(255) DEFAULT NULL COMMENT '车辆型号',
  `syrxm` varchar(255) DEFAULT NULL COMMENT '所有人姓名',
  `syrsfzh` varchar(255) DEFAULT NULL COMMENT '所有人身份证号',
  PRIMARY KEY (`id`)
) COMMENT='风铃风险人员-车辆信息表';

-- 5. 人员虚拟身份表 (t_fl_zdry_xnsf)
-- 存储重点人员的网络虚拟身份信息
DROP TABLE IF EXISTS `t_fl_zdry_xnsf`;
CREATE TABLE `t_fl_zdry_xnsf` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `glbh` varchar(100) DEFAULT NULL COMMENT '关联编号',
  `wxh` varchar(255) DEFAULT NULL COMMENT '微信号',
  `qqh` varchar(255) DEFAULT NULL COMMENT 'qq号',
  `wl_id` varchar(255) DEFAULT NULL COMMENT '网络id',
  `wlqz` varchar(255) DEFAULT NULL COMMENT '网络群组',
  `qt` varchar(255) DEFAULT NULL COMMENT '其它',
  PRIMARY KEY (`id`)
) COMMENT='虚拟身份表';

-- 6. 人员社会关系表 (t_fl_zdry_shgx)
-- 记录重点人员的社会关系网络
DROP TABLE IF EXISTS `t_fl_zdry_shgx`;
CREATE TABLE `t_fl_zdry_shgx` (
  `id` int(12) NOT NULL AUTO_INCREMENT,
  `glbh` varchar(100) NOT NULL COMMENT '关联重点人员编号',
  `gxlb` varchar(255) DEFAULT NULL COMMENT '关系类别',
  `xm` varchar(255) DEFAULT NULL COMMENT '姓名',
  `sfzh` varchar(255) DEFAULT NULL COMMENT '身份证号',
  `gzdw` varchar(255) DEFAULT NULL COMMENT '工作单位',
  `zw` varchar(255) DEFAULT NULL COMMENT '职务',
  `lxfs` varchar(255) DEFAULT NULL COMMENT '联系方式',
  PRIMARY KEY (`id`)
) COMMENT='风铃风险人员-社会关系表';

-- 7. 人员标签字典表 (t_fl_rybq)
-- 人员分类标签的字典表，支持层级结构
DROP TABLE IF EXISTS `t_fl_rybq`;
CREATE TABLE `t_fl_rybq` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '标签ID',
  `parent_id` bigint(20) DEFAULT NULL COMMENT '父ID',
  `ancestors` varchar(1000) DEFAULT '' COMMENT '祖级列表',
  `code` varchar(100) DEFAULT NULL COMMENT '标签代码',
  `name` varchar(100) DEFAULT NULL COMMENT '标签名称',
  `alias` varchar(100) DEFAULT NULL COMMENT '标签别名',
  `describe` varchar(2000) DEFAULT NULL COMMENT '标签说明',
  `order_index` int(11) DEFAULT NULL COMMENT '标签排序',
  `status` char(1) DEFAULT '0' COMMENT '标签状态（0正常 1停用）',
  `flow_status` char(1) DEFAULT '1' COMMENT '流程状态（0待审批 1通过 2驳回）',
  PRIMARY KEY (`id`)
) COMMENT='人员标签';

-- 8. 要素标签表 (t_fl_label)
-- 各种要素的标签字典，如民族、学历等
DROP TABLE IF EXISTS `t_fl_label`;
CREATE TABLE `t_fl_label` (
  `label_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '标签id',
  `parent_id` bigint(20) DEFAULT '0' COMMENT '父标签id',
  `ancestors` varchar(1000) DEFAULT '' COMMENT '祖级列表',
  `label_number` varchar(100) DEFAULT NULL COMMENT '标签编号',
  `label_name` varchar(255) DEFAULT '' COMMENT '标签名称',
  `alias_name` varchar(100) DEFAULT NULL COMMENT '标签别名',
  `label_describe` varchar(255) DEFAULT NULL COMMENT '标签描述',
  `order_num` int(4) DEFAULT '0' COMMENT '显示顺序',
  `label_category` varchar(255) DEFAULT NULL COMMENT '标签类别',
  `status` char(1) DEFAULT '0' COMMENT '标签状态（0正常 1停用）',
  PRIMARY KEY (`label_id`)
) COMMENT='标签表';

-- 9. 指令信息表 (t_fl_zlxx)
-- 指令管理的主表，存储指令的基本信息和状态
DROP TABLE IF EXISTS `t_fl_zlxx`;
CREATE TABLE `t_fl_zlxx` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `jbzlxxbh` varchar(100) DEFAULT NULL COMMENT '指令信息编号',
  `jbzlbh` varchar(100) DEFAULT NULL COMMENT '指令编号',
  `jbzlbt` varchar(255) DEFAULT NULL COMMENT '指令标题',
  `jbzlnr` longtext COMMENT '指令内容',
  `jbzllx` varchar(50) DEFAULT NULL COMMENT '指令类型',
  `jbzljjcd` char(1) DEFAULT NULL COMMENT '紧急程度（1红色2橙色3蓝色）',
  `jlxzsj` datetime DEFAULT NULL COMMENT '下发时间',
  `sqdw` varchar(255) DEFAULT NULL COMMENT '申请单位',
  `sqdwdm` bigint(20) DEFAULT NULL COMMENT '申请单位代码',
  `sjly` char(1) DEFAULT NULL COMMENT '数据来源（1部云控2省厅指令5省厅转阅3情报线索4铁路布控9其他）',
  `zlzt` char(1) DEFAULT NULL COMMENT '状态（1工作中 2待反馈 3待市局核录 4已反馈）',
  `zl_type` char(1) DEFAULT NULL COMMENT '1向上2向下',
  `del_flag` char(1) DEFAULT '0' COMMENT '删除标志（0代表存在 1代表删除）',
  PRIMARY KEY (`id`)
) COMMENT='风铃指令信息';

-- 10. 指令反馈表 (t_fl_zlxx_feedback_ssry)
-- 存储指令执行后的反馈信息
DROP TABLE IF EXISTS `t_fl_zlxx_feedback_ssry`;
CREATE TABLE `t_fl_zlxx_feedback_ssry` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `zlxx_id` bigint(20) DEFAULT NULL COMMENT '指令ID',
  `ssry_id` bigint(20) DEFAULT NULL COMMENT '涉事人员ID',
  `xm` varchar(255) DEFAULT NULL COMMENT '姓名',
  `gmsfhm` varchar(255) DEFAULT NULL COMMENT '身份证',
  `wkzt` varchar(50) DEFAULT NULL COMMENT '稳控状态',
  `fknr` varchar(4000) DEFAULT NULL COMMENT '反馈内容',
  `ry_lxdh` varchar(50) DEFAULT NULL COMMENT '人员联系电话',
  `hcmjxm` varchar(50) DEFAULT NULL COMMENT '核查民警姓名',
  `hcmjdh` varchar(50) DEFAULT NULL COMMENT '核查民警电话',
  `rybq` varchar(255) DEFAULT NULL COMMENT '人员标签',
  `fkr` varchar(50) DEFAULT NULL COMMENT '反馈人',
  `fkdw` varchar(255) DEFAULT NULL COMMENT '反馈单位',
  `fksj` datetime DEFAULT NULL COMMENT '反馈时间',
  PRIMARY KEY (`id`)
) COMMENT='指令信息涉事人员反馈表';

-- 11. 系统部门表 (sys_dept)
-- 组织架构管理的部门表
DROP TABLE IF EXISTS `sys_dept`;
CREATE TABLE `sys_dept` (
  `dept_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '部门id',
  `parent_id` bigint(20) DEFAULT '0' COMMENT '父部门id',
  `ancestors` varchar(50) DEFAULT '' COMMENT '祖级列表',
  `dept_name` varchar(30) DEFAULT '' COMMENT '部门名称',
  `order_num` int(4) DEFAULT '0' COMMENT '显示顺序',
  `leader` varchar(20) DEFAULT NULL COMMENT '负责人',
  `phone` varchar(11) DEFAULT NULL COMMENT '联系电话',
  `status` char(1) DEFAULT '0' COMMENT '部门状态（0正常 1停用）',
  `del_flag` char(1) DEFAULT '0' COMMENT '删除标志（0代表存在 2代表删除）',
  `dept_type` char(1) DEFAULT NULL COMMENT '部门类型 1:市局 2:分局 3:派出所 4:警种部门',
  PRIMARY KEY (`dept_id`)
) COMMENT='部门表';

-- 主要字段说明和业务逻辑：
-- rybh: 人员编号，作为人员信息的主键
-- sfzh: 身份证号，常用的人员唯一标识
-- glbh: 关联编号，通常关联到身份证号
-- rybq: 人员标签，多个标签可以用逗号分隔
-- fxdj: 风险等级，评估人员的风险程度
-- gkjb: 管控级别，确定管控的强度
-- zrld/zrmj: 责任领导和责任民警，负责管控工作
-- zlxx相关表: 管理指令的下发、执行、反馈流程
""")
    print("DDL文档训练完成")
except Exception as e:
    print(f"DDL文档训练失败: {e}")
    logging.error(f"DDL文档训练失败: {e}")




# --- Add Question-SQL Pairs ---
# These pairs teach Vanna how to translate natural language questions into specific SQL queries.
# Include variations of questions that map to the same SQL.

print("Training Question-SQL pairs...")
try:
    # 人员基本信息查询
    vn.train(
    question="查询身份证号为320302199909090009的人员基本信息",
    sql="""
SELECT rybh,
       sfzh,
       xm,
       mz,
       jg,
       gj,
       xb,
       csrq,
       zw,
       zzmm,
       hyzt,
       whcd,
       fwcs,
       hjdz,
       hjdzsf,
       hjdzds,
       hjdzqx,
       xzdz,
       xzdzsf,
       xzdzds,
       xzdzqx,
       rybq,
       rybq_old,
       rybq_old_0331,
       jzbq,
       hjdpcs,
       yxx,
       ryjb,
       ssd,
       ssdsf,
       ssdds,
       ssdqx,
       bjxwbq,
       wffzjl,
       zysq,
       sffqzj,
       dqhjczqk,
       fxpg,
       fxqk,
       pic,
       djr,
       djdwmc,
       djdwdm,
       djsj,
       xgr,
       xgdwmc,
       xgdwdm,
       xgsj,
       cxr,
       cxdwmc,
       cxdw,
       cxsj,
       cxyy,
       shsf
FROM t_fl_zdry
WHERE (sfzh = '320302199909090009');
"""
)

    # 人员管控信息查询
    vn.train(
        question="查询身份证号为320302199909090009的人员管控信息",
        sql="""
    SELECT id,
           glbh,
           gkdwmc,
           gkdwdm,
           lgsj,
           sfcg,
           cgly,
           zrld,
           zrldlxdh,
           zrmj,
           zrmjlxdh,
           fxdj,
           gkjb,
           lgyj,
           djr,
           djdwmc,
           djdwdm,
           djsj,
           xgr,
           xgdwmc,
           xgdwdm,
           xgsj,
           zrmjjh
    FROM t_fl_zdry_gkxx
    WHERE (glbh = '320302199909090009');
    """
    )
    
    # 人员车辆信息查询
    vn.train(
        question="查询身份证号为320302199909090009的人员车辆信息",
        sql="""
    SELECT id,
           glbh,
           czgx,
           cllx,
           clzl,
           cphm,
           clpp,
           clxh,
           syrxm,
           syrsfzh,
           csys,
           cpys,
           djr,
           djdwmc,
           djdwdm,
           djsj,
           xgr,
           xgdwmc,
           xgdwdm,
           xgsj
    FROM t_fl_zdry_clxx
    WHERE (glbh = '320302199909090009');
    """
    )
    
    # 人员通讯信息查询
    vn.train(
        question="查询身份证号为320302199909090009的人员通讯信息",
        sql="""
    SELECT id,
           glbh,
           sjhm,
           mac_address,
           imsi,
           imei,
           wifi_name,
           wifi_address,
           djr,
           djdwmc,
           djdwdm,
           djsj,
           xgr,
           xgdwmc,
           xgdwdm,
           xgsj
    FROM t_fl_zdry_txxx
    WHERE (glbh = '320302199909090009');
    """
    )
    
    # 人员虚拟身份查询
    vn.train(
        question="查询身份证号为320302199909090009的人员虚拟身份信息",
        sql="""
    SELECT id,
           glbh,
           wxh,
           qqh,
           wl_id,
           wlqz,
           qt,
           djr,
           djdwmc,
           djdwdm,
           djsj,
           xgr,
           xgdwmc,
           xgdwdm,
           xgsj
    FROM t_fl_zdry_xnsf
    WHERE (glbh = '320302199909090009');
    """
    )
    
    # 人员社会关系查询
    # vn.train(
    #     question="查询身份证号为320302199909090009的人员社会关系",
    #     sql="""
    # SELECT id,
    #        glbh,
    #        gxlb,
    #        xm,
    #        sfzh,
    #        gzdw,
    #        zw,
    #        lxfs,
    #        cl,
    #        djr,
    #        djdwmc,
    #        djdwdm,
    #        djsj,
    #        xgr,
    #        xgdwmc,
    #        xgdwdm,
    #        xgsj
    # FROM t_fl_zdry_shgx
    # WHERE (glbh = '320302199909090009');
    # """
    # )
    
    # 民警走访记录查询
    vn.train(
        question="查询身份证号为320302199909090009的人员民警走访记录",
        sql="""
    SELECT id,
           glbh,
           zflx,
           sfzk,
           zfsj,
           zfdz,
           zfnr,
           djr,
           djdwmc,
           djdwdm,
           djsj,
           xgr,
           xgdwmc,
           xgdwdm,
           xgsj
    FROM t_fl_zdry_mjzfjl
    WHERE (glbh = '320302199909090009');
    """
    )
    
    # 指令信息查询
    vn.train(
        question="查询身份证号为320302199909090009的基础信息",
        sql="""
    SELECT dp.dept_name as fkfjmc,
           dp.dept_id   as fkfjdm,
           d.dept_name  as fkpcsmc,
           d.dept_id    as fkpcsdm,
           s.rybh,
           fs.id        as feedbackId,
           fs.xm,
           fs.gmsfhm,
           fs.wkzt,
           fs.fknr,
           fs.fkdw,
           fs.fksj,
           fs.ry_lxdh   as lxdh,
           fs.hcmjxm    as hcmj,
           fs.hcmjdh,
           fs.ywfx,
           fs.fkr,
           fs.rybq      as fkRybq,
           fs.fsjjmd,
           s.cc,
           s.sfz,
           s.ddz,
           s.ccrq,
           z.sqdw,
           z.sjly,
           z.jbzlbh,
           z.jbzljjcd,
           z.jlxzsj,
           r.rybq,
           r.ryjb,
           z.zl_type    as zlType,
           z.jbzlbt,
           z.fl_fksx    as flFksx,
           z.db_type    as dbType
    FROM t_fl_zlxx_feedback_ssry fs
             LEFT JOIN t_fl_zlxx_ssry s ON s.id = fs.ssry_id
             LEFT JOIN t_fl_zlxx z ON z.id = fs.zlxx_id
             LEFT JOIN t_fl_zdry r ON r.sfzh = fs.gmsfhm
             LEFT JOIN sys_dept d ON d.dept_id = s.rldwdm
             LEFT JOIN sys_dept dp ON dp.dept_id = d.parent_id
    WHERE z.del_flag = '0'
      and fs.id IN (SELECT max(id) FROM t_fl_zlxx_feedback_ssry f2 WHERE fs.zlxx_id = f2.zlxx_id GROUP BY f2.ssry_id)
      AND fs.gmsfhm = '320302199909090009'
    ORDER BY fs.fksj DESC;
    """
    )
    
    # 云控指令反馈查询
    # vn.train(
    #     question="查询指令反馈云控指令反馈信息",
    #     sql="""
    # select id,
    #        feedback_id,
    #        sfbr,
    #        mbfxzt,
    #        czjg,
    #        czcs,
    #        czms,
    #        fxzrdw,
    #        fxzrdwdm,
    #        fxzrmj,
    #        fxzrmjsfz,
    #        czzrdw,
    #        czzrdwdm,
    #        czzrmj,
    #        czzrmjsfz,
    #        czsj,
    #        czddqh,
    #        czddxz,
    #        remark
    # from t_fl_zlxx_feedback_byk
    # """
    # )
    
    # 人员标签字典查询
    # vn.train(
    #     question="查询人员标签代码为00010001001600010001的人员标签字典信息",
    #     sql="""
    # SELECT id,
    #        parent_id,
    #        ancestors,
    #        area_code,
    #        area_name,
    #        code,
    #        `name`,
    #        `alias`,
    #        `describe`,
    #        order_index,
    #        remark,
    #        `status`,
    #        flow_status,
    #        del_flag,
    #        create_by,
    #        create_time,
    #        update_by,
    #        update_time,
    #        audit_by,
    #        audit_time,
    #        delete_by,
    #        delete_time
    # FROM t_fl_rybq
    # WHERE (code IN ('00010001001600010001'));
    # """
    # )
    
    # 民族要素字典查询
    # vn.train(
    #     question="查询民族要素字典信息",
    #     sql="""
    # select label_id,
    #        parent_id,
    #        ancestors,
    #        label_name,
    #        alias_name,
    #        label_number,
    #        label_describe,
    #        order_num,
    #        label_category,
    #        leader,
    #        phone,
    #        email,
    #        status,
    #        flow_status,
    #        del_flag,
    #        create_by,
    #        create_time,
    #        update_by,
    #        update_time,
    #        remark
    # from t_fl_label
    # WHERE label_category = '民族'
    #   and status = '0'
    #   and del_flag = '0'
    #   and flow_status = '1'
    # order by order_num;
    # """
    # )
    
    
    print("Training Question-SQL pairs完成")
except Exception as e:
    print(f"Question-SQL Pairs训练失败: {e}")
    logging.error(f"Question-SQL Pairs训练失败: {e}")

print("所有训练完成！AI系统已准备就绪。")

# After running these, you can ask Vanna questions like:
# vn.ask("What is the website for Tsinghua University?")
# vn.ask("List 985 schools in Beijing")
# vn.ask("Compare the 2023 lowest score for Peking University and Tsinghua University for science students in the first batch")
# vn.ask("Show me the enrollment plan for Computer Science at Fudan University in 2023")


training_data = vn.get_training_data()
training_data

# You can remove training data if there's obsolete/incorrect information. 
# vn.remove_training_data(id='1-ddl')
  
# 设置基本日志配置  
logging.basicConfig(  
    filename='vanna_app.log',  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
  
# 创建logs目录  
os.makedirs('logs', exist_ok=True)  
  
class LoggingVannaFlaskApp(VannaFlaskApp):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
          
        # 添加请求拦截器  
        @self.flask_app.before_request  
        def log_request():  
            if request.path == '/api/v0/generate_sql':  
                question = request.args.get('question')  
                if question:  
                    log_message = f"[{datetime.now()}] Question: {question}"  
                      
                    # 控制台日志  
                    print(log_message)  
                      
                    # 文件日志  
                    with open('logs/vanna_queries.log', 'a') as f:  
                        f.write(log_message + '\n')  
                      
                    # Python日志  
                    logging.info(f"Question: {question}")  
          
        # 添加响应拦截器  
        @self.flask_app.after_request  
        def log_response(response):  
            if request.path == '/api/v0/generate_sql':  
                try:  
                    # 克隆响应以避免消耗它  
                    response_clone = response.get_data(as_text=True)  
                    response_data = json.loads(response_clone)  
                      
                    if response_data and response_data.get('type') == 'sql':  
                        sql = response_data.get('text')  
                        log_message = f"[{datetime.now()}] SQL: {sql}"  
                          
                        # 控制台日志  
                        print(log_message)  
                          
                        # 文件日志  
                        with open('logs/vanna_queries.log', 'a') as f:  
                            f.write(log_message + '\n')  
                          
                        # Python日志  
                        logging.info(f"SQL: {sql}")  
                except Exception as e:  
                    print(f"Error logging response: {e}")  
                    logging.error(f"Error logging response: {e}")  
              
            return response  
  
# 使用修改后的类  
app = LoggingVannaFlaskApp(vn,  
                chart=False,  
                title="欢迎使用汇享易问数智能体",  
                subtitle="Your AI-powered copilot for SQL", 
                logo="https://pub-10375556b89a45e0a56aff68854a2214.r2.dev/%E9%97%AE%E6%95%B0%E6%99%BA%E8%83%BD%E4%BD%93.jpg", 
                summarization=False,  
                ask_results_correct=True,  
                debug=True,  # 这个设置Vanna的debug，不是Flask的  
                sql=True,  
                suggested_questions=True,  
                show_training_data=True,  
                function_generation=True  
                )  
  
# 显式设置Flask的debug模式  
app.run(host="0.0.0.0", port=8084, debug=True)
