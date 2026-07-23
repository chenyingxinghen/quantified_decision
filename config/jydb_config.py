"""聚源数据源配置。

连接信息只从环境变量读取，避免把数据库凭据写入仓库。
"""
import os

from .data_config import DATABASE_DIR, DATABASE_PATH, META_DB_PATH


JYDB_ENABLED = os.getenv("JYDB_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
JYDB_DRIVER = os.getenv("JYDB_DRIVER", "SQL Server")
JYDB_SERVER = os.getenv("JYDB_SERVER", "")
JYDB_DATABASE = os.getenv("JYDB_DATABASE", "JYDB")
JYDB_USERNAME = os.getenv("JYDB_USERNAME", "")
JYDB_PASSWORD = os.getenv("JYDB_PASSWORD", "")
JYDB_ADO_PROVIDER = os.getenv("JYDB_ADO_PROVIDER", "SQLOLEDB")

# 聚源数据清洗后的本地 PIT 特征库。训练、选股和回测只读取该库，
# 不直接依赖远端 SQL Server，以保证可复现性和运行稳定性。
# 特征库是 build 的「产物」，必须可写。云端按 raw-only 策略：raw 留在只读挂载
# (GEMINI_DATA_IN1)，特征/行情/元数据三个产物库写到可写输出 GEMINI_DATA_OUT。
# 因此这里优先 GEMINI_FEATURE_DB_PATH，其次 GEMINI_DATA_OUT/<文件名>，最后回退 DATABASE_DIR。
# 注意 GEMINI_DATA_OUT 是"输出目录"，需在其下拼接文件名（见 data_config._DATA_OUT 复用）。
JYDB_FEATURE_DB_PATH = os.getenv(
    "GEMINI_FEATURE_DB_PATH",
    os.path.join(os.getenv("GEMINI_DATA_OUT", DATABASE_DIR), "jydb_features.db"),
)
JYDB_RAW_DB_PATH = os.getenv(
    "JYDB_RAW_DB_PATH",
    os.path.join(DATABASE_DIR, "jydb_raw.db"),
)


def build_connection_string() -> str:
    """构造 pyodbc 连接串；缺少必要配置时给出明确错误。"""
    missing = [
        name
        for name, value in {
            "JYDB_SERVER": JYDB_SERVER,
            "JYDB_USERNAME": JYDB_USERNAME,
            "JYDB_PASSWORD": JYDB_PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"缺少聚源连接环境变量: {', '.join(missing)}")
    return (
        f"DRIVER={{{JYDB_DRIVER}}};SERVER={JYDB_SERVER};DATABASE={JYDB_DATABASE};"
        f"UID={JYDB_USERNAME};PWD={JYDB_PASSWORD};TrustServerCertificate=yes;"
    )


def build_ado_connection_string() -> str:
    """构造 adodbapi/OLE DB 连接串。"""
    # 复用相同的必填项校验，但不复用 ODBC 格式。
    missing = [
        name
        for name, value in {
            "JYDB_SERVER": JYDB_SERVER,
            "JYDB_USERNAME": JYDB_USERNAME,
            "JYDB_PASSWORD": JYDB_PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"缺少聚源连接环境变量: {', '.join(missing)}")
    return (
        f"Provider={JYDB_ADO_PROVIDER};Data Source={JYDB_SERVER};"
        f"Initial Catalog={JYDB_DATABASE};User ID={JYDB_USERNAME};Password={JYDB_PASSWORD};"
    )


__all__ = [
    "JYDB_ENABLED",
    "JYDB_FEATURE_DB_PATH",
    "JYDB_RAW_DB_PATH",
    "build_connection_string",
    "build_ado_connection_string",
]
