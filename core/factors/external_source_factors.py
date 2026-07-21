"""外部结构化数据源因子。"""
import os
from typing import Optional, Sequence

import pandas as pd

from config.jydb_config import JYDB_FEATURE_DB_PATH
from core.data.jydb_feature_store import JYDBFeatureStore


class ExternalSourceFactors:
    """把本地聚源 PIT 特征对齐到日行情。"""

    def __init__(self, db_path: str = JYDB_FEATURE_DB_PATH):
        self.db_path = db_path
        self.store = JYDBFeatureStore(db_path)

    @property
    def available(self) -> bool:
        return bool(self.db_path and os.path.exists(self.db_path))

    def calculate_series(
        self,
        code: str,
        daily_data: pd.DataFrame,
        feature_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if "date" not in daily_data.columns or not self.available:
            return pd.DataFrame(index=daily_data.index)
        daily = self.store.get_daily_series(
            code, daily_data["date"], feature_names=feature_names
        )
        pit = self.store.get_pit_series(
            code, daily_data["date"], feature_names=feature_names
        )
        result = daily.combine_first(pit)
        result.index = daily_data.index
        return result

    def get_feature_names(self):
        return self.store.get_feature_names() if self.available else []
