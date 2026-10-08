"""复权因子的独立月度标记回归测试。不访问行情 API。

背景：`update_adjust_factor_data` 原先挂在 `last_finance_sync` 上节制，但那个
stamp 的语义是「**财务数据已入库**」，只在真写进行时才落。2026-10 三季报披露前，
财务接口对全市场 5000+ 只**天天返回空**，于是整月都不 stamp，复权因子陪跑每晚
一次全历史重抓（~5600 次调用/夜），要一直烧到三季报出来。

改为 `low_freq_sync` 表里的独立标记，语义是「本月已检查过」——与这次是否拿到
数据无关，但**请求失败时不能打标**，否则一次网络抖动会让这只股票整月不再刷新。
"""
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.data.baostock_fetcher_methods import QuotaExceededError
from core.data.baostock_main import ADJUST_FACTOR_KIND, BaostockDataManager


def _bare_manager():
    """绕开 __init__（它会连生产库 / 建表），只测方法逻辑。"""
    mgr = object.__new__(BaostockDataManager)
    mgr._bs_login = lambda: True
    return mgr


class MarkerStorageTests(unittest.TestCase):
    """low_freq_sync 的读写与 upsert。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / 'meta.db'
        conn = sqlite3.connect(str(path))
        conn.execute('CREATE TABLE low_freq_sync (code TEXT NOT NULL, kind TEXT NOT NULL,'
                     ' period TEXT NOT NULL, updated_at TEXT, PRIMARY KEY (code, kind))')
        conn.commit()
        conn.execute('ATTACH DATABASE ? AS meta', (str(path),))
        self.conn = conn
        # addCleanup 是 LIFO：这个必须先于 temp.cleanup 跑，否则 Windows 上
        # 删不掉还被 sqlite 句柄占着的文件，失败会伪装成用例失败。
        self.addCleanup(conn.close)
        self.mgr = _bare_manager()
        self.mgr._conn = conn

    def test_missing_marker_reads_as_none(self):
        self.assertIsNone(self.mgr.get_low_freq_marker('sh.600000', ADJUST_FACTOR_KIND))

    def test_round_trip_and_upsert(self):
        self.mgr.set_low_freq_marker('sh.600000', ADJUST_FACTOR_KIND, '2026-10')
        self.assertEqual(self.mgr.get_low_freq_marker('sh.600000', ADJUST_FACTOR_KIND), '2026-10')
        # 同月重打标是 upsert，不是插重复行
        self.mgr.set_low_freq_marker('sh.600000', ADJUST_FACTOR_KIND, '2026-11')
        self.assertEqual(self.mgr.get_low_freq_marker('sh.600000', ADJUST_FACTOR_KIND), '2026-11')
        n = self.conn.execute('SELECT COUNT(*) FROM meta.low_freq_sync').fetchone()[0]
        self.assertEqual(n, 1)

    def test_kinds_are_independent(self):
        self.mgr.set_low_freq_marker('sh.600000', ADJUST_FACTOR_KIND, '2026-10')
        self.assertIsNone(self.mgr.get_low_freq_marker('sh.600000', 'other_kind'))


class AdjustFactorGatingTests(unittest.TestCase):
    """标记命中就跳过；没命中才请求；请求结果决定要不要打标。"""

    def setUp(self):
        self.mgr = _bare_manager()
        self.this_month = datetime.now().strftime('%Y-%m')

    def test_skips_when_already_marked_this_month(self):
        self.mgr.get_low_freq_marker = lambda code, kind: self.this_month
        stamp = []
        self.mgr.set_low_freq_marker = lambda c, k, p: stamp.append((c, k, p))
        with patch('core.data.baostock_main.fetch_adjust_factor') as fetch:
            self.mgr.update_adjust_factor_data('sh.600000')
        fetch.assert_not_called()
        self.assertEqual(stamp, [])

    def test_requests_when_marker_is_a_previous_month(self):
        self.mgr.get_low_freq_marker = lambda code, kind: '2026-09'
        self.mgr.set_low_freq_marker = lambda c, k, p: None
        with patch('core.data.baostock_main.fetch_adjust_factor',
                   return_value=pd.DataFrame()) as fetch:
            self.mgr.update_adjust_factor_data('sh.600000')
        fetch.assert_called_once()

    def test_stamps_even_when_result_is_empty(self):
        """空返回 = 已退市等结构性缺失，本月不会变 —— 也要打标，否则每晚重抓。"""
        self.mgr.get_low_freq_marker = lambda code, kind: None
        stamp = []
        self.mgr.set_low_freq_marker = lambda c, k, p: stamp.append((c, k, p))
        with patch('core.data.baostock_main.fetch_adjust_factor',
                   return_value=pd.DataFrame()):
            self.mgr.update_adjust_factor_data('sh.600000')
        self.assertEqual(stamp, [('sh.600000', ADJUST_FACTOR_KIND, self.this_month)])

    def test_does_not_stamp_when_request_fails(self):
        """失败的请求不能打标，否则一次配额/网络故障就让这只股票整月不再刷新。"""
        self.mgr.get_low_freq_marker = lambda code, kind: None
        stamp = []
        self.mgr.set_low_freq_marker = lambda c, k, p: stamp.append((c, k, p))
        with patch('core.data.baostock_main.fetch_adjust_factor',
                   side_effect=QuotaExceededError('Daily Baostock API quota exceeded')):
            self.mgr.update_adjust_factor_data('sh.600000')
        self.assertEqual(stamp, [])

    def test_stamps_and_saves_when_data_returned(self):
        self.mgr.get_low_freq_marker = lambda code, kind: None
        stamp, saved = [], []
        self.mgr.set_low_freq_marker = lambda c, k, p: stamp.append((c, k, p))
        self.mgr._save_adjust_factor = lambda df: saved.append(df)
        self.mgr._conn = sqlite3.connect(':memory:')
        self.mgr._conn.execute('CREATE TABLE adjust_factor (code TEXT, date TEXT)')
        df = pd.DataFrame({'code': ['sh.600000'], 'date': ['2026-10-08']})
        with patch('core.data.baostock_main.fetch_adjust_factor', return_value=df):
            self.mgr.update_adjust_factor_data('sh.600000')
        self.assertEqual(len(saved), 1)
        self.assertEqual(stamp, [('sh.600000', ADJUST_FACTOR_KIND, self.this_month)])

    def test_save_failure_leaves_unstamped(self):
        """落库失败同样不算「检查过」，下次重试。"""
        self.mgr.get_low_freq_marker = lambda code, kind: None
        stamp = []
        self.mgr.set_low_freq_marker = lambda c, k, p: stamp.append((c, k, p))

        def boom(df):
            raise RuntimeError('disk full')

        self.mgr._save_adjust_factor = boom
        self.mgr._conn = sqlite3.connect(':memory:')
        self.mgr._conn.execute('CREATE TABLE adjust_factor (code TEXT, date TEXT)')
        df = pd.DataFrame({'code': ['sh.600000'], 'date': ['2026-10-08']})
        with patch('core.data.baostock_main.fetch_adjust_factor', return_value=df):
            self.mgr.update_adjust_factor_data('sh.600000')
        self.assertEqual(stamp, [])


if __name__ == '__main__':
    unittest.main()
