"""配额 / 季度护栏的回归测试。不访问行情 API、不碰生产数据库。

背景（2026-10-05~07 连续三晚定时任务 exit 1）：
  * 财务增量每晚为全市场 5000+ 只各请求一次「2026Q4」——报告期 12-31 还没到，
    答案必然是空，纯烧配额；
  * 配额打满后每个 worker 仍会为剩下每只股票各开 8 次 SQLite 连接抛异常，
    白跑几小时把日志刷到 1MB；
  * 指数落库排在个股阶段之后，被配额饿死，`ingest_index` 抛出的
    QuotaExceededError 无人接住 → 进程 exit 1 → 调度器的覆盖校验和成交价
    回填全被跳过（而当天日线其实早就写好了）。

这里只加载待测函数（AST 抽取），避免 import 期就建立数据库连接 / 登录 baostock。
"""
import ast
import concurrent.futures
import contextlib
from datetime import datetime
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / 'core' / 'data' / 'baostock_main.py'
UPDATER_PATH = PROJECT_ROOT / 'scripts' / 'update_daily_data.py'


def load_functions(path, names, **namespace):
    """把指定顶层函数抽出来单独执行，省掉整个模块的导入副作用。"""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
    assert {node.name for node in functions} == set(names)
    module = types.ModuleType(path.stem)
    module.__dict__.update(namespace)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'),
         module.__dict__)
    return module


class QuarterPublishedTests(unittest.TestCase):
    """报告期没过完的季度不该发请求。"""

    def setUp(self):
        self.mod = load_functions(MAIN_PATH, ['quarter_published'],
                                  datetime=datetime, Optional=Optional,
                                  QUARTER_ENDS={1: '03-31', 2: '06-30',
                                                3: '09-30', 4: '12-31'})

    def test_current_quarter_unfinished_is_not_requested(self):
        # 2026-10-08：Q3 报告期已过（值得问），Q4 报告期 12-31 还没到（一定为空）
        self.assertTrue(self.mod.quarter_published(2026, 3, '2026-10-08'))
        self.assertFalse(self.mod.quarter_published(2026, 4, '2026-10-08'))

    def test_boundary_is_strict(self):
        # 报告期最后一天当天仍不可能有报表；次日才开始值得问
        self.assertFalse(self.mod.quarter_published(2026, 3, '2026-09-30'))
        self.assertTrue(self.mod.quarter_published(2026, 3, '2026-10-01'))

    def test_historical_quarters_are_requested(self):
        for year, quarter in [(2025, 1), (2025, 4), (2026, 1), (2026, 2)]:
            self.assertTrue(self.mod.quarter_published(year, quarter, '2026-10-08'))


class DrainFuturesQuotaTests(unittest.TestCase):
    """配额打满时立刻收手，不要为剩下几千只各抛一遍异常。"""

    def setUp(self):
        # 每次读时钟走 10s：让「距上次查额度 ≥5s」在第一次循环就成立，
        # 免得测试为了跨过节流阈值去真的 sleep。
        self.clock = [1000.0]

        def tick():
            self.clock[0] += 10.0
            return self.clock[0]

        self.pbar = Mock()
        self.mod = load_functions(
            MAIN_PATH, ['_drain_futures'],
            time=types.SimpleNamespace(time=tick),
            quota_remaining=lambda: 0,
            TASK_TIMEOUT_SECONDS=120,
            TASK_HARD_TIMEOUT_SECONDS=300,
            TASK_MAX_STALLS=5,
            _kill_executor_workers=lambda executor: None,
        )

    @staticmethod
    def _batch(n):
        """n 个真实 Future，字典值用代码串（函数只用它来打印/回传）。"""
        futures = {}
        for i in range(n):
            f = concurrent.futures.Future()
            f.set_result(None)
            futures[f] = f'code{i:03d}'
        return futures

    def test_aborts_on_first_round_once_quota_is_gone(self):
        futures = self._batch(4)
        order = list(futures)
        # 第一轮只完成 1 个，剩余 3 个仍 pending；配额已耗尽
        fake_wait = Mock(return_value=([order[0]], order[1:]))

        with patch.object(concurrent.futures, 'wait', fake_wait):
            remaining, killed, abort_reason = self.mod._drain_futures(
                futures, self.pbar, task_label='finance', executor=None)

        self.assertEqual(abort_reason, 'quota')
        self.assertFalse(killed)
        # 按提交顺序回传未处理的，交给调用方（它据此停止，不再重建进程池）
        self.assertEqual(remaining, [futures[f] for f in order[1:]])
        # 关键：只等了一轮就把剩下的取消了，没有继续 grind
        self.assertEqual(fake_wait.call_count, 1)
        # 1 次是结算已完成的那只，3 次是被取消的
        self.assertEqual(self.pbar.update.call_count, 1 + 3)

    def test_runs_to_completion_when_quota_is_available(self):
        self.mod.quota_remaining = lambda: 9999
        futures = self._batch(3)
        order = list(futures)
        fake_wait = Mock(return_value=(order, []))

        with patch.object(concurrent.futures, 'wait', fake_wait):
            remaining, killed, abort_reason = self.mod._drain_futures(
                futures, self.pbar, task_label='daily', executor=None)

        self.assertEqual(remaining, [])
        self.assertFalse(killed)
        self.assertIsNone(abort_reason)
        self.assertEqual(self.pbar.update.call_count, 3)


class IndexMacroStepTests(unittest.TestCase):
    """指数/货币供应这一步永不带崩整个每夜更新。"""

    def _load(self, **ns):
        ns.setdefault('datetime', datetime)
        ns.setdefault('print', print)
        return load_functions(UPDATER_PATH, ['_run_index_and_macro_step'], **ns)

    def test_swallows_exception_and_keeps_going(self):
        def boom(*a, **kw):
            raise RuntimeError('Daily Baostock API quota exceeded (45000/45000)')

        mod = self._load(update_index_and_macro=boom)
        with contextlib.redirect_stdout(__import__('io').StringIO()) as buf:
            mod._run_index_and_macro_step(True, True, '2005-01-01')
        self.assertIn('不中断个股同步', buf.getvalue())

    def test_reports_partial_failure_without_raising(self):
        mod = self._load(update_index_and_macro=lambda *a, **kw: False)
        with contextlib.redirect_stdout(__import__('io').StringIO()) as buf:
            mod._run_index_and_macro_step(True, True, '2005-01-01')
        self.assertIn('未完全成功', buf.getvalue())

    def test_passes_flags_through(self):
        calls = []

        def record(start, end, do_index=True, do_money=True):
            calls.append((start, do_index, do_money))
            return True

        mod = self._load(update_index_and_macro=record)
        mod._run_index_and_macro_step(False, True, '2005-01-01')
        self.assertEqual([(c[1], c[2]) for c in calls], [(False, True)])


if __name__ == '__main__':
    unittest.main()
