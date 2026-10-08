"""每日更新的日志/调度回归测试，不访问行情 API 或生产数据库。

只加载待测函数，避免 app.deps 的导入期数据库初始化。
"""
import ast
import asyncio
import contextlib
from datetime import datetime
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATER_PATH = PROJECT_ROOT / 'scripts' / 'update_daily_data.py'
SCHEDULER_PATH = PROJECT_ROOT / 'quantification-system' / 'backend' / 'app' / 'scheduler.py'


def load_functions(path, names, **namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
    assert {node.name for node in functions} == set(names)
    module = types.ModuleType(path.stem)
    module.__dict__.update(namespace)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


class DailyUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.updater = load_functions(
            UPDATER_PATH, ['_stdout_to_log', '_socket_timeout', 'update_all_stocks'],
            contextlib=contextlib, datetime=datetime, os=os, sys=sys, socket=socket,
            PROJECT_ROOT=str(self.root), BAOSTOCK_SOCKET_TIMEOUT=30,
            _update_all_stocks_impl=Mock(),
        )
        self.scheduler = load_functions(
            SCHEDULER_PATH, ['_run_daily_data_update', 'daily_data_update_job'],
            datetime=datetime, os=os, sys=sys, subprocess=subprocess, asyncio=asyncio,
            logger=logging.getLogger('test_daily_scheduler'),
            get_project_root=lambda: str(self.root),
            verify_daily_update=Mock(), auto_fill_paper_trading_prices=Mock(),
        )

    def test_utf8_stdout_stderr_append_and_restore(self):
        original = sys.stdout, sys.stderr
        for _ in range(2):
            with self.updater._stdout_to_log() as log_path:
                print('更新完成 ✓')
                print('错误详情 ⚠', file=sys.stderr)
                # 行缓冲应立即落盘，不必等任务结束。
                self.assertIn('更新完成 ✓', Path(log_path).read_text(encoding='utf-8'))
            self.assertEqual((sys.stdout, sys.stderr), original)
        contents = Path(log_path).read_text(encoding='utf-8')
        self.assertEqual(contents.count('更新完成 ✓'), 2)
        self.assertEqual(contents.count('错误详情 ⚠'), 2)

    def test_missing_or_broken_console_is_not_touched(self):
        broken = Mock()
        broken.flush.side_effect = OSError(6, 'invalid handle')
        broken.write.side_effect = OSError(6, 'invalid handle')
        for stream in (None, broken):
            with self.subTest(stream=stream):
                with patch.object(sys, 'stdout', stream), patch.object(sys, 'stderr', stream):
                    with patch.object(os, 'dup2', side_effect=AssertionError('must not replace fd 1')):
                        with self.updater._stdout_to_log():
                            print('日志正常')
                    self.assertIs(sys.stdout, stream)
                    self.assertIs(sys.stderr, stream)
        broken.flush.assert_not_called()
        broken.write.assert_not_called()

    def test_body_failure_propagates_and_restores_streams_and_timeout(self):
        original = sys.stdout, sys.stderr
        old_timeout = socket.getdefaulttimeout()
        error = RuntimeError('update failed')
        self.updater._update_all_stocks_impl.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.updater.update_all_stocks()
        self.assertIs(caught.exception, error)
        self.assertEqual((sys.stdout, sys.stderr), original)
        self.assertEqual(socket.getdefaulttimeout(), old_timeout)
        self.updater._update_all_stocks_impl.assert_called_once_with(
            incremental=True, workers=None, start_date=None, end_date='2030-01-01',
            do_index=True, do_money=True, index_start='2005-01-01',
        )

    def test_log_open_failure_is_not_reported_as_success(self):
        with patch('builtins.open', side_effect=PermissionError('log not writable')):
            with self.assertRaises(PermissionError):
                self.updater.update_all_stocks()
        self.updater._update_all_stocks_impl.assert_not_called()

    def test_subprocess_has_explicit_handles_and_utf8_environment(self):
        original = sys.stdout, sys.stderr
        env = os.environ.copy()
        with patch.object(subprocess, 'run') as run:
            log_path = self.scheduler._run_daily_data_update()
        args, kwargs = run.call_args
        self.assertEqual(args[0], [sys.executable, '-u', str(self.root / 'scripts' / 'update_daily_data.py')])
        self.assertEqual(kwargs['cwd'], str(self.root))
        self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(kwargs['stdout'].name, log_path)
        self.assertTrue(kwargs['stdout'].closed)
        self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
        self.assertTrue(kwargs['check'])
        self.assertEqual(kwargs['env']['PYTHONIOENCODING'], 'utf-8')
        self.assertEqual(kwargs['env']['PYTHONUNBUFFERED'], '1')
        self.assertEqual(dict(os.environ), env)
        self.assertEqual((sys.stdout, sys.stderr), original)

    def write_child(self, code):
        script = self.root / 'scripts' / 'update_daily_data.py'
        script.parent.mkdir(exist_ok=True)
        script.write_text(textwrap.dedent(code), encoding='utf-8')

    def test_real_spawn_worker_stdout_and_stderr_reach_log(self):
        self.write_child('''
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            import sys

            def worker():
                print('worker stdout 中文 ✓', flush=True)
                print('worker stderr 中文 ⚠', file=sys.stderr, flush=True)
                return 42

            if __name__ == '__main__':
                assert sys.stdin.read() == ''
                print('parent stdout 中文 ✓', flush=True)
                print('parent stderr 中文 ⚠', file=sys.stderr, flush=True)
                with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
                    assert pool.submit(worker).result(timeout=30) == 42
        ''')
        log_path = self.scheduler._run_daily_data_update()
        contents = Path(log_path).read_text(encoding='utf-8')
        for expected in ('parent stdout 中文 ✓', 'parent stderr 中文 ⚠',
                         'worker stdout 中文 ✓', 'worker stderr 中文 ⚠'):
            self.assertIn(expected, contents)

    def test_real_child_failure_includes_exit_code_and_log_path(self):
        self.write_child('''
            import sys
            print('失败原因 ✓', file=sys.stderr, flush=True)
            sys.exit(7)
        ''')
        with self.assertRaisesRegex(RuntimeError, '退出码 7.*daily_update_') as caught:
            self.scheduler._run_daily_data_update()
        self.assertIsInstance(caught.exception.__cause__, subprocess.CalledProcessError)
        logs = list((self.root / 'diagnose_output').glob('*.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('失败原因 ✓', logs[0].read_text(encoding='utf-8'))

    def test_failed_job_raises_and_skips_success_hooks(self):
        error = OSError(6, 'invalid handle')
        with patch.object(self.scheduler, '_run_daily_data_update', side_effect=error):
            with self.assertLogs('test_daily_scheduler', level='ERROR') as logs:
                with self.assertRaises(OSError) as caught:
                    asyncio.run(self.scheduler.daily_data_update_job())
        self.assertIs(caught.exception, error)
        self.assertIsNotNone(logs.records[0].exc_info)
        self.scheduler.verify_daily_update.assert_not_called()
        self.scheduler.auto_fill_paper_trading_prices.assert_not_called()

    def test_apscheduler_emits_error_event_on_failure(self):
        from apscheduler.events import EVENT_JOB_ERROR
        from apscheduler.executors.base import run_coroutine_job

        job = types.SimpleNamespace(
            id='daily_data_update', func=self.scheduler.daily_data_update_job,
            args=(), kwargs={}, misfire_grace_time=None,
        )
        error = RuntimeError('child update failed')
        with patch.object(self.scheduler, '_run_daily_data_update', side_effect=error):
            with self.assertLogs('test_daily_scheduler', level='ERROR'):
                events = asyncio.run(run_coroutine_job(
                    job, 'default', [datetime.now()], 'test_daily_scheduler',
                ))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].code, EVENT_JOB_ERROR)
        self.assertIs(events[0].exception, error)

    def test_successful_job_runs_hooks_in_order(self):
        calls = []
        self.scheduler.verify_daily_update.side_effect = lambda: calls.append('verify')
        self.scheduler.auto_fill_paper_trading_prices.side_effect = lambda: calls.append('fill')
        with patch.object(self.scheduler, '_run_daily_data_update', side_effect=lambda: calls.append('update')):
            asyncio.run(self.scheduler.daily_data_update_job())
        self.assertEqual(calls, ['update', 'verify', 'fill'])

    @unittest.skipUnless(sys.platform == 'win32', 'requires WindowsConsoleIO')
    def test_windows_console_handle_survives_redirect(self):
        code = textwrap.dedent('''
            import ctypes
            import io
            import os
            import runpy
            import sys

            helpers = runpy.run_path(sys.argv[1], run_name='regression_helpers')
            module = helpers['load_functions'](
                helpers['UPDATER_PATH'], ['_stdout_to_log'],
                contextlib=helpers['contextlib'], datetime=helpers['datetime'],
                os=os, sys=sys, PROJECT_ROOT=sys.argv[2],
            )
            ctypes.WinDLL('kernel32', use_last_error=True).AllocConsole()
            try:
                fd = os.open('CONOUT$', os.O_RDWR)
            except OSError:
                sys.exit(77)
            os.dup2(fd, 1)
            os.close(fd)
            original = sys.stdout
            sys.stdout = io.TextIOWrapper(io._WindowsConsoleIO(1, 'w', closefd=False), encoding='utf-8')
            try:
                print('Before redirect', flush=True)
                with module._stdout_to_log() as path:
                    print('Inside redirect 中文 ✓', flush=True)
                print('After redirect', flush=True)
            finally:
                sys.stdout = original
        ''')
        result = subprocess.run(
            [sys.executable, '-c', code, str(Path(__file__).resolve()), str(self.root)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=30,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 77:
            self.skipTest('Windows console unavailable')
        self.assertEqual(result.returncode, 0, result.stderr)
        logs = list((self.root / 'diagnose_output').glob('*.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('Inside redirect 中文 ✓', logs[0].read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
