from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from compute_as_a_teacher.training.errors import TrainingError  # noqa: E402
from compute_as_a_teacher.training import verl_adapter  # noqa: E402
from compute_as_a_teacher.training.verl_adapter import (  # noqa: E402
    exclusive_launch,
    run_command_with_log,
)


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    return environment


class LaunchLifecycleTests(unittest.TestCase):
    def test_repeated_interrupts_cannot_escape_process_group_cleanup(self) -> None:
        process = Mock(pid=31415)
        with patch.object(
            verl_adapter,
            "_wait_for_process",
            side_effect=(False, False, True),
        ), patch.object(
            verl_adapter,
            "_signal_process_group",
            side_effect=(KeyboardInterrupt(), SystemExit(143), True),
        ), patch.object(
            verl_adapter, "_wait_for_group_exit", return_value=True
        ):
            verl_adapter._stop_and_wait(process)

    def test_repeated_sigterm_cannot_escape_process_group_cleanup(self) -> None:
        worker_code = (
            "import os,signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii'); "
            "print('ready', flush=True); time.sleep(60)"
        )
        parent_code = (
            "import os,sys; from pathlib import Path; "
            "from compute_as_a_teacher.training import verl_adapter as v; "
            "v._PROCESS_EXIT_GRACE_SECONDS=0.05; v._PROCESS_GROUP_STOP_SECONDS=0.3; "
            "run_dir=Path(sys.argv[1]); "
            "context=v.exclusive_launch(run_dir); context.__enter__(); "
            "v.run_command_with_log((sys.executable,'-c',sys.argv[2],"
            "str(run_dir/'child.pid')),cwd=run_dir,environment=os.environ.copy(),"
            "log_path=run_dir/'trainer.log')"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            parent = subprocess.Popen(
                [sys.executable, "-c", parent_code, str(run_dir), worker_code],
                cwd=REPOSITORY_ROOT,
                env=_subprocess_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.addCleanup(self._stop_process, parent)
            assert parent.stdout is not None
            self.assertEqual(parent.stdout.readline().strip(), "ready")
            child_pid = int((run_dir / "child.pid").read_text(encoding="ascii"))
            parent.send_signal(signal.SIGTERM)
            time.sleep(0.1)
            parent.send_signal(signal.SIGTERM)
            return_code = parent.wait(timeout=5)
            assert parent.stderr is not None
            self.assertEqual(return_code, 128 + signal.SIGTERM, parent.stderr.read())
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            with exclusive_launch(run_dir):
                pass

    def test_process_crash_releases_advisory_lock_without_deleting_inode(self) -> None:
        code = (
            "import os,sys; from pathlib import Path; "
            "from compute_as_a_teacher.training.verl_adapter import exclusive_launch; "
            "run_dir=Path(sys.argv[1]); "
            "context=exclusive_launch(run_dir); context.__enter__(); "
            "os._exit(23)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            completed = subprocess.run(
                [sys.executable, "-c", code, str(run_dir)],
                cwd=REPOSITORY_ROOT,
                env=_subprocess_environment(),
                check=False,
            )
            self.assertEqual(completed.returncode, 23)
            self.assertTrue((run_dir / ".launch.lock").is_file())
            with exclusive_launch(run_dir):
                pass

    def test_advisory_lock_rejects_a_live_process_then_recovers(self) -> None:
        code = (
            "import sys; from pathlib import Path; "
            "from compute_as_a_teacher.training.verl_adapter import exclusive_launch; "
            "run_dir=Path(sys.argv[1]); "
            "context=exclusive_launch(run_dir); context.__enter__(); "
            "print('locked', flush=True); sys.stdin.readline(); context.__exit__(None,None,None)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(run_dir)],
                cwd=REPOSITORY_ROOT,
                env=_subprocess_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.addCleanup(self._stop_process, process)
            assert process.stdout is not None
            self.assertEqual(process.stdout.readline().strip(), "locked")
            with self.assertRaisesRegex(TrainingError, "already locked"):
                with exclusive_launch(run_dir):
                    self.fail("a second process acquired the live launch lock")
            assert process.stdin is not None
            process.stdin.write("\n")
            process.stdin.flush()
            self.assertEqual(process.wait(timeout=5), 0)
            with exclusive_launch(run_dir):
                pass

    def test_interrupt_and_exception_terminate_and_reap_child_before_unlock(self) -> None:
        child_code = (
            "import os,sys,time; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii'); "
            "print('ready', flush=True); time.sleep(60)"
        )
        for failure in (KeyboardInterrupt(), RuntimeError("output sink failed")):
            with self.subTest(failure=type(failure).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    run_dir = Path(temporary)
                    pid_path = run_dir / "child.pid"
                    with self.assertRaises(type(failure)):
                        with exclusive_launch(run_dir):
                            with patch("builtins.print", side_effect=failure):
                                run_command_with_log(
                                    (sys.executable, "-c", child_code, str(pid_path)),
                                    cwd=run_dir,
                                    environment=os.environ.copy(),
                                    log_path=run_dir / "logs/trainer.log",
                                )
                    child_pid = int(pid_path.read_text(encoding="ascii"))
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                    with exclusive_launch(run_dir):
                        pass

    def test_sigterm_is_forwarded_and_child_is_reaped_before_unlock(self) -> None:
        child_code = (
            "import os,sys,time; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii'); "
            "print('ready', flush=True); time.sleep(60)"
        )
        parent_code = (
            "import os,sys; from pathlib import Path; "
            "from compute_as_a_teacher.training.verl_adapter import "
            "exclusive_launch,run_command_with_log; "
            "run_dir=Path(sys.argv[1]); child_code=sys.argv[2]; "
            "context=exclusive_launch(run_dir); context.__enter__(); "
            "run_command_with_log((sys.executable,'-c',child_code,str(run_dir/'child.pid')),"
            "cwd=run_dir,environment=os.environ.copy(),log_path=run_dir/'trainer.log')"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            parent = subprocess.Popen(
                [sys.executable, "-c", parent_code, str(run_dir), child_code],
                cwd=REPOSITORY_ROOT,
                env=_subprocess_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.addCleanup(self._stop_process, parent)
            assert parent.stdout is not None
            self.assertEqual(parent.stdout.readline().strip(), "ready")
            child_pid = int((run_dir / "child.pid").read_text(encoding="ascii"))
            parent.send_signal(signal.SIGTERM)
            return_code = parent.wait(timeout=10)
            assert parent.stderr is not None
            self.assertEqual(
                return_code,
                128 + signal.SIGTERM,
                parent.stderr.read(),
            )
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            with exclusive_launch(run_dir):
                pass

    def test_exited_leader_cannot_leave_same_group_workers(self) -> None:
        worker_code = (
            "import os,sys,time; from pathlib import Path; "
            "os.close(0); os.close(1); os.close(2); "
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii'); "
            "time.sleep(60)"
        )
        leader_code = (
            "import subprocess,sys,time; from pathlib import Path; "
            "path=Path(sys.argv[2]); "
            "subprocess.Popen((sys.executable,'-c',sys.argv[1],sys.argv[2]),"
            "close_fds=True); "
            "[time.sleep(0.01) for _ in range(500) if not path.exists()]; "
            "assert path.exists()"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            pid_path = run_dir / "worker.pid"
            return_code = run_command_with_log(
                (sys.executable, "-c", leader_code, worker_code, str(pid_path)),
                cwd=run_dir,
                environment=os.environ.copy(),
                log_path=run_dir / "logs/trainer.log",
            )
            self.assertEqual(return_code, 0)
            worker_pid = int(pid_path.read_text(encoding="ascii"))
            with self.assertRaises(ProcessLookupError):
                os.kill(worker_pid, 0)

    @staticmethod
    def _stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


if __name__ == "__main__":
    unittest.main()
