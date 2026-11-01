"""Where torch gets imported, and which thread touches CUDA.

Deliberately a separate file from `test_recap.py`.  That suite's hard property is that
the recap can be read with no model library present at all -- it installs a meta path
finder that fails the run if `torch` or `transformers` is imported.  The tests here need
the opposite: a *stub* torch in `sys.modules`, so they can watch which thread calls into
it.  Mixing the two would cost the recap suite the guarantee it exists to make, so this
file stands alone and `test_recap.py` must never import it.

What is actually being guarded:

Pressing "Run Capture" used to kill the process with a Windows access violation -- no
Python exception, so nothing in the application's own error handling ever saw it.  The
faulting thread was inside the process's first `import torch`, running on a worker
QThread while the Qt event loop pumped Windows messages.  The same import raised a
first-chance RPC_E_CANTCALLOUT_ININPUTSYNCCALL on *every* run, crash or not.  The fix is
`TensorScope.preimport_model_stack()`, called on the GUI thread before any worker starts,
plus one long-lived `ModelWorker` thread that owns load, generate and unload so a CUDA
context is never created on a thread that then exits.

Both halves are invisible at rest: the application works either way, most of the time.
That is exactly why they need tests.

    ./.venv/Scripts/python.exe test_threading.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TENSORSCOPE_CRASH_LOG", "0")

APP_DIR = Path(__file__).resolve().parent
SOURCE = APP_DIR / "TensorScope.py"

MAIN_THREAD = threading.get_ident()
CALLS: list[tuple[str, int]] = []       # (what happened, which thread it happened on)


def _record(what: str) -> None:
    CALLS.append((what, threading.get_ident()))


def _install_stub_model_stack() -> None:
    """Put fake torch and transformers modules in sys.modules.

    A `sys.modules` hit short-circuits the import machinery entirely, so
    `preimport_model_stack()` and every deferred `import torch` in the application find
    these without any real library being present -- and without this test needing a GPU,
    8 GB of weights, or thirty seconds.
    """
    if isinstance(sys.modules.get("torch"), types.ModuleType) and \
            getattr(sys.modules["torch"], "_tensorscope_stub", False):
        return

    torch = types.ModuleType("torch")
    torch._tensorscope_stub = True
    cuda = types.ModuleType("torch.cuda")

    def is_available() -> bool:
        _record("torch.cuda.is_available")
        return False                     # keeps the VRAM warning out of the way

    cuda.is_available = is_available
    cuda.empty_cache = lambda: _record("torch.cuda.empty_cache")
    cuda.get_device_properties = lambda index: types.SimpleNamespace(
        total_memory=16 * 10 ** 9)
    torch.cuda = cuda
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"
    torch.no_grad = lambda: _NullContext()

    transformers = types.ModuleType("transformers")
    transformers._tensorscope_stub = True
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    masking_utils = types.ModuleType("transformers.masking_utils")
    transformers.modeling_utils = modeling_utils
    transformers.masking_utils = masking_utils

    sys.modules.update({
        "torch": torch, "torch.cuda": cuda,
        "transformers": transformers,
        "transformers.modeling_utils": modeling_utils,
        "transformers.masking_utils": masking_utils,
    })


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


_install_stub_model_stack()

from PyQt5.QtCore import QEventLoop, QTimer                           # noqa: E402
from PyQt5.QtWidgets import QApplication                              # noqa: E402

import TensorScope as app                                             # noqa: E402

QAPP = QApplication.instance() or QApplication([])


def pump(milliseconds: int = 60) -> None:
    """Let the event loop run, so zero-timers and queued signals are delivered."""
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec_()


class ImportSiteTests(unittest.TestCase):
    """Static facts about where the model stack may be imported."""

    @staticmethod
    def module_level_imports(path: Path) -> set[str]:
        """Top-level import names only -- the ones that run at import time."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
        return names

    def test_the_app_never_imports_the_model_stack_at_module_scope(self):
        """Browsing saved runs must work on a machine with no torch installed."""
        imported = self.module_level_imports(SOURCE)
        self.assertNotIn("torch", imported)
        self.assertNotIn("transformers", imported)

    def test_preimport_model_stack_imports_both_libraries(self):
        """The helper is the single place the GUI thread warms the stack."""
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next((node for node in tree.body
                         if isinstance(node, ast.FunctionDef)
                         and node.name == "preimport_model_stack"), None)
        self.assertIsNotNone(function, "preimport_model_stack() is the fix; do not remove it")
        imported = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertIn("torch", imported)
        self.assertIn("transformers", imported)

    def test_the_load_path_calls_it_before_reaching_the_worker(self):
        """A `load_requested` emit that is not preceded by the warm-up is the bug back."""
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        window = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef)
                      and node.name == "TensorScopeMainWindow")
        start_load = next((node for node in window.body
                           if isinstance(node, ast.FunctionDef)
                           and node.name == "_start_load"), None)
        self.assertIsNotNone(start_load)
        called = {node.func.id for node in ast.walk(start_load)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("preimport_model_stack", called)
        # And the emit lives in the same function, after it.
        source = ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), start_load)
        self.assertLess(source.index("preimport_model_stack"),
                        source.index("load_requested.emit"))

    def test_only_one_worker_class_owns_torch(self):
        """The two one-shot QThreads are gone; a reintroduced one would split the context."""
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("ModelWorker", classes)
        self.assertNotIn("LoadWorker", classes)
        self.assertNotIn("GenerationWorker", classes)


class LoadThreadTests(unittest.TestCase):
    """Which thread actually runs each piece of work."""

    def setUp(self):
        CALLS.clear()
        self._silence_dialogs()
        # A scratch database, never the shipped one: tensorscope_runs.sqlite3 is a hashed
        # three-tier fixture that test_recap.py measures.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.window = app.TensorScopeMainWindow(
            db_path=Path(self._tmp.name) / "probe.sqlite3")
        self.loads: list[tuple[str, int]] = []
        self.generates: list[tuple[str, int, str]] = []
        self._patch_capture()

    def _silence_dialogs(self):
        """Stop the modal boxes.

        `load_failed` and `run_failed` end in QMessageBox.critical, which spins a nested
        event loop.  Inside a queued slot, under the offscreen platform, that never
        returns and takes the process with it -- which has nothing to do with what these
        tests are about.
        """
        saved = {name: getattr(app.QMessageBox, name)
                 for name in ("critical", "information", "warning", "question")}

        def restore():
            for name, original in saved.items():
                setattr(app.QMessageBox, name, original)

        self.addCleanup(restore)
        app.QMessageBox.critical = staticmethod(lambda *a, **k: None)
        app.QMessageBox.information = staticmethod(lambda *a, **k: None)
        app.QMessageBox.warning = staticmethod(lambda *a, **k: app.QMessageBox.Ok)
        app.QMessageBox.question = staticmethod(lambda *a, **k: app.QMessageBox.Yes)

    def tearDown(self):
        window = self.window
        window._busy = None
        if window.model_thread is not None:
            window.model_thread.quit()
            window.model_thread.wait(5_000)
        window.deleteLater()
        pump(10)

    def _patch_capture(self):
        """Replace the real load/generate with recorders, keeping the Qt plumbing real."""
        loads, generates = self.loads, self.generates

        def fake_load(capture_self, progress=None):
            loads.append((capture_self.model_id, threading.get_ident()))
            capture_self.model = object()      # makes `loaded` true
            capture_self.device = "cpu"
            if progress:
                progress("stub load")

        def fake_describe(capture_self):
            return f"{capture_self.model_id} — stub"

        def fake_generate(capture_self, prompt, max_new_tokens=128, on_text=None):
            generates.append((capture_self.model_id, threading.get_ident(), prompt))
            raise app.CaptureProtocolError("stub: no real generation in this test")

        self._saved = (app.ModelCapture.load, app.ModelCapture.describe,
                       app.ModelCapture.generate)
        app.ModelCapture.load = fake_load
        app.ModelCapture.describe = fake_describe
        app.ModelCapture.generate = fake_generate
        self.addCleanup(self._restore)

    def _restore(self):
        app.ModelCapture.load, app.ModelCapture.describe, app.ModelCapture.generate = \
            self._saved

    def _load_and_settle(self):
        self.window.model_id.setText("stub/model")
        self.window.load_model()
        pump(150)

    def test_the_model_stack_is_imported_on_the_gui_thread(self):
        """The regression test.

        `_vram_is_sufficient` runs right after the warm-up and on the same thread, so the
        stub's `torch.cuda.is_available` records the thread the import happened on.  If
        this lands on a worker thread, the access violation is back.
        """
        self._load_and_settle()
        probes = [ident for what, ident in CALLS if what == "torch.cuda.is_available"]
        self.assertTrue(probes, "the load path never touched torch on any thread")
        for ident in probes:
            self.assertEqual(ident, MAIN_THREAD)

    def test_load_and_generate_run_on_one_worker_thread_that_is_not_the_gui(self):
        """A CUDA context must not be created on a thread that then exits."""
        self._load_and_settle()
        self.assertEqual(len(self.loads), 1, self.loads)
        load_thread = self.loads[0][1]
        self.assertNotEqual(load_thread, MAIN_THREAD)

        self.window.prompt.setPlainText("what is your favorite color?")
        self.window.run_model()
        pump(150)
        self.assertEqual(len(self.generates), 1, self.generates)
        self.assertEqual(self.generates[0][1], load_thread)

    def test_a_second_load_request_while_one_is_in_flight_is_ignored(self):
        """Rebinding a live worker used to abort the process."""
        self.window.model_id.setText("stub/model")
        self.window.load_model()
        self.window.load_model()        # the zero-timer for the first has not fired yet
        pump(150)
        self.assertEqual(len(self.loads), 1, self.loads)

    def test_closing_the_window_stops_the_worker_thread(self):
        self._load_and_settle()
        thread = self.window.model_thread
        self.assertIsNotNone(thread)
        self.assertTrue(thread.isRunning())
        self.window.close()
        pump(50)
        self.assertTrue(thread.isFinished(), "closeEvent left the torch thread running")

    def test_browsing_never_starts_a_torch_thread(self):
        """Opening the window and reading saved runs must cost no thread at all."""
        self.assertIsNone(self.window.model_thread)
        self.assertIsNone(self.window.model_worker)


class UnloadTests(unittest.TestCase):
    """Teardown has to free the model without reaching for torch itself."""

    class _Handle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    def test_unload_removes_every_hook_and_clears_every_holder(self):
        capture = app.ModelCapture("stub/model")
        handles = [self._Handle() for _ in range(4)]
        capture._hooks.extend(handles)
        capture.model = object()
        capture.tokenizer = object()
        capture.attention = types.SimpleNamespace(sink={"q": "tensor"})
        capture._scratch[(0, "q")] = "tensor"
        capture._embedding_scratch["embedding"] = "tensor"
        capture._per_layer[0] = {"attention_scores": "tensor"}
        capture._capturing = True

        capture.unload()

        self.assertTrue(all(handle.removed for handle in handles),
                        "a hook left installed keeps the model alive through its closure")
        self.assertEqual(capture._hooks, [])
        self.assertEqual(capture._scratch, {})
        self.assertEqual(capture._embedding_scratch, {})
        self.assertEqual(capture._per_layer, {})
        self.assertFalse(capture._capturing)
        self.assertIsNone(capture.model)
        self.assertIsNone(capture.tokenizer)
        self.assertIsNone(capture.attention)

    def test_unload_tolerates_a_capture_that_never_loaded(self):
        """Browse-only mode may unload a ModelCapture that holds nothing."""
        app.ModelCapture("stub/model").unload()

    def test_unload_reaches_torch_through_sys_modules_only(self):
        """`import torch` in unload() would drag the library in on a browse-only run."""
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        model_capture = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "ModelCapture")
        unload = next(node for node in model_capture.body
                      if isinstance(node, ast.FunctionDef) and node.name == "unload")
        for node in ast.walk(unload):
            self.assertNotIsInstance(node, ast.Import)
            self.assertNotIsInstance(node, ast.ImportFrom)
        self.assertIn("sys.modules.get(\"torch\")", ast.get_source_segment(source, unload))


if __name__ == "__main__":
    unittest.main(verbosity=2)
