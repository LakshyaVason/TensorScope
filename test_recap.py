"""Saved-run UI regression checks, using real captures and no model imports.

Run: ./.venv/Scripts/python.exe test_recap.py [path/to/runs.sqlite3]
The existing database is opened read-only and hashed before/after every test run.
No model weights are loaded and no example tensors are substituted for captures.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import importlib.abc
import itertools
import json
import os
import re
from pathlib import Path
import sqlite3
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ForbidModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "accelerate"}:
            raise AssertionError(f"Saved-run browsing tried to import {fullname}")
        return None


sys.meta_path.insert(0, ForbidModelImports())

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QMouseEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QLabel, QPushButton, QTableWidget

import TensorScope as app
import tensorscope_diagrams as diagrams
from tensor_widgets import AttentionExplorer, TensorInspector, TensorTableModel, exact_scalar
from tensorscope_common import (
    COLORS, AnnotatedValueRow, Disclosure, EvidenceBadge, PassTimeline, PredictCheck,
    ResponseTextView, StepReveal,
)
from tensorscope_content import (
    ATTENTION_SINK_THRESHOLD, CHAPTERS, CONCEPTUAL_VARIABLES, CONTINUATION,
    EVIDENCE_CONCEPTUAL, EVIDENCE_INLINE_TIERS, EVIDENCE_KINDS, EVIDENCE_OBSERVED,
    EVIDENCE_UNATTESTED, LOGITS_UNAVAILABLE, NOTEBOOK_CHECKS, NOTEBOOK_OPERATIONS,
    NOTEBOOK_TEXT, NOTEBOOK_TEXT_FIELDS, PROMPT_SPAN_UNKNOWN, PROVENANCE_ABSENT,
    PROVENANCE_CAPTURED, PROVENANCE_DISPLAY, PROVENANCE_LEARNED, REFERENCE_TOPICS,
    ROLE_LEGEND, TELEMETRY_UNAVAILABLE, TEMPLATE_BAND, TYPED_BAND, VARIABLE_ROLES,
    WEIGHTS_NOT_PROBABILITIES, learn_facts, prompt_token_bands, readable_spelling,
)
import tensorscope_operations as operations
from tensorscope_diagrams import CausalMaskTriangle, Diagram
from tensorscope_ui import DEFAULT_MODE, MODE_INDEX, MODES, ComputationRecap
from tensorscope_variables import (
    EquationRow, VariableChip, VariablePanel, VariableRegistry,
)
from tensorscope_views import NotebookView, RawView, ReferenceView


def _database_argument():
    """Take argv[1] as the database only when it actually names one.

    `unittest` reads the rest of argv as test names, so an unconditional `pop(1)` turns
    `python test_recap.py SavedCaptureTests.test_x` into "open a database called
    SavedCaptureTests.test_x" -- which fails in setUpClass with a FileNotFoundError that
    says nothing about the real mistake, and which also makes `pytest test_recap.py`
    swallow pytest's own first argument.
    """
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        candidate = Path(sys.argv[1])
        if candidate.suffix in {".sqlite3", ".sqlite", ".db"} or candidate.is_file():
            sys.argv.pop(1)
            return candidate
    return app.DB_PATH


DATABASE = _database_argument()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_arrays(capture):
    yield "embedding", capture.embedding
    yield "generated_embedding", capture.generated_embedding
    if capture.logits is not None:
        yield "logits", capture.logits
    for phase, layers in (("prefill", capture.layers), ("first_generated_token", capture.generated_layers)):
        for index, layer in sorted(layers.items()):
            for key, array in sorted(layer.tensors.items()):
                yield f"{phase}/{index}/{key}", array


def capture_digest(capture):
    """Hash every scalar's bytes as well as the dtype and shape, not a preview."""
    result = hashlib.sha256()
    for name, array in all_arrays(capture):
        result.update(name.encode())
        result.update(str((array.dtype.str, array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def evidence_claims(widget):
    """Every provenance claim under `widget`, whatever it looks like on screen.

    A badge, a value box's left edge, an edged sentence or table, a reveal step: each sets
    the `evidenceTier` property, so this sees all of them.  Reading only `EvidenceBadge`s
    stopped being enough the moment OBSERVED and CONCEPTUAL lost their words.
    """
    from PyQt5.QtWidgets import QWidget
    return [(child, child.property('evidenceTier')) for child in widget.findChildren(QWidget)
            if child.property('evidenceTier')]


def open_every_disclosure(widget, *, rounds=8):
    """Click open every collapsed Disclosure under `widget`, building its body.

    A Disclosure builds nothing until it is toggled, so an unopened one proves nothing --
    and opening one can reveal more of them, which is exactly what the notebook's
    per-layer group does. One pass would therefore stop short of the bodies that matter
    most. The bound is there so unexpectedly deep nesting fails loudly instead of
    silently truncating the walk.
    """
    for _ in range(rounds):
        pending = []
        for disclosure in widget.findChildren(Disclosure):
            button = disclosure.findChild(QPushButton)
            if button is not None and button.isCheckable() and not button.isChecked():
                pending.append(button)
        if not pending:
            return
        for button in pending:
            button.click()
    raise AssertionError("Disclosures kept appearing; nesting is deeper than expected")


def paint_every_diagram(test, widget):
    """Run every diagram's drawing code under `widget`, and check what it says.

    `paintEvent` output cannot be read back, and under the offscreen platform it may not
    run at all unless something asks for the pixels -- so an exception inside a `draw`
    would otherwise pass silently and reach a reader instead.

    This paints into a pixmap of its own rather than calling `QWidget.grab()`. Grabbing a
    child of a window that was never mapped crashes the offscreen platform outright, and
    painting directly is the sharper test anyway: it exercises exactly the arithmetic and
    text layout a diagram is responsible for, with none of Qt's render machinery in the
    way. `DiagramTests` covers the real `paintEvent` path, on parentless widgets where
    `grab()` is safe.
    """
    for diagram in widget.findChildren(Diagram):
        with test.subTest(diagram=type(diagram).__name__):
            size = QSize(max(360, diagram.canvas.width()),
                         max(diagram.BODY_HEIGHT, diagram.canvas.height()))
            pixmap = QPixmap(size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            try:
                diagram.draw(painter, QRect(1, 1, size.width() - 2, size.height() - 2))
            finally:
                painter.end()
            described = diagram.describe()
            test.assertTrue(described)
            test.assertTrue(all(str(text).strip() for text in described))


def section_text(widget):
    """Everything readable under a widget: labels and table cells alike.

    Table cells count. Half of what the notebook states in words it states in a two-column
    table -- the stopping decision and the closing summary both do -- and a helper that
    read only labels would quietly pass a test about text that is right there on screen.
    """
    pieces = [child.text() for child in widget.findChildren(QLabel)]
    for table in widget.findChildren(QTableWidget):
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is not None:
                    pieces.append(item.text())
    return chr(10).join(pieces)


def candidate_rows(widget):
    """The rows of a candidate-score table, as text, in the order they are displayed."""
    rows = []
    for table in widget.findChildren(QTableWidget):
        headings = [table.horizontalHeaderItem(column).text()
                    for column in range(table.columnCount())
                    if table.horizontalHeaderItem(column) is not None]
        if "Token ID" not in headings:
            continue
        for row in range(table.rowCount()):
            rows.append(tuple(
                table.item(row, column).text() if table.item(row, column) else ""
                for column in range(table.columnCount())))
    return rows


class ReadOnlyDatabase(app.RunDatabase):
    def _setup(self):
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def connect(self):
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection


class SavedCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])
        cls.database = ReadOnlyDatabase(DATABASE)
        cls.original_database_hash = file_digest(DATABASE)
        cls.captures = {row["id"]: cls.database.load(row["id"]) for row in cls.database.list_runs()}
        if not cls.captures:
            raise AssertionError("Supply a database containing real saved captures")

    @classmethod
    def tearDownClass(cls):
        assert file_digest(DATABASE) == cls.original_database_hash, "The saved database changed"
        assert not {"torch", "transformers", "accelerate"}.intersection(sys.modules)

    def test_load_every_saved_tensor_matches_database(self):
        """Use the persisted blob as the oracle for every loaded array and its dtype."""
        for run_id, capture in self.captures.items():
            with self.subTest(run_id=run_id), self.database.connect() as connection:
                capture.validate()
                row = connection.execute("SELECT embedding_blob FROM runs WHERE id = ?", (run_id,)).fetchone()
                expected = app.blob_to_array(row[0])
                self.assertEqual(expected.dtype, capture.embedding.dtype)
                self.assertEqual(expected.tobytes(), capture.embedding.tobytes())
                for row in connection.execute("SELECT * FROM tensors WHERE run_id = ?", (run_id,)):
                    name = row["name"]
                    if name == "final_logits":
                        actual = capture.logits
                    elif name == "first_generated_token:embedding":
                        actual = capture.generated_embedding
                    elif name.startswith("first_generated_token:"):
                        actual = capture.generated_layers[row["layer_index"]].tensors[name.split(":", 1)[1]]
                    else:
                        actual = capture.layers[row["layer_index"]].tensors[name]
                    expected = app.blob_to_array(row["data_blob"])
                    self.assertEqual(tuple(json.loads(row["shape_json"])), actual.shape)
                    self.assertEqual(row["dtype"], str(actual.dtype))
                    self.assertEqual(expected.dtype, actual.dtype)
                    self.assertEqual(expected.tobytes(), actual.tobytes())
                self.assertEqual(set(capture.layers), set(capture.generated_layers))
                for layers in (capture.layers, capture.generated_layers):
                    for layer in layers.values():
                        self.assertTrue(app.REQUIRED_LAYER_TENSORS <= layer.tensors.keys())

    def test_all_required_tensors_have_explanations(self):
        self.assertEqual(app.REQUIRED_LAYER_TENSORS, set(dict(app.TENSOR_LABELS)))
        self.assertEqual(app.REQUIRED_LAYER_TENSORS, set(app.TENSOR_EXPLANATIONS))
        self.assertTrue(all(text.strip() for text in app.TENSOR_EXPLANATIONS.values()))
        self.assertEqual(set(app.EXTRA_TENSOR_LABELS), set(app.EXTRA_TENSOR_EXPLANATIONS))
        self.assertEqual(set(app.TENSOR_SYMBOLS), set(app.VARIABLE_ROLES))
        # Every line of copy on either page, formatted against every saved run. A line
        # naming a fact the oldest run lacks fails here rather than in front of a reader.
        for run_id, capture in self.captures.items():
            facts = learn_facts(capture)
            with self.subTest(run_id=run_id):
                for topic in REFERENCE_TOPICS:
                    for field in ('heading', 'purpose', 'formula', 'caption', 'deeper'):
                        getattr(topic, field).format(**facts)
                for operation in NOTEBOOK_OPERATIONS + [CONTINUATION]:
                    filled = {**facts, **{name: 0 for name in operation.fields}}
                    for field in ('heading', 'lede', 'formula', 'takeaway', 'note',
                                  'action'):
                        getattr(operation, field).format(**filled)
                for key, text in NOTEBOOK_TEXT.items():
                    text.format(**{**facts, **{name: 0 for name in
                                               NOTEBOOK_TEXT_FIELDS.get(key, ())}})
                for name, check in NOTEBOOK_CHECKS.items():
                    filled = {**facts, **{key: 0 for key in check['fields']}}
                    check['question'].format(**filled)
                    check['explanation'].format(**filled)
                    for option in check['options']:
                        option.format(**filled)
                for key, role in VARIABLE_ROLES.items():
                    self.assertTrue(role.strip(), key)
                for key, spec in CONCEPTUAL_VARIABLES.items():
                    spec.role.format(**facts)
                    spec.shape_text.format(**facts)

    def test_virtual_tables_reach_every_saved_axis_without_mutation(self):
        """All heads/slices and boundary cells remain accessible at exact precision."""
        for run_id, capture in self.captures.items():
            before = capture_digest(capture)
            for name, array in all_arrays(capture):
                with self.subTest(run_id=run_id, tensor=name):
                    model = TensorTableModel(array)
                    self.assertIs(model.array, array)
                    expected_rows = array.shape[-2] if array.ndim >= 2 else array.size
                    expected_columns = array.shape[-1] if array.ndim >= 2 else 1
                    self.assertEqual(model.rowCount(), expected_rows)
                    self.assertEqual(model.columnCount(), expected_columns)
                    for leading in itertools.product(*(range(size) for size in array.shape[:-2])):
                        model.set_slice(leading)
                        for row, column in itertools.product({0, expected_rows - 1}, {0, expected_columns - 1}):
                            coordinate = leading + (row, column) if array.ndim >= 2 else (row,)
                            value = array[coordinate]
                            index = model.index(row, column)
                            self.assertEqual(model.tensor_index(row, column), coordinate)
                            self.assertEqual(model.value_at(row, column).tobytes(), value.tobytes())
                            self.assertEqual(model.data(index, Qt.DisplayRole), exact_scalar(value))
                            self.assertFalse(model.flags(index) & Qt.ItemIsEditable)
                            self.assertIn(str(coordinate), model.data(index, Qt.ToolTipRole))
            self.assertEqual(before, capture_digest(capture))

    def test_full_inspector_controls_reach_last_value(self):
        for capture in self.captures.values():
            before = capture_digest(capture)
            examples = [("embedding", capture.embedding), ("generated embedding", capture.generated_embedding)]
            if capture.logits is not None:
                examples.append(("logits", capture.logits))
            for layers in (capture.layers, capture.generated_layers):
                examples.extend(layers[max(layers)].tensors.items())
            for name, array in examples:
                with self.subTest(tensor=name):
                    inspector = TensorInspector(array, name)
                    for selector, size in zip(inspector.slice_selectors, array.shape[:-2]):
                        selector.setValue(size - 1)
                    row = inspector.model.rowCount() - 1
                    column = inspector.model.columnCount() - 1
                    inspector.row_jump.setValue(row)
                    inspector.column_jump.setValue(column)
                    inspector._jump()
                    coordinate = tuple(size - 1 for size in array.shape)
                    self.assertEqual(inspector.model.tensor_index(row, column), coordinate)
                    self.assertEqual(inspector.table.currentIndex().row(), row)
                    self.assertIn(exact_scalar(array[coordinate]), inspector.selection_label.text())
                    copied = inspector.table.copy_selection(include_indices=True)
                    self.assertEqual(copied, f"{coordinate}\t{exact_scalar(array[coordinate])}")
                    inspector.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertEqual(before, capture_digest(capture))

    def test_attention_context_both_phases_every_head(self):
        """Decoding query labels must differ from the cache-expanded key labels."""
        for run_id, capture in self.captures.items():
            before = capture_digest(capture)
            for phase, layers in (("prefill", capture.layers), ("first_generated_token", capture.generated_layers)):
                query_labels = capture.prompt_tokens if phase == "prefill" else capture.tokens[:1]
                key_labels = capture.prompt_tokens if phase == "prefill" else capture.prompt_tokens + capture.tokens[:1]
                for layer_index in {min(layers), max(layers)}:
                    with self.subTest(run_id=run_id, phase=phase, layer=layer_index):
                        explorer = AttentionExplorer(layers[layer_index].tensors, query_labels, key_labels,
                                                     layer_index=layer_index, phase=phase)
                        source = layers[layer_index].tensors["attention_scores"]
                        self.assertEqual(explorer.query_selector.count(), source.shape[-2])
                        explorer.query_selector.setCurrentIndex(source.shape[-2] - 1)
                        explorer.key_page.setValue(explorer.key_page.maximum())
                        for head in range(source.shape[-3]):
                            explorer.head_selector.setValue(head)
                            for kind, model in explorer.models.items():
                                array = layers[layer_index].tensors[f"attention_{kind}"]
                                self.assertIs(model.array, array)
                                row, column = model.rowCount() - 1, model.columnCount() - 1
                                coordinate = model.tensor_index(row, column)
                                self.assertEqual(coordinate, (0, head, source.shape[-2] - 1, source.shape[-1] - 1))
                                explorer.tables[kind].setCurrentIndex(model.index(row, column))
                                self.assertIn(repr(query_labels[-1]), explorer.selection_label.text())
                                self.assertIn(repr(key_labels[-1]), explorer.selection_label.text())
                                self.assertIn(exact_scalar(array[coordinate]), explorer.selection_label.text())
                                self.assertIn(phase, explorer.selection_label.text())
                                self.assertFalse(model.flags(model.index(row, column)) & Qt.ItemIsEditable)
                        dialog = explorer.open_full_tensor("weights")
                        full = dialog.findChild(TensorInspector)
                        self.assertIsNotNone(full)
                        self.assertIs(full.array, layers[layer_index].tensors["attention_weights"])
                        self.assertEqual(full.model.leading_indices, (0, source.shape[-3] - 1))
                        dialog.close()
                        explorer.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertEqual(before, capture_digest(capture))

    # ── The two pages ─────────────────────────────────────────────────────────

    def pages(self, capture):
        return (ReferenceView(capture, tokens=app.TOKENS),
                NotebookView(capture, tokens=app.TOKENS))

    def test_free_scrolling_tracks_the_section_at_the_top(self):
        """Scroll both pages by hand, which `go_to` never does.

        This is the test that was missing.  `go_to` sets `_scrolling` around its own
        scroll, so `_on_scrolled` returned immediately and no test ever ran its body --
        while a reader's very first wheel turn did.  The handler mapped the viewport into
        the page with `mapTo`, which walks the parent chain *upwards*; the page is a child
        of the viewport, so the walk ran off the top of the hierarchy and dereferenced
        null.  Viewing a saved run and scrolling killed the process every time.

        A recurrence crashes the interpreter rather than failing an assertion, so this
        test going bang *is* the signal -- there is no way to catch an access violation.
        """
        capture = next(iter(self.captures.values()))
        for page in self.pages(capture):
            with self.subTest(page=type(page).__name__):
                page.resize(1130, 820)
                page.show()                      # unshown, the scroll area has no range
                for key, _ in page.NAV:          # final positions need built bodies
                    page.go_to(key)
                self.application.processEvents()
                bar = page.scroll.verticalScrollBar()
                self.assertGreater(bar.maximum(), 0, 'the page does not scroll at all')

                # Put each section's top at the viewport top and ask what the page thinks
                # the reader is looking at.
                for key, _ in page.NAV:
                    top = page.section(key).y()
                    bar.setValue(top)
                    if bar.value() == top:       # reachable: not clamped at the bottom
                        self.assertEqual(page.current_stage(), key)

                # And a continuous sweep, the way a wheel delivers it.
                step = max(1, bar.maximum() // 60)
                for value in range(0, bar.maximum() + 1, step):
                    bar.setValue(value)
                bar.setValue(0)
                self.assertEqual(page.current_stage(), page.NAV[0][0])
            page.deleteLater()

    def test_every_section_of_both_pages_builds_without_mutating_the_capture(self):
        """Walk every section of both pages, over every saved run.

        The pages build their bodies lazily and keep every heavy widget behind a
        `Disclosure`, so a section nobody visited during development can carry a stale
        attribute name indefinitely. This visits all of them -- both phases, the first and
        last layer, the first and last head, every prompt position, every generated token
        -- clicks open every disclosure, paints every diagram, and then re-hashes the
        capture: reading a tensor must never rewrite it.
        """
        for run_id, capture in self.captures.items():
            before = capture_digest(capture)
            reference, notebook = self.pages(capture)

            for key, _ in reference.NAV:
                with self.subTest(run_id=run_id, page='reference', topic=key):
                    reference.go_to(key)
                    self.assertEqual(reference.current_stage(), key)
                    self.assertTrue(reference.section(key).is_built())
                    open_every_disclosure(reference)
                    paint_every_diagram(self, reference.section(key))

            for key, _ in notebook.NAV:
                with self.subTest(run_id=run_id, page='notebook', operation=key):
                    notebook.go_to(key)
                    self.assertEqual(notebook.current_stage(), key)
                    self.assertTrue(notebook.section(key).is_built())

            heads = {0, max(0, notebook.head_count() - 1)}
            layers = {min(notebook.registry.layer_indices),
                      max(notebook.registry.layer_indices)}
            for generated in (False, True):
                for layer_index in layers:
                    for head in heads:
                        with self.subTest(run_id=run_id, generated=generated,
                                          layer=layer_index, head=head):
                            notebook.select(generated=generated, layer_index=layer_index,
                                            head=head)
                            open_every_disclosure(notebook)
            notebook.select(generated=False, layer_index=min(layers), head=0)
            for position in range(len(capture.prompt_tokens)):
                notebook.select_position(position)
            open_every_disclosure(notebook)
            paint_every_diagram(self, notebook)

            response = notebook.findChild(ResponseTextView)
            self.assertIsNotNone(response)
            for step in range(len(capture.tokens)):
                response.activate_anchor(f't:{step}')
            open_every_disclosure(notebook)

            for widget in (reference, notebook):
                widget.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertEqual(before, capture_digest(capture))

    def test_the_notebook_keeps_the_readers_place_across_navigation(self):
        """Navigating must not cost the reader their selections, reveals or expansions.

        The old views destroyed and rebuilt the whole screen on every navigation and
        restored reader state from explicit dicts. The notebook builds each section once
        and scrolls, so state survives because the widgets are still there -- which is a
        stronger property, and `id(section)` is how it is checked.
        """
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            keys = [key for key, _ in notebook.NAV]
            for key in keys:
                notebook.go_to(key)
            identities = {key: id(notebook.section(key)) for key in keys}

            # Disclosures first: the optional checks sit behind one, collapsed.
            open_every_disclosure(notebook)
            for reveal in notebook.findChildren(StepReveal):
                reveal.reveal_all()
            for check in notebook.findChildren(PredictCheck):
                check.choose(-1)
            opened = [disclosure for disclosure in notebook.findChildren(Disclosure)
                      if disclosure.findChild(QPushButton).isChecked()]
            self.assertTrue(opened, 'the notebook should have something to expand')

            notebook.select(generated=True, layer_index=max(notebook.registry.layer_indices),
                            head=0, position=0)
            notebook.select(generated=False, layer_index=min(notebook.registry.layer_indices))
            chosen = dict(notebook.selection())
            for key in reversed(keys):
                notebook.go_to(key)

            with self.subTest(run_id=run_id):
                # Nothing was rebuilt: the same widgets are still on the page.
                self.assertEqual({key: id(notebook.section(key)) for key in keys},
                                 identities)
                self.assertEqual(notebook.selection(), chosen)
                for reveal in notebook.findChildren(StepReveal):
                    self.assertEqual(reveal.revealed(), reveal.step_count)
                for check in notebook.findChildren(PredictCheck):
                    self.assertEqual(check.chosen(), -1)
                for disclosure in opened:
                    self.assertTrue(disclosure.findChild(QPushButton).isChecked())
                self.assertEqual(notebook.current_stage(), keys[0])
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_navigation_moves_the_scrollbar_rather_than_resetting_it(self):
        """`go_to` moves the reader to a section; it is not a scroll-to-top."""
        capture = next(iter(self.captures.values()))
        notebook = NotebookView(capture, tokens=app.TOKENS)
        keys = [key for key, _ in notebook.NAV]
        notebook.resize(1200, 800)
        notebook.show()
        notebook.go_to(keys[0])
        top = notebook.scroll.verticalScrollBar().value()
        notebook.go_to(keys[-1])
        bottom = notebook.scroll.verticalScrollBar().value()
        self.assertGreater(bottom, top)
        self.assertEqual(notebook.current_stage(), keys[-1])
        notebook.hide()
        notebook.deleteLater()

    def test_token_span_selection_matches_the_recorded_candidates(self):
        """Clicking a token of the answer must show that step's own saved scores."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            notebook.go_to('continuation')
            response = notebook.findChild(ResponseTextView)
            self.assertIsNotNone(response)
            # The faithful branch is a property of the stored spellings, not an assumption.
            self.assertEqual(response.is_faithful(),
                             ''.join(capture.tokens).strip() == capture.response.strip())
            if response.is_faithful():
                self.assertEqual(response.anchor_count(), len(capture.tokens))
            trace = capture.generation_trace
            for step in range(len(capture.tokens)):
                with self.subTest(run_id=run_id, step=step):
                    response.activate_anchor(f't:{step}')
                    shown = section_text(notebook.section('continuation'))
                    self.assertIn(repr(capture.tokens[step]), shown)
                    self.assertIn(str(capture.token_ids[step]), shown)
                    if step >= len(trace):
                        self.assertIn(TELEMETRY_UNAVAILABLE, shown)
                        continue
                    decision = trace[step]
                    rows = candidate_rows(notebook.section('continuation'))
                    self.assertEqual([int(row[1]) for row in rows],
                                     [int(entry['id']) for entry in decision.top_candidates])
                    expected = EVIDENCE_KINDS[
                        EVIDENCE_OBSERVED if (step == 0 or decision.attested)
                        else EVIDENCE_UNATTESTED][0]
                    self.assertIn(expected, shown)
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_counts_on_screen_are_this_capture_s_counts(self):
        """Run 4 has 14 positions and 75 tokens, so a hardcoded 15 or 81 fails here."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            notebook.go_to('continuation')
            facts = learn_facts(capture)
            shown = section_text(notebook.section('continuation'))
            with self.subTest(run_id=run_id):
                self.assertEqual(facts['continuation_count'], len(capture.token_ids) - 1)
                self.assertEqual(facts['position_count'], len(capture.prompt_tokens))
                self.assertEqual(facts['layer_count'], len(capture.layers))
                self.assertEqual(facts['phase_count'], 2)
                self.assertIn(str(len(capture.token_ids) - 1), shown)
                self.assertIn(str(len(capture.layers)), shown)
                self.assertIn(facts['stop_reason_text'], shown)
                # The stopping marker is scored but never emitted, so it is in neither the
                # answer nor the count.
                recorded = capture.metadata.get('stop_token_id')
                if recorded is None:
                    self.assertIn('does not record which entry won', shown)
                else:
                    self.assertIn(f'{int(recorded):,}', shown)
                    self.assertNotIn(int(recorded), capture.token_ids)
            notebook.deleteLater()

    def test_every_colour_a_chip_can_carry_is_named_in_the_colour_key(self):
        """Colour is never the only carrier, and the key must cover what is on the page.

        The chips' `role` property and the diagrams' palette both come from COLORS; the
        key names those roles.  A chip tinted a hue the key does not explain would be a
        meaning carried by colour alone.
        """
        capture = next(iter(self.captures.values()))
        notebook = NotebookView(capture, tokens=app.TOKENS)
        for key, _ in notebook.NAV:          # bodies build on first arrival
            notebook.go_to(key)
        open_every_disclosure(notebook)
        chips = notebook.findChildren(VariableChip)
        self.assertTrue(chips, 'the page built no variable chips at all')
        named = set(notebook.legend.describe())
        roles_on_screen = {chip.property('role') for chip in chips} - {None, ''}
        self.assertTrue(roles_on_screen, 'no chip carried a role property')
        for role in roles_on_screen:
            self.assertIn(role, COLORS, f'chip role {role!r} has no colour')
            name = dict(ROLE_LEGEND).get(role)
            self.assertIsNotNone(name, f'chip role {role!r} is absent from the colour key')
            self.assertIn(name, named)
        notebook.deleteLater()

    def test_the_generated_phase_attention_is_rectangular(self):
        """One query against the cached prompt keys plus itself, never a square."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            notebook.select(generated=True)
            notebook.go_to('scores')
            scores = capture.generated_layers[notebook.registry.layer_index] \
                .tensors['attention_scores']
            queries, keys = int(scores.shape[-2]), int(scores.shape[-1])
            with self.subTest(run_id=run_id):
                self.assertEqual(queries, 1)
                self.assertEqual(keys, len(capture.prompt_tokens) + 1)
                masks = notebook.section('scores').findChildren(CausalMaskTriangle)
                self.assertTrue(masks)
                described = masks[0].describe()
                self.assertIn(f'1 query × {keys:,} keys', described)
                self.assertIn(f'{keys - queries:,} keys came from the cache', described)
                self.assertFalse(any('queries' in text for text in described))
            notebook.deleteLater()

    def test_an_operation_shows_only_the_evidence_tiers_it_declares(self):
        """A badge is a claim about where a number came from; it may not appear loosely."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            for operation in NOTEBOOK_OPERATIONS + [CONTINUATION]:
                with self.subTest(run_id=run_id, operation=operation.key):
                    notebook.go_to(operation.key)
                    section = notebook.section(operation.key)
                    open_every_disclosure(section)
                    for reveal in section.findChildren(StepReveal):
                        reveal.reveal_all()
                    allowed = {EVIDENCE_KINDS[tier][0] for tier in operation.tiers}
                    # Badges still name their tier in words, and the words must agree with
                    # the property: a badge cannot say one tier and claim another.
                    for badge in section.findChildren(EvidenceBadge):
                        name = badge.text().split('  ·  ')[0]
                        self.assertIn(name, allowed,
                                      f'{operation.key} showed an undeclared tier')
                        self.assertEqual(EVIDENCE_KINDS[badge.property('evidenceTier')][0], name)
                    for widget, tier in evidence_claims(section):
                        self.assertIn(tier, operation.tiers,
                                      f'{operation.key} marked an undeclared tier '
                                      f'on a {type(widget).__name__}')
            notebook.deleteLater()

    def test_the_unattested_tier_appears_only_where_it_belongs(self):
        """Two places only: a later generation step's scores, and the stopping score."""
        unattested = EVIDENCE_KINDS[EVIDENCE_UNATTESTED][0]
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            for key, _ in notebook.NAV:
                notebook.go_to(key)
                open_every_disclosure(notebook.section(key))
                found = [badge for badge in notebook.section(key).findChildren(EvidenceBadge)
                         if badge.text().startswith(unattested)]
                found += [widget for widget, tier in evidence_claims(notebook.section(key))
                          if tier == EVIDENCE_UNATTESTED and widget not in found]
                with self.subTest(run_id=run_id, operation=key):
                    if key != 'continuation':
                        self.assertEqual(found, [], f'{key} claimed an unattested value')
            notebook.deleteLater()

    def test_observed_values_carry_an_edge_not_a_badge(self):
        """Only DERIVED and UNATTESTED keep words inline; the key to all four is in the
        panel.  An observed number is still marked -- by its box's edge and property."""
        inline = {EVIDENCE_KINDS[tier][0] for tier in EVIDENCE_INLINE_TIERS}
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            for key, _ in notebook.NAV:
                notebook.go_to(key)
                section = notebook.section(key)
                open_every_disclosure(section)
                for reveal in section.findChildren(StepReveal):
                    reveal.reveal_all()
                with self.subTest(run_id=run_id, section=key):
                    for badge in section.findChildren(EvidenceBadge):
                        if notebook.section('continuation').isAncestorOf(badge) and \
                                badge.parentWidget().objectName() == 'evidenceLegend':
                            continue                    # the key itself, in the limits
                        self.assertIn(badge.text().split('  ·  ')[0], inline, key)
                    for box in section.findChildren(QFrame):
                        if box.objectName() == 'valueBox':
                            self.assertTrue(box.property('evidenceTier'), key)
                            self.assertIn(EVIDENCE_KINDS[box.property('evidenceTier')][0],
                                          box.toolTip())
            legend = notebook.panel.evidence_legend
            with self.subTest(run_id=run_id, part='legend'):
                spoken = ' '.join(legend.describe())
                for name, _description in EVIDENCE_KINDS.values():
                    self.assertIn(name, spoken)
                self.assertTrue(notebook.panel.isAncestorOf(legend))
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_template_tokens_are_explained_and_faded_only_when_aligned(self):
        """Faded chips and the count sentence need the alignment; without it, neither."""
        for run_id, capture in self.captures.items():
            bands = prompt_token_bands(capture.prompt, capture.prompt_tokens)
            template = {index for start, end, name in (bands or []) if name == TEMPLATE_BAND
                        for index in range(start, end)}
            facts = learn_facts(capture)
            notebook = NotebookView(capture, tokens=app.TOKENS)
            section = notebook.section('input')
            with self.subTest(run_id=run_id):
                self.assertIsNotNone(bands, 'the fixture runs all align')
                chips = [chip for chip in section.findChildren(QPushButton)
                         if chip.objectName() == 'tokenChip']
                self.assertEqual(len(chips), len(capture.prompt_tokens))
                self.assertEqual({index for index, chip in enumerate(chips)
                                  if chip.property('template')}, template)
                self.assertIn(NOTEBOOK_TEXT['template_tokens_why'].format(**facts),
                              section_text(section))
            notebook.deleteLater()
        # The unaligned branch, on a copy whose prompt the stored spellings do not contain.
        capture = copy.copy(next(iter(self.captures.values())))
        capture.prompt = 'text the stored spellings do not contain'
        notebook = NotebookView(capture, tokens=app.TOKENS)
        section = notebook.section('input')
        shown = section_text(section)
        self.assertIn(PROMPT_SPAN_UNKNOWN, shown)
        self.assertNotIn('tokens you did not type', shown)
        self.assertFalse(any(chip.property('template') for chip in section.findChildren(QPushButton)
                             if chip.objectName() == 'tokenChip'))
        notebook.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_attention_sink_note_follows_its_stated_rule(self):
        """The note appears exactly when the weight row on screen gives position 0 at least
        the threshold, says what the rule is, and is never the model 'reading' a token."""
        threshold = f'{ATTENTION_SINK_THRESHOLD:.0%}'
        both = {'above': 0, 'below': 0}
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            position = notebook.registry.position
            cases = {}
            for layer in sorted(capture.layers):
                weights = np.asarray(capture.layers[layer].tensors['attention_weights'])
                for head in range(weights.shape[1]):
                    side = ('above' if float(weights[0, head, position, 0])
                            >= ATTENTION_SINK_THRESHOLD else 'below')
                    cases.setdefault(side, (layer, head))
            for side, (layer, head) in cases.items():
                both[side] += 1
                notebook.select(layer_index=layer, head=head)
                notebook.go_to('scores')
                notes = [note for note in notebook.section('scores').findChildren(QLabel)
                         if note.objectName() == 'sinkNote']
                with self.subTest(run_id=run_id, side=side, layer=layer, head=head):
                    if side == 'below':
                        self.assertEqual(notes, [])
                        continue
                    self.assertEqual(len(notes), 1)
                    text = notes[0].text()
                    self.assertIn(f'at least {threshold}', text)
                    self.assertIn(repr(readable_spelling(capture.prompt_tokens[0])), text)
                    self.assertIn('not evidence that the model is reading', text)
                    self.assertIn(WEIGHTS_NOT_PROBABILITIES, text)
            # Op 11's form counts the layers whose head-mean row trips the same rule.
            notebook.select(layer_index=min(capture.layers), head=0)
            notebook.go_to('layers')
            expected = sum(
                1 for layer in capture.layers.values()
                if np.asarray(layer.tensors['attention_weights'], dtype=np.float64)
                [0, :, position, 0].mean() >= ATTENTION_SINK_THRESHOLD)
            notes = [note.text() for note in notebook.section('layers').findChildren(QLabel)
                     if note.objectName() == 'sinkNote']
            with self.subTest(run_id=run_id, part='layers'):
                if expected:
                    self.assertEqual(len(notes), 1)
                    self.assertTrue(notes[0].startswith(f'In {expected} of {len(capture.layers)} '))
                else:
                    self.assertEqual(notes, [])
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertTrue(both['above'] and both['below'],
                        'the fixture must exercise both sides of the threshold')

    def test_every_attention_percentage_sits_beside_the_not_probabilities_line(self):
        """approx_share may appear outside operation 7 only with its caveat beside it."""
        for capture in self.captures.values():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            for key, _ in notebook.NAV:
                notebook.go_to(key)
                section = notebook.section(key)
                open_every_disclosure(section)
                for reveal in section.findChildren(StepReveal):
                    reveal.reveal_all()
                readable = re.sub(r'<[^>]*>', '', section_text(section))
                with self.subTest(section=key):
                    if '%' in readable:
                        self.assertIn('not next-token probabilities', readable)
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            break                         # one run: the copy is the same for all of them

    def test_legacy_runs_render_without_inventing_anything(self):
        """Selected by capability, not by id, so a thinner database fails loudly."""
        no_logits = [capture for capture in self.captures.values()
                     if capture.logits is None]
        no_trace = [capture for capture in self.captures.values()
                    if not capture.generation_trace and capture.logits is not None]
        self.assertTrue(no_logits, 'the database must contain a pre-logits run to test')
        self.assertTrue(no_trace, 'the database must contain a pre-telemetry run to test')

        for capture in no_logits:
            notebook = NotebookView(capture, tokens=app.TOKENS)
            notebook.go_to('decide')
            open_every_disclosure(notebook.section('decide'))
            shown = section_text(notebook.section('decide'))
            self.assertIn(LOGITS_UNAVAILABLE, shown)
            self.assertEqual(candidate_rows(notebook.section('decide')), [])
            # The selected token itself is still known: the emitted text was saved.
            self.assertIn(repr(capture.tokens[0]), shown)
            notebook.deleteLater()

        for capture in no_logits + no_trace:
            notebook = NotebookView(capture, tokens=app.TOKENS)
            notebook.go_to('continuation')
            response = notebook.findChild(ResponseTextView)
            response.activate_anchor('t:0')
            self.assertIn(TELEMETRY_UNAVAILABLE,
                          section_text(notebook.section('continuation')))
            notebook.deleteLater()

    def test_no_matplotlib_canvas_is_built_until_the_reader_asks(self):
        """36 layers must not cost 70 canvases up front, which is why bodies are lazy."""
        capture = next(iter(self.captures.values()))
        for factory in (ReferenceView, NotebookView):
            page = factory(capture, tokens=app.TOKENS)
            with self.subTest(page=factory.__name__):
                self.assertEqual(page.findChildren(FigureCanvas), [])
                for key, _ in page.NAV:
                    page.go_to(key)
                self.assertLessEqual(len(page.findChildren(FigureCanvas)), 3)
            page.deleteLater()

    def test_a_section_stays_a_readable_height_at_laptop_width(self):
        """A long page is only usable if its sections are not pages themselves.

        Measured at the width the design has to work at. A size hint for word-wrapped text
        means nothing until the widget has a width, so the page is given one before
        anything is read off it.
        """
        capture = next(iter(self.captures.values()))
        notebook = NotebookView(capture, tokens=app.TOKENS)
        notebook.resize(1130, 740)              # the page, inside a 1366-wide window
        for key, _ in notebook.NAV:
            notebook.go_to(key)
        notebook.layout().activate()
        # Measured under the offscreen platform at this width, the twelve operations run
        # from about 1,250 to about 2,520 pixels.  Offscreen fonts render roughly twice the
        # size of the real ones: with Segoe UI 10 the same sections measure 525 to 1,271,
        # and the continuation 1,545 -- one to two screens of a 1366x768 laptop each, the
        # readable chunk Next moves between.  Every operation now carries its in/does/out
        # line and its "Next" line by design; the optional checks, the feed-forward
        # interior and each step's variable list sit behind a `Disclosure`.  The bounds are
        # set just above the offscreen measurements, so a change that turns a section into
        # a page of its own still fails here.
        heights = {}
        for section in notebook.sections():
            heights[section.key] = section.sizeHint().height()
            with self.subTest(section=section.key):
                # The continuation holds the whole answer plus three tables, which is the
                # one place where showing everything at once is the point.
                limit = 3400 if section.key == 'continuation' else 2600
                self.assertLessEqual(heights[section.key], limit)
        self.assertLessEqual(sum(heights.values()), 24000, heights)
        notebook.deleteLater()

    def test_both_themes_build_both_pages(self):
        capture = next(iter(self.captures.values()))
        for name, tokens in (('dark', app._TOKENS_DARK), ('light', app._TOKENS_LIGHT)):
            recap = ComputationRecap(capture, run_id=1, tokens=tokens)
            with self.subTest(theme=name):
                for key, _, _, _ in MODES:
                    recap._show_mode(key)
                    for stage_key, _ in recap.view.NAV:
                        recap._nav_to(stage_key)
                    paint_every_diagram(self, recap.view)
            recap.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_reader_can_reach_the_pages_from_the_keyboard(self):
        capture = next(iter(self.captures.values()))
        recap = ComputationRecap(capture, run_id=1, tokens=app.TOKENS)
        self.assertNotEqual(recap.next_stage.focusPolicy(), Qt.NoFocus)
        self.assertNotEqual(recap.previous_stage.focusPolicy(), Qt.NoFocus)
        recap._show_mode('run')
        notebook = recap.view
        for section in notebook.sections():
            self.assertEqual(section.focusPolicy(), Qt.StrongFocus)
            self.assertTrue(section.accessibleName())
        for chip in notebook.findChildren(VariableChip):
            self.assertEqual(chip.focusPolicy(), Qt.StrongFocus)
            self.assertTrue(chip.accessibleName())
        # The pickers are contextual now: each appears in the operation that explains it.
        for key in ('prepare', 'layers', 'continuation'):
            notebook.go_to(key)
        for kind in ('head', 'layer', 'phase'):
            built = notebook.pickers(kind)
            self.assertTrue(built, f'no {kind} control was built')
            for widget in built:
                self.assertTrue(widget.accessibleName(), kind)
        chip = notebook.toolbar_widget()
        self.assertNotEqual(chip.focusPolicy(), Qt.NoFocus)
        self.assertTrue(chip.accessibleName())
        self.assertNotEqual(recap.contents.focusPolicy(), Qt.NoFocus)
        # Every emitted token must be reachable without a mouse, because the response
        # anchors only answer linkActivated.
        notebook.go_to('continuation')
        response = notebook.findChild(ResponseTextView)
        steppers = [button for button in notebook.section('continuation')
                    .findChildren(QPushButton)
                    if button.text() in ('← Previous token', 'Next token →')]
        self.assertEqual(len(steppers), 2)
        following = next(button for button in steppers if 'Next' in button.text())
        following.click()
        self.assertEqual(response.current(), 1)
        recap.deleteLater()

    def test_the_shell_reaches_both_pages_and_every_section(self):
        """The shell's navigation must stay in step with whatever sections a page declares."""
        run_id, capture = next(iter(self.captures.items()))
        before = capture_digest(capture)
        recap = ComputationRecap(capture, run_id=run_id, tokens=app.TOKENS)
        self.assertEqual([key for key, _, _, _ in MODES], ['run', 'understand'])
        self.assertEqual(recap._mode, DEFAULT_MODE)
        chapter_titles = {chapter.title for chapter in CHAPTERS}
        for key, name, _, _ in MODES:
            recap._show_mode(key)
            self.assertTrue(recap._mode_buttons[key].isChecked())
            self.assertEqual(set(recap.contents.actions_by_key),
                             {k for k, _ in recap.view.NAV})
            for position, (stage_key, _stage_name) in enumerate(recap.view.NAV):
                recap._nav_to(stage_key)
                self.assertEqual(recap.view.current_stage(), stage_key)
                self.assertTrue(recap.contents.actions_by_key[stage_key].isChecked())
                if key == 'run':
                    # Every section reports which of the four chapters it belongs to.
                    self.assertIn(recap.view.chapter_of(stage_key), chapter_titles)
                    self.assertIn(recap.view.chapter_of(stage_key), recap.progress.text())
                else:
                    self.assertIn(f'Topic {position + 1} of {len(recap.view.NAV)}',
                                  recap.progress.text())
            # Previous / Next walk the same list the contents menu does.
            recap._nav_to(recap.view.NAV[0][0])
            recap._advance(1)
            self.assertEqual(recap.view.current_stage(), recap.view.NAV[1][0])
            recap._advance(-1)
            self.assertEqual(recap.view.current_stage(), recap.view.NAV[0][0])
        recap.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertEqual(before, capture_digest(capture))

    def test_switching_modes_retires_the_old_controls_instead_of_piling_them_up(self):
        """A control from the page you left must not survive, visible and clickable.

        `takeAt` + `deleteLater` looked like it emptied the navigator and did not: the
        widget kept its parent, its geometry and its visibility until the event loop got
        round to the deferred delete, so the chips accumulated 13 -> 25 -> 38 across two
        mode switches. Clicking one called `go_to` with a key the live page does not own,
        which raises inside a slot, which makes PyQt abort the process with no traceback
        at all -- the user sees "Segmentation fault".
        """
        run_id, capture = next(iter(self.captures.items()))
        recap = ComputationRecap(capture, run_id=run_id, tokens=app.TOKENS)

        for _ in range(3):
            for mode, _name, _purpose, _cls in MODES:
                recap._show_mode(mode)
                live = {key for key, _ in recap.view.NAV}
                # No deferred-delete pump: the point is that they are gone *now*.  The
                # contents menu is the one list of sections; it must hold exactly the live
                # page's, not accumulate the other page's across switches.
                actions = recap.contents.section_actions()
                self.assertEqual(len(actions), len(live))
                self.assertEqual(set(recap.contents.actions_by_key), live)
                # And only the live page's toolbar control is shown.
                for key, toolbar in recap._toolbars.items():
                    self.assertEqual(not toolbar.isHidden(), key == mode)

        # And a navigation slot handed a key from the other page does nothing, rather
        # than raising out of a slot.
        recap._show_mode('run')
        reference_keys = {key for key, _ in MODE_INDEX['understand'][2].NAV}
        run_keys = {key for key, _ in MODE_INDEX['run'][2].NAV}
        foreign = next(iter(reference_keys - run_keys))
        before = recap.view.current_stage()
        recap._nav_to(foreign)
        self.assertEqual(recap.view.current_stage(), before)
        recap.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_pages_link_to_each_other(self):
        run_id, capture = next(iter(self.captures.items()))
        recap = ComputationRecap(capture, run_id=run_id, tokens=app.TOKENS)
        recap._jump_to('understand', 'qkv')
        self.assertEqual(recap._mode, 'understand')
        self.assertEqual(recap.view.current_stage(), 'qkv')
        recap._jump_to('run', 'scores')
        self.assertEqual(recap._mode, 'run')
        self.assertEqual(recap.view.current_stage(), 'scores')
        # Every operation's "explain this" target and every topic's "follow this" target
        # must exist on the other page, or a button is a dead end.
        for operation in NOTEBOOK_OPERATIONS:
            if operation.topic:
                recap._jump_to('understand', operation.topic)
                self.assertEqual(recap.view.current_stage(), operation.topic)
        recap.deleteLater()

    def test_every_captured_array_stays_reachable_without_a_mode_of_its_own(self):
        """Raw access is a secondary control now, so it must still reach everything."""
        for run_id, capture in self.captures.items():
            before = capture_digest(capture)
            recap = ComputationRecap(capture, run_id=run_id, tokens=app.TOKENS)
            recap._show_mode('run')
            recap.view.jump_requested.emit('raw', '')
            dialog = recap._raw_dialog
            self.assertIsNotNone(dialog, 'the ribbon must open the raw dialog')
            raw = dialog.view
            seen = 0
            for phase in range(raw.phase_picker.count()):
                raw.phase_picker.setCurrentIndex(phase)
                for layer in range(raw.layer_picker.count()):
                    raw.layer_picker.setCurrentIndex(layer)
                    self.assertGreater(raw.tensor_picker.count(), 0)
                    for tensor in range(raw.tensor_picker.count()):
                        raw.tensor_picker.setCurrentIndex(tensor)
                        inspector = raw.findChild(TensorInspector)
                        self.assertIsNotNone(inspector)
                        seen += 1
            with self.subTest(run_id=run_id):
                self.assertGreater(seen, 0)
            recap.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertEqual(before, capture_digest(capture))


    # ── The presentation's own promises ───────────────────────────────────────

    def test_initial_focus_is_a_meaningful_prompt_token(self):
        """The page follows the last typed word, never a template marker or punctuation."""
        for run_id, capture in self.captures.items():
            bands = prompt_token_bands(capture.prompt, capture.prompt_tokens)
            typed = [index for start, end, name in (bands or []) if name == TYPED_BAND
                     for index in range(start, end)]
            words = [index for index in typed
                     if any(ch.isalnum() for ch in readable_spelling(capture.prompt_tokens[index]))]
            expected = (words[-1] if words else typed[-1] if typed
                        else len(capture.prompt_tokens) - 1)
            notebook = NotebookView(capture, tokens=app.TOKENS)
            with self.subTest(run_id=run_id):
                self.assertEqual(notebook.registry.position, expected)
                self.assertIn(f'at {expected}', notebook.toolbar_widget().text())
            notebook.hide()
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_decision_shifts_to_the_final_prompt_position(self):
        """The followed position must never look as if it chose the token."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            facts = learn_facts(capture)
            final = len(capture.prompt_tokens) - 1
            followed = notebook.registry.position
            notebook.go_to('decide')
            section = notebook.section('decide')
            shown = section_text(section)
            with self.subTest(run_id=run_id):
                self.assertNotEqual(followed, final, 'the fixture runs follow a typed word')
                self.assertIn(f'final prompt position, {final}', shown)
                self.assertIn(f'followed position {followed}', shown)
                # The page offers the jump; it does not make it.
                self.assertEqual(notebook.registry.position, followed)
                projection = section.findChildren(diagrams.VocabularyProjection)
                self.assertTrue(projection)
                spoken = ' '.join(projection[0].describe())
                self.assertIn(f'only position {final} chose', spoken)
                self.assertIn(f'you followed position {followed}', spoken)
                if capture.logits is not None:
                    self.assertIn(f'= {int(np.argmax(capture.logits))}', shown)
                    self.assertIn(repr(capture.tokens[0]), shown)
                    self.assertNotIn('%', shown)
                jump = [button for button in section.findChildren(QPushButton)
                        if button.text() == NOTEBOOK_TEXT['follow_final'].format(**facts)]
                self.assertEqual(len(jump), 1)
                jump[0].click()
                self.assertEqual(notebook.registry.position, final)
                self.assertIn(NOTEBOOK_TEXT['focus_is_final'].format(**facts),
                              section_text(section))
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_mix_products_add_up_to_the_shown_total(self):
        """Operation 8's visible products plus the remainder are the whole sum."""
        for run_id, capture in self.captures.items():
            before = capture_digest(capture)
            notebook = NotebookView(capture, tokens=app.TOKENS)
            context = notebook.context_builder
            for generated in (False, True):
                for head in {0, notebook.head_count() - 1}:
                    notebook.select(generated=generated, head=head)
                    example = operations.mix_example(context)
                    with self.subTest(run_id=run_id, generated=generated, head=head):
                        self.assertIsNotNone(example)
                        layers = capture.generated_layers if generated else capture.layers
                        tensors = layers[notebook.registry.layer_index].tensors
                        query = min(notebook.registry.position,
                                    tensors['attention_weights'].shape[-2] - 1)
                        weights = tensors['attention_weights'][0, head, query].astype(np.float64)
                        values = tensors['v_attended'][0, head, :, 0].astype(np.float64)
                        expected_keys = sorted(np.argsort(-weights, kind='stable')[
                            :operations.MIX_KEYS].tolist())
                        self.assertEqual(example['shown'], expected_keys)
                        products = example['products']
                        shown = float(products[example['shown']].sum())
                        rest = float(products[example['rest']].sum()) if example['rest'] else 0.0
                        self.assertLessEqual(abs(shown + rest - example['total']), 1e-12)
                        self.assertLessEqual(abs(example['total'] - float(weights @ values)),
                                             1e-12)
            notebook.hide()
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertEqual(before, capture_digest(capture))

    def test_the_per_layer_view_reads_captured_attention(self):
        """One row per layer, each from that layer's own captured arrays."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            rows = operations.layer_summary_rows(notebook.context_builder)
            with self.subTest(run_id=run_id):
                self.assertEqual([row['layer'] for row in rows], sorted(capture.layers))
                position = notebook.registry.position
                for row in rows:
                    tensors = capture.layers[row['layer']].tensors
                    mean = tensors['attention_weights'][0, :, position, :].astype(
                        np.float64).mean(axis=0)
                    self.assertEqual(row['key'], int(np.argmax(mean)))
                    self.assertLessEqual(row['key'], position)      # causal: never later
                    output = tensors['layer_output'][0, position].astype(np.float64)
                    self.assertEqual(row['feature'], int(np.argmax(np.abs(output))))
            notebook.go_to('layers')
            # The thirty-six identical shape tables are gone: one summary table, one shape
            # table behind a single disclosure.
            titles = [d.title for d in notebook.section('layers').findChildren(Disclosure)]
            self.assertFalse(any(title.startswith('Layer ') for title in titles), titles)
            notebook.hide()
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_rounded_values_carry_their_exact_value(self):
        """Rounding is for reading: the exact stored digits are always one hover away."""
        capture = next(iter(self.captures.values()))
        notebook = NotebookView(capture, tokens=app.TOKENS)
        for key, _ in notebook.NAV:
            notebook.go_to(key)
        for reveal in notebook.findChildren(StepReveal):
            reveal.reveal_all()
        boxes = [frame for frame in notebook.findChildren(QLabel)
                 if frame.objectName() == 'valueBoxNumber']
        self.assertTrue(boxes)
        for box in boxes:
            with self.subTest(value=box.text()):
                self.assertTrue(box.toolTip().startswith('Exact stored value: '))
                if box.text().startswith('≈'):
                    self.assertNotEqual(box.toolTip().split(': ', 1)[1], box.text())
        # Logits are scores, never percentages.  Markup is not text a reader sees, so the
        # response's rich-text styling ("line-height: 160%") is stripped first.
        for key in ('decide', 'continuation'):
            readable = re.sub(r'<[^>]*>', '', section_text(notebook.section(key)))
            self.assertNotIn('%', readable, key)
        notebook.hide()
        notebook.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_controls_appear_where_they_explain_something(self):
        """Head, layer and phase pickers live in the operations that need them."""
        capture = next(iter(self.captures.values()))
        recap = ComputationRecap(capture, run_id=1, tokens=app.TOKENS)
        notebook = recap.view
        for key, _ in notebook.NAV:
            notebook.go_to(key)
        homes = {'head': {'prepare', 'scores', 'mix'}, 'layer': {'layers'},
                 'phase': {'continuation'}}
        for kind, allowed in homes.items():
            for widget in notebook.pickers(kind):
                owner = next(section.key for section in notebook.sections()
                             if section.isAncestorOf(widget))
                with self.subTest(kind=kind, section=owner):
                    self.assertIn(owner, allowed)
                    self.assertFalse(recap.reading_bar.isAncestorOf(widget))
        # Several copies of one picker agree, because they all follow the registry.
        notebook.select(head=3, layer_index=max(notebook.registry.layer_indices))
        for picker in notebook.pickers('head'):
            self.assertEqual(picker.combo.currentIndex(), 3)
        for picker in notebook.pickers('layer'):
            self.assertEqual(picker.combo.currentData(), notebook.registry.layer_index)
        notebook.pickers('phase')[0].click()
        self.assertTrue(notebook.registry.generated)
        self.assertIn('generated-token pass', notebook.toolbar_widget().text())
        recap.hide()
        recap.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_timeline_marks_what_each_pass_kept(self):
        """Layer tensors for two passes, a full vector for one, candidates where recorded."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            notebook.go_to('continuation')
            timelines = notebook.section('continuation').findChildren(PassTimeline)
            self.assertEqual(len(timelines), 1)
            timeline = timelines[0]
            stop_known = capture.metadata.get('stop_token_id') is not None
            trace = capture.generation_trace
            with self.subTest(run_id=run_id):
                self.assertEqual(timeline.count(), len(capture.tokens) + int(stop_known))
                for index in range(timeline.count()):
                    row = timeline.row(index)
                    self.assertEqual(row['layers'], index <= 1)
                    if index >= len(capture.tokens):
                        self.assertIn('stopping', row['title'])
                        self.assertEqual(row['tier'], EVIDENCE_UNATTESTED)
                        continue
                    self.assertIn(repr(capture.tokens[index]), row['title'])
                    if index == 0 and capture.logits is not None:
                        self.assertIn('full', row['capabilities'])
                    elif index < len(trace):
                        self.assertIn('leading candidate', row['capabilities'])
                    else:
                        self.assertIn('not recorded', row['capabilities'])
                        self.assertEqual(row['tier'], EVIDENCE_CONCEPTUAL)
                # The two passes that kept layer tensors are always on screen.
                self.assertTrue(timeline.is_row_shown(0) and timeline.is_row_shown(1))
                # Selecting in either place selects in both.
                response = notebook.findChild(ResponseTextView)
                last = len(capture.tokens) - 1
                response.select(last)
                self.assertEqual(timeline.current(), last)
                timeline._clicked(1)
                self.assertEqual(response.current(), 1)
            notebook.hide()
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_shell_fits_a_laptop_screen(self):
        """Two bands of chrome and no sideways scrolling at 1366 x 768."""
        capture = next(iter(self.captures.values()))
        recap = ComputationRecap(capture, run_id=1, tokens=app.TOKENS)
        recap.resize(1366, 768)
        recap.show()
        self.application.processEvents()
        # Measured in lines and characters of this platform's font, not raw pixels: the
        # offscreen platform substitutes Helvetica 12 (17 px per average character) for
        # Segoe UI 10 (7 px at 96 dpi), so a pixel bound would test the font, not the
        # layout.  A 1366-pixel window holds about 195 Segoe UI characters across.
        metrics = recap.fontMetrics()
        chrome_lines = (recap.banner.height() + recap.reading_bar.height()) / metrics.lineSpacing()
        self.assertLessEqual(chrome_lines, 6.5, chrome_lines)
        across = recap.minimumSizeHint().width() / max(1, metrics.averageCharWidth())
        self.assertLessEqual(across, 1366 / 7, across)
        for mode, _name, _purpose, _cls in MODES:
            recap._show_mode(mode)
            for key, _ in recap.view.NAV:
                recap._nav_to(key)
            self.application.processEvents()
            with self.subTest(mode=mode):
                self.assertTrue(recap.next_stage.isVisible())
                self.assertTrue(recap.contents.isVisible())
        recap.hide()
        recap.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_variable_panel_opens_on_demand_and_keeps_the_readers_place(self):
        """No permanent empty panel; choosing a variable opens it, closing hides it, and a
        visit to the reference page and back loses nothing."""
        capture = next(iter(self.captures.values()))
        recap = ComputationRecap(capture, run_id=1, tokens=app.TOKENS)
        notebook = recap.view
        self.assertFalse(notebook.side_panel_open())
        notebook.go_to('norm')
        chip = next(chip for chip in notebook.section('norm').findChildren(VariableChip)
                    if chip.key() == 'normalized_input')
        chip.click()
        self.assertTrue(notebook.side_panel_open())
        self.assertEqual(notebook.panel.current_symbol(), 'normalized_input')
        notebook.select(head=2)
        chosen = dict(notebook.selection())
        following = notebook.toolbar_widget().text()
        recap._jump_to('understand', 'qkv')
        self.assertEqual(recap.view.current_stage(), 'qkv')
        recap._jump_to('run', 'norm')
        self.assertIs(recap.view, notebook)
        self.assertTrue(notebook.side_panel_open())
        self.assertEqual(notebook.panel.current_symbol(), 'normalized_input')
        self.assertEqual(notebook.selection(), chosen)
        self.assertEqual(notebook.toolbar_widget().text(), following)
        notebook.panel.close_button.click()
        self.assertFalse(notebook.side_panel_open())
        self.assertEqual(notebook.current_stage(), 'norm')
        recap.hide()
        recap.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_the_page_opens_on_its_destination(self):
        """Prompt, the start of the real answer, and the token being explained, first."""
        for run_id, capture in self.captures.items():
            notebook = NotebookView(capture, tokens=app.TOKENS)
            first = notebook.section('input')
            preface = first.preface
            with self.subTest(run_id=run_id):
                self.assertIsNotNone(preface)
                shown = section_text(preface)
                self.assertIn(capture.prompt, shown)
                self.assertIn(repr(capture.tokens[0]), shown)
                self.assertIn(learn_facts(capture)['response_excerpt'].split(' ')[1], shown)
                self.assertIs(first.anchor(), preface)
            notebook.hide()
            notebook.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class ModuleLayeringTests(unittest.TestCase):
    """The import graph is a declared, one-directional DAG, not a convention.

    CLAUDE.md says the layering is the point, and the one thing holding it up today is a
    comment plus `MiniHeatmap`'s deferred `from TensorScope import display_matrix`. A
    module-level import of `TensorScope` from any recap module runs while `TensorScope` is
    still half-initialised, which is a confusing failure a long way from its cause. This
    reads the real imports instead of trusting the comment.
    """

    # Who each of our modules is allowed to import at module scope. A module absent from
    # the filesystem is skipped, so this stays green while the redesign lands in stages.
    ALLOWED = {
        "tensor_widgets": set(),
        "tensorscope_content": set(),
        "tensorscope_common": {"tensor_widgets", "tensorscope_content"},
        "tensorscope_diagrams": {"tensorscope_content", "tensorscope_common"},
        "tensorscope_variables": {"tensorscope_content", "tensorscope_common", "tensor_widgets"},
        "tensorscope_operations": {"tensorscope_content", "tensorscope_common",
                                   "tensorscope_diagrams", "tensorscope_variables",
                                   "tensor_widgets"},
        "tensorscope_notebook": {"tensorscope_content", "tensorscope_common",
                                 "tensorscope_diagrams", "tensorscope_variables",
                                 "tensorscope_operations", "tensor_widgets"},
        "tensorscope_reference": {"tensorscope_content", "tensorscope_common",
                                  "tensorscope_diagrams", "tensorscope_variables"},
        "tensorscope_views": {"tensorscope_content", "tensorscope_common",
                              "tensorscope_variables", "tensorscope_notebook",
                              "tensorscope_reference"},
        "tensorscope_ui": {"tensorscope_common", "tensorscope_views"},
    }

    def our_modules(self):
        here = Path(__file__).parent
        for name in self.ALLOWED:
            path = here / f"{name}.py"
            if path.is_file():
                yield name, path

    def module_level_imports(self, path):
        """Every name imported at module scope. Imports inside a function are not ours."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
        return found

    def test_no_recap_module_imports_the_app_at_module_scope(self):
        for name, path in self.our_modules():
            with self.subTest(module=name):
                self.assertNotIn("TensorScope", self.module_level_imports(path))

    def test_every_edge_is_declared(self):
        known = set(self.ALLOWED)
        for name, path in self.our_modules():
            edges = self.module_level_imports(path) & known
            with self.subTest(module=name):
                self.assertLessEqual(edges, self.ALLOWED[name],
                                     f"{name} imports {sorted(edges - self.ALLOWED[name])}, "
                                     "which the declared layering does not allow")

    def bound_names(self, tree):
        """Every name a module binds: imports, definitions, parameters, assignments."""
        names = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update((alias.asname or alias.name).split(".")[0]
                             for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                arguments = getattr(node, "args", None)
                if arguments is not None:
                    for group in (arguments.args, arguments.posonlyargs,
                                  arguments.kwonlyargs):
                        names.update(argument.arg for argument in group)
                    for extra in (arguments.vararg, arguments.kwarg):
                        if extra is not None:
                            names.add(extra.arg)
            elif isinstance(node, ast.Lambda):
                for group in (node.args.args, node.args.posonlyargs,
                              node.args.kwonlyargs):
                    names.update(argument.arg for argument in group)
                for extra in (node.args.vararg, node.args.kwarg):
                    if extra is not None:
                        names.add(extra.arg)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                names.update(node.names)
            elif isinstance(node, ast.comprehension):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        return names

    def test_no_module_uses_a_name_it_never_defines(self):
        """Catch a name that is used but not imported, without running the code path.

        Importing a module proves nothing about a `NameError` inside a method nobody
        called -- which is exactly how trimming an unused-looking import from `RawView`
        broke the raw-tensor dialog while every import still succeeded. Qt widgets are
        built lazily all over this codebase, so a check that does not need to reach a line
        to judge it is worth having.
        """
        for name, path in self.our_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            known = self.bound_names(tree)
            missing = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in known:
                        missing.setdefault(node.id, node.lineno)
            with self.subTest(module=name):
                self.assertEqual(missing, {},
                                 f"{name}.py uses names it never defines: "
                                 + ", ".join(f"{key} (line {line})"
                                             for key, line in sorted(missing.items())))

    def test_diagrams_never_reach_a_captured_array(self):
        """A diagram draws a structure. It must not be able to read the capture."""
        path = Path(__file__).parent / "tensorscope_diagrams.py"
        if not path.is_file():
            self.skipTest("diagrams module not present yet")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Any import at all, at any depth -- the deferred-import escape hatch that
        # MiniHeatmap legitimately needs is not available to a diagram.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([alias.name for alias in node.names]
                         if isinstance(node, ast.Import) else [node.module or ""])
                for candidate in names:
                    self.assertNotEqual(candidate.split(".")[0], "TensorScope",
                                        "a diagram reached for the app module")


class DiagramTests(unittest.TestCase):
    """Every diagram paints, in both themes, under the offscreen platform.

    `paintEvent` output cannot be read back, so two things stand in for looking at it:
    `grab()` forces a real paint (an exception in `_draw` surfaces here rather than in
    front of a reader), and `describe()` returns the text that was drawn, which is what
    every other test asserts against.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def themes(self):
        return (("dark", app._TOKENS_DARK), ("light", app._TOKENS_LIGHT))

    def test_every_diagram_paints_in_both_themes(self):
        painted = set()
        for theme_name, tokens in self.themes():
            for diagram in diagrams.gallery(tokens):
                with self.subTest(theme=theme_name, diagram=type(diagram).__name__):
                    diagram.resize(diagram.sizeHint())
                    self.assertFalse(diagram.grab().isNull())
                    self.assertGreater(diagram.canvas.height(), 0)
                painted.add(type(diagram))
        self.assertEqual(painted, set(diagrams.DIAGRAMS),
                         "the gallery must cover every diagram, or a widget ships unpainted")

    def test_describe_lists_only_real_non_empty_labels(self):
        for theme_name, tokens in self.themes():
            for diagram in diagrams.gallery(tokens):
                with self.subTest(theme=theme_name, diagram=type(diagram).__name__):
                    described = diagram.describe()
                    self.assertTrue(described)
                    for text in described:
                        self.assertIsInstance(text, str)
                        self.assertTrue(text.strip())

    def test_colours_come_from_the_host_tokens(self):
        """A hard-coded hex in a `_draw` would make both themes identical here."""
        dark = diagrams.LayerStack(36, 0, tokens=app._TOKENS_DARK)
        light = diagrams.LayerStack(36, 0, tokens=app._TOKENS_LIGHT)
        self.assertNotEqual(dark.ink().name(), light.ink().name())
        self.assertNotEqual(dark.surface().name(), light.surface().name())

    def test_selected_row_is_answerable_by_every_diagram(self):
        """The reader's token position is carried across pictures by one call."""
        for diagram in diagrams.gallery(app._TOKENS_DARK):
            with self.subTest(diagram=type(diagram).__name__):
                diagram.resize(diagram.sizeHint())
                diagram.set_selected_row(1)
                self.assertEqual(diagram.selected_row(), 1)
                self.assertFalse(diagram.grab().isNull())
                diagram.set_selected_row(None)
                self.assertIsNone(diagram.selected_row())

    def test_clickable_cells_report_their_operands(self):
        for factory in (lambda: diagrams.MatrixGrid(15, 2560, clickable=True,
                                                    tokens=app._TOKENS_DARK),
                        lambda: diagrams.CausalMaskTriangle(15, 15, clickable=True,
                                                            tokens=app._TOKENS_DARK)):
            widget = factory()
            with self.subTest(diagram=type(widget).__name__):
                widget.resize(420, 220)
                widget.grab()                      # lays the grid out
                seen = []
                widget.cellClicked.connect(lambda row, col: seen.append((row, col)))
                widget.canvas.mousePressEvent(QMouseEvent(
                    QEvent.MouseButtonPress, QPoint(200, 90), Qt.LeftButton,
                    Qt.LeftButton, Qt.NoModifier))
                self.assertEqual(len(seen), 1)
                self.assertEqual(widget.selected_row(), seen[0][0])

    def test_a_mask_click_reports_the_real_position_not_the_drawn_cell(self):
        """The grid draws 8 of 15 query rows, and `pick` indexes captured tensors.

        Emitting the drawn row index pointed every click below the cursor: the bottom
        row of a 15-query mask is query 14, not query 7.
        """
        mask = diagrams.CausalMaskTriangle(15, 15, clickable=True, tokens=app._TOKENS_DARK)
        mask.resize(420, 220)
        mask.grab()                                # lays the grid out
        seen = []
        mask.cellClicked.connect(lambda row, col: seen.append((row, col)))
        grid = mask._grid
        # Click the last drawn row and the last drawn column.
        mask.canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPoint(grid.right() - 1, grid.bottom() - 1),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(seen, [(14, 14)])
        self.assertEqual(mask.selected_row(), 14)
        # And the first cell is still position zero.
        mask.canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPoint(grid.x() + 1, grid.y() + 1),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(seen[-1], (0, 0))
        # Every emitted pair must be addressable in a 15x15 captured matrix.
        for row, col in seen:
            self.assertTrue(0 <= row < 15 and 0 <= col < 15, (row, col))

    def test_a_strip_never_labels_its_drawn_cells_as_the_real_count(self):
        """2,560 features drawn as a dozen cells must still say 2,560."""
        strip = diagrams.VectorStrip(2560, symbol="X", tokens=app._TOKENS_DARK)
        self.assertIn("2,560 features", strip.describe())
        grid = diagrams.MatrixGrid(15, 2560, symbol="X", tokens=app._TOKENS_DARK)
        self.assertTrue(any("2,560" in text for text in grid.describe()))

    def test_the_feed_forward_interior_width_is_never_invented(self):
        """This capture does not record it, so no number may appear in its place."""
        unknown = diagrams.GatedFFNPath(2560, None, tokens=app._TOKENS_DARK)
        spoken = " ".join(unknown.describe())
        self.assertIn("not recorded in this capture", spoken)
        # 2,560 is the in/out width and is real; nothing else numeric may appear.
        self.assertEqual([word for word in spoken.replace(",", "").split()
                          if word.strip("().,").isdigit()], ["2560"])
        known = diagrams.GatedFFNPath(2560, 9728, tokens=app._TOKENS_DARK)
        self.assertIn("9,728 features", known.describe())

    def test_masked_scores_are_not_described_as_negative_infinity(self):
        mask = diagrams.CausalMaskTriangle(15, 15, tokens=app._TOKENS_DARK)
        spoken = " ".join(mask.describe()).lower()
        self.assertIn("large negative number", spoken)
        self.assertNotIn("-inf", spoken)
        self.assertNotIn("infinity", spoken.replace("not necessarily minus infinity", ""))

    def test_a_masked_column_is_drawn_rather_than_dropped(self):
        """Masked entries leave the colour scale, never the picture."""
        scores = np.array([2.0, 3.0, -3.4e38], dtype=np.float32)
        weights = np.array([0.4, 0.6, 0.0], dtype=np.float32)
        bars = diagrams.SoftmaxBars(scores, weights, labels=["0", "1", "2"],
                                    masked_indices=(2,), tokens=app._TOKENS_DARK)
        bars.resize(bars.sizeHint())
        self.assertFalse(bars.grab().isNull())
        self.assertIn("3 of 3 key positions drawn", bars.describe())

    def test_rectangular_cached_attention_reads_as_one_query(self):
        mask = diagrams.CausalMaskTriangle(1, 16, cached=15, tokens=app._TOKENS_DARK)
        described = mask.describe()
        self.assertIn("1 query × 16 keys", described)
        self.assertIn("15 keys came from the cache", described)
        self.assertFalse(any("queries" in text for text in described))

    def test_a_highlighted_row_is_labelled_with_its_own_index(self):
        """Position 13 of 15 is drawn on the row that stands for 13, not clamped to 14."""
        grid = diagrams.MatrixGrid(15, 2560, symbol='X', highlight_row=13,
                                   tokens=app._TOKENS_DARK)
        self.assertIn(13, grid.drawn_rows())
        self.assertIn('row 13', grid.describe())
        for total, mark in ((15, 4), (16, 5), (14, 13), (36, 20)):
            window = diagrams.Diagram.row_window(total, 8, mark)
            self.assertIn(mark, window)
            self.assertEqual(window[0], 0)
            self.assertEqual(window[-1], total - 1)
            real = [index for index in window if index is not None]
            self.assertEqual(real, sorted(set(real)))

    def test_attention_bars_name_their_keys(self):
        scores = np.array([2.0, 3.0, 1.0, 0.5], dtype=np.float32)
        weights = np.array([0.2, 0.5, 0.2, 0.1], dtype=np.float32)
        bars = diagrams.SoftmaxBars(scores, weights, labels=['0', '1', '2', '3'],
                                    sublabels=['the', ' color', '?', 'x'], indices=[1, 3],
                                    tokens=app._TOKENS_DARK)
        self.assertEqual(bars.drawn_keys(), [1, 3])
        self.assertIn(' color', bars.describe())
        self.assertIn('2 of 4 key positions drawn', bars.describe())

    def test_the_vocabulary_projection_names_both_rows(self):
        projection = diagrams.VocabularyProjection(
            2560, 151936, positions=15, final_position=14, focus_position=4,
            tokens=app._TOKENS_DARK)
        spoken = ' '.join(projection.describe())
        self.assertIn('only position 14 chose', spoken)
        self.assertIn('you followed position 4', spoken)
        self.assertIn(4, projection.drawn_rows())
        self.assertIn(14, projection.drawn_rows())

    def test_no_diagram_creates_a_matplotlib_canvas(self):
        """A 15x15 grid is not worth a FigureCanvas; painting it is why this module exists."""
        for diagram in diagrams.gallery(app._TOKENS_DARK):
            diagram.resize(diagram.sizeHint())
            diagram.grab()
            with self.subTest(diagram=type(diagram).__name__):
                self.assertEqual(diagram.findChildren(FigureCanvas), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
