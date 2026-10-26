"""Saved-run UI regression checks, using real captures and no model imports.

Run: ./.venv/Scripts/python.exe test_recap.py [path/to/runs.sqlite3]
The existing database is opened read-only and hashed before/after every test run.
No model weights are loaded and no example tensors are substituted for captures.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.abc
import itertools
import json
import os
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
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QTableWidget

import TensorScope as app
import tensorscope_diagrams as diagrams
from tensor_widgets import AttentionExplorer, TensorInspector, TensorTableModel, exact_scalar
from tensorscope_common import (
    AnnotatedValueRow, Disclosure, EvidenceBadge, PredictCheck, ResponseTextView,
    StepReveal,
)
from tensorscope_content import (
    CONCEPTUAL_VARIABLES, CONTINUATION, EVIDENCE_KINDS, EVIDENCE_OBSERVED,
    EVIDENCE_UNATTESTED, LOGITS_UNAVAILABLE, NOTEBOOK_CHECKS, NOTEBOOK_OPERATIONS,
    PROVENANCE_ABSENT, PROVENANCE_CAPTURED, PROVENANCE_DISPLAY, PROVENANCE_LEARNED,
    REFERENCE_TOPICS, TELEMETRY_UNAVAILABLE, VARIABLE_ROLES, learn_facts,
)
from tensorscope_diagrams import CausalMaskTriangle, Diagram
from tensorscope_ui import DEFAULT_MODE, MODES, ComputationRecap
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
                    for field in ('heading', 'lede', 'formula'):
                        getattr(operation, field).format(**filled)
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

            for reveal in notebook.findChildren(StepReveal):
                reveal.reveal_all()
            for check in notebook.findChildren(PredictCheck):
                check.choose(-1)
            open_every_disclosure(notebook)
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
                    open_every_disclosure(notebook.section(operation.key))
                    allowed = {EVIDENCE_KINDS[tier][0] for tier in operation.tiers}
                    for badge in notebook.section(operation.key).findChildren(EvidenceBadge):
                        name = badge.text().split('  ·  ')[0]
                        self.assertIn(name, allowed,
                                      f'{operation.key} showed an undeclared tier')
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
                with self.subTest(run_id=run_id, operation=key):
                    if key != 'continuation':
                        self.assertEqual(found, [], f'{key} claimed an unattested value')
            notebook.deleteLater()

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
        # Measured at this width, the twelve operations run from about 720 to about 2,000
        # pixels -- one to three screens each, which is the readable chunk the Next
        # operation control is there to move between. The bounds are set just above that,
        # so a change that turns a section into a page of its own fails here. They are
        # not an aspiration: heavy content belongs behind a `Disclosure`, which is what
        # `test_no_matplotlib_canvas_is_built_until_the_reader_asks` checks.
        heights = {}
        for section in notebook.sections():
            heights[section.key] = section.sizeHint().height()
            with self.subTest(section=section.key):
                # The continuation holds the whole answer plus three tables, which is the
                # one place where showing everything at once is the point.
                limit = 3200 if section.key == 'continuation' else 2200
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
        for name in ('phase_picker', 'layer_picker', 'head_picker'):
            self.assertTrue(getattr(notebook, name).accessibleName(), name)
        # Every emitted token must be reachable without a mouse, because the response
        # anchors only answer linkActivated.
        notebook.go_to('continuation')
        response = notebook.findChild(ResponseTextView)
        steppers = [button for button in notebook.section('continuation')
                    .findChildren(QPushButton) if 'token' in button.text()]
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
        self.assertEqual([key for key, _, _, _ in MODES], ['understand', 'run'])
        self.assertEqual(recap._mode, DEFAULT_MODE)
        for key, name, _, _ in MODES:
            recap._show_mode(key)
            self.assertTrue(recap._mode_buttons[key].isChecked())
            self.assertEqual(set(recap._stage_buttons), {k for k, _ in recap.view.NAV})
            self.assertIn(name, recap.breadcrumb.text())
            for stage_key, stage_name in recap.view.NAV:
                recap._nav_to(stage_key)
                self.assertEqual(recap.view.current_stage(), stage_key)
                self.assertTrue(recap._stage_buttons[stage_key].isChecked())
                self.assertIn(stage_name, recap.breadcrumb.text())
            recap._advance(1)
            recap._advance(-1)
        recap.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertEqual(before, capture_digest(capture))

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

    def test_no_diagram_creates_a_matplotlib_canvas(self):
        """A 15x15 grid is not worth a FigureCanvas; painting it is why this module exists."""
        for diagram in diagrams.gallery(app._TOKENS_DARK):
            diagram.resize(diagram.sizeHint())
            diagram.grab()
            with self.subTest(diagram=type(diagram).__name__):
                self.assertEqual(diagram.findChildren(FigureCanvas), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
