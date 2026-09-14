"""Saved-run UI regression checks, using real captures and no model imports.

Run: ./.venv/Scripts/python.exe test_recap.py [path/to/runs.sqlite3]
The existing database is opened read-only and hashed before/after every test run.
No model weights are loaded and no example tensors are substituted for captures.
"""

from __future__ import annotations

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
from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtWidgets import QApplication

import TensorScope as app
from tensor_widgets import AttentionExplorer, TensorInspector, TensorTableModel, exact_scalar


DATABASE = Path(sys.argv.pop(1)) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else app.DB_PATH


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
        for capture in self.captures.values():
            for stage in app.STORY_STAGES:
                stage.heading.format(**app.story_facts(capture))
                stage.plain.format(**app.story_facts(capture))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
