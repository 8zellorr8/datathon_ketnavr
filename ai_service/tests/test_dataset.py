"""Meaningful importer checks using tiny synthetic MOV placeholders."""

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from forte_ai.dataset import (
    DatasetValidationError,
    SCENARIO_GROUPS,
    normalize_video_id,
    prepare_dataset,
)


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
HEADERS = (
    "Файл", "Метка организатора", "Таймкод (сек.)", "Что происходит",
    "Завершение", "Моё наблюдение", "Что непонятно или не видно",
    "Что видно около события", "Что было до и после",
    "Рабочая трактовка / предположение",
)
NUMBERS = sorted(number for group in SCENARIO_GROUPS for number in group)


def write_workbook(path, rows):
    sheet = ET.Element(f"{{{MAIN_NS}}}worksheet")
    data = ET.SubElement(sheet, f"{{{MAIN_NS}}}sheetData")
    for number, values in [(6, HEADERS)] + list(enumerate(rows, start=7)):
        row = ET.SubElement(data, f"{{{MAIN_NS}}}row", r=str(number))
        for column, value in zip("ABCDEFGHIJ", values):
            cell = ET.SubElement(row, f"{{{MAIN_NS}}}c", r=f"{column}{number}", t="inlineStr")
            inline = ET.SubElement(cell, f"{{{MAIN_NS}}}is")
            ET.SubElement(inline, f"{{{MAIN_NS}}}t").text = str(value)
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet))
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN_NS}" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Review" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.workbook = self.root / "review.xlsx"
        self.output = self.root / "dataset"
        self.rows = [
            [f"X-{number}", "нарушение" if index % 2 == 0 else "без нарушения",
             "7.5" if index % 2 == 0 else "", "Наблюдение", "Завершение",
             "Моё наблюдение", "Неясно", "Около события", "До и после", "Гипотеза"]
            for index, number in enumerate(NUMBERS)
        ]
        for number in NUMBERS:
            (self.videos / f"X-{number}.mov").write_bytes(f"unique-video-{number}".encode())

    def run_import(self):
        write_workbook(self.workbook, self.rows)
        return prepare_dataset(self.workbook, self.videos, self.output)

    def test_cyrillic_ids_match_without_changing_real_paths(self):
        self.rows[0][0] = "Х-1"
        original = self.videos / "X-1.mov"
        real_path = self.videos / "х-1.MOV"
        original.rename(real_path)
        result = self.run_import()
        record = result["records"][0]
        self.assertEqual(record["id"], "X-1")
        self.assertEqual(record["video_path"], str(real_path.resolve()))
        self.assertEqual(record["event_time_sec"], 7.5)
        self.assertEqual(record["hypothesis"], "Гипотеза")
        self.assertIn("Завершение: Завершение", record["notes"])
        self.assertEqual(record["source_row"], 7)
        self.assertEqual(normalize_video_id(" Х-001.mov "), "X-1")
        self.assertEqual(len(result["records"]), 42)
        self.assertTrue((self.output / "manifest.json").is_file())

    def test_unknown_label_fails_and_writes_invalid_audit(self):
        self.rows[0][1] = "наверное нарушение"
        with self.assertRaisesRegex(DatasetValidationError, "Unknown organizer label"):
            self.run_import()
        audit = json.loads((self.output / "dataset_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "invalid")
        self.assertFalse((self.output / "manifest.json").exists())

    def test_missing_video_fails(self):
        (self.videos / "X-1.mov").unlink()
        with self.assertRaisesRegex(DatasetValidationError, "Missing video for X-1"):
            self.run_import()

    def test_exact_duplicates_merge_split_units_and_preserve_labels(self):
        # X-1 and X-48 are in different manual groups, with different labels.
        (self.videos / "X-48.mov").write_bytes((self.videos / "X-1.mov").read_bytes())
        self.rows[-1][1] = "без нарушения"
        self.rows[-1][2] = ""
        result = self.run_import()
        by_id = {record["id"]: record for record in result["records"]}
        self.assertNotEqual(by_id["X-1"]["source_scenario_group"], by_id["X-48"]["source_scenario_group"])
        self.assertEqual(by_id["X-1"]["split_group"], by_id["X-48"]["split_group"])
        self.assertEqual(by_id["X-1"]["sha256"], by_id["X-48"]["sha256"])
        self.assertEqual(by_id["X-1"]["label"], 1)
        self.assertEqual(by_id["X-48"]["label"], 0)
        audit = json.loads((self.output / "dataset_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["duplicate_groups"][0]["ids"], ["X-1", "X-48"])
        self.assertTrue(audit["duplicate_groups"][0]["label_conflict"])
        self.assertTrue(audit["duplicate_groups"][0]["event_time_conflict"])
        self.assertEqual(audit["duplicate_groups"][0]["event_times_by_id"], {"X-1": 7.5, "X-48": None})

    def test_duplicate_event_time_conflict_preserves_original_timestamps(self):
        (self.videos / "X-47.mov").write_bytes((self.videos / "X-4.mov").read_bytes())
        for row in self.rows:
            if row[0] in {"X-4", "X-47"}:
                row[1] = "нарушение"
                row[2] = "57" if row[0] == "X-4" else "68"
        result = self.run_import()
        by_id = {record["id"]: record for record in result["records"]}
        self.assertEqual(by_id["X-4"]["event_time_sec"], 57.0)
        self.assertEqual(by_id["X-47"]["event_time_sec"], 68.0)
        audit = json.loads((self.output / "dataset_audit.json").read_text(encoding="utf-8"))
        duplicate = audit["duplicate_groups"][0]
        self.assertFalse(duplicate["label_conflict"])
        self.assertTrue(duplicate["event_time_conflict"])
        self.assertEqual(duplicate["event_times_by_id"], {"X-4": 57.0, "X-47": 68.0})
        self.assertTrue(any("conflicting event timestamps" in warning for warning in audit["warnings"]))

    def test_duplicate_normalized_ids_fail(self):
        self.rows[1][0] = "Х-1"
        with self.assertRaisesRegex(DatasetValidationError, "Duplicate normalized ID"):
            self.run_import()

    def test_negative_event_time_and_wrong_row_count_fail(self):
        self.rows[0][2] = "-1"
        self.rows.pop()
        with self.assertRaises(DatasetValidationError) as caught:
            self.run_import()
        self.assertIn("nonnegative", str(caught.exception))
        self.assertIn("Expected 42 data rows, found 41", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
