import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import maintenance_v19 as m


def monthly_book():
    return {
        "sha256": "a" * 64, "sheet": "2025 год", "date1904": False, "hidden_rows": {8}, "merged": [],
        "cells": {
            "G2": "График ТО и ТР на 2026г", "B6": "1", "C6": "Объект", "G6": "ТО",
            "D7": "Газоанализаторы", "E7": "3", "F7": "шт.", "G7": "ТО", "R7": "ТО+ТР",
            "D8": "БП", "E8": "2", "F8": "шт.", "N8": "ТР",
            "D9": "Кабельные линии", "E9": "25", "F9": "м.",
            "D10": "Марка", "G10": "Месяц ТО",
        },
    }


def ppr_book():
    return {
        "sha256": "b" * 64, "sheet": "РЭК", "date1904": False, "hidden_rows": set(),
        "merged": [(f"{column}10", (10, column_number), (11, column_number))
                   for column, column_number in (("B", 2), ("E", 5), ("F", 6), ("G", 7))],
        "cells": {
            "A7": "План-график ППР на 2026 год", "B10": "Январь",
            "C10": "Объект 1", "D10": "3", "E10": "46034", "F10": "до 900 13.01.2026",
            "G10": "46044", "H10": "46049", "C11": "Объект 2", "D11": "5", "H11": "46050",
        },
    }


def write_book(book, path, formula=False):
    namespace = m.NS["m"]
    workbook = ET.Element(f"{{{namespace}}}workbook")
    ET.SubElement(workbook, f"{{{namespace}}}workbookPr", date1904=str(int(book["date1904"])))
    sheets = ET.SubElement(workbook, f"{{{namespace}}}sheets")
    ET.SubElement(sheets, f"{{{namespace}}}sheet", {"name": book["sheet"], "sheetId": "1", m.REL: "r1"})
    worksheet = ET.Element(f"{{{namespace}}}worksheet")
    data = ET.SubElement(worksheet, f"{{{namespace}}}sheetData")
    rows = {}
    for address, value in book["cells"].items():
        number = m.position(address)[0]
        if number not in rows:
            rows[number] = ET.SubElement(data, f"{{{namespace}}}row",
                                         {"r": str(number), "hidden": str(int(number in book["hidden_rows"]))})
        cell = ET.SubElement(rows[number], f"{{{namespace}}}c", r=address, t="inlineStr")
        inline = ET.SubElement(cell, f"{{{namespace}}}is")
        ET.SubElement(inline, f"{{{namespace}}}t").text = value
        if formula:
            ET.SubElement(cell, f"{{{namespace}}}f").text = "1+1"
    merged = ET.SubElement(worksheet, f"{{{namespace}}}mergeCells")
    for anchor, _, end in book["merged"]:
        column = chr(64 + end[1])
        ET.SubElement(merged, f"{{{namespace}}}mergeCell", ref=f"{anchor}:{column}{end[0]}")
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", ET.tostring(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels",
                         '<Relationships><Relationship Id="r1" Target="/xl/worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(worksheet))


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.records = m.extract_ppr(ppr_book())[0]
        self.binding = {
            "source_sha256": "b" * 64, "object_key": "ppr:1", "object_id": "20",
            "available_at": "2025-12-31T23:00:00+03:00",
            "mapping_evidence": "Source crosswalk", "availability_evidence": "Version approval",
        }

    def test_monthly_preserves_equipment_hidden_rows_and_empty_schedule(self):
        records, audit = m.extract_monthly(monthly_book())
        self.assertEqual([record["quantity"] for record in records], [3, 2, 25])
        self.assertTrue(records[1]["hidden_row"])
        self.assertEqual(records[2]["works"], [])
        self.assertEqual(audit["rows_without_work"], 1)
        self.assertEqual(audit["hidden_work_marks"], 1)

    def test_monthly_reconciles_unassigned_marker(self):
        records, audit = m.extract_monthly(monthly_book())
        self.assertEqual(audit["all_work_marks"], 4)
        self.assertEqual(audit["work_marks"], 3)
        self.assertEqual(audit["unassigned_work_cells"], ["G6"])
        self.assertEqual(sum(len(record["works"]) for record in records) + len(audit["unassigned_work_cells"]), 4)

    def test_monthly_title_year_and_december_interval(self):
        records, audit = m.extract_monthly(monthly_book())
        self.assertTrue(audit["sheet_year_conflict"])
        self.assertEqual(records[0]["year"], 2026)
        december = records[0]["works"][1]
        self.assertEqual((december["start_on"], december["end_before"], december["precision"]),
                         ("2026-12-01", "2027-01-01", "month"))

    def test_monthly_rejects_unknown_mark_and_duplicate_section(self):
        book = monthly_book()
        book["cells"]["G7"] = "Выполнено"
        with self.assertRaises(ValueError):
            m.extract_monthly(book)
        book = monthly_book()
        book["cells"].update(B11="1", C11="Объект")
        with self.assertRaises(ValueError):
            m.extract_monthly(book)

    def test_schedule_year_requires_explicit_title(self):
        for title in (None, "", "График", "лист 2025 год"):
            with self.subTest(title=title), self.assertRaises(ValueError):
                m.title_year(title)

    def test_ppr_merged_stages_keep_individual_commission_dates(self):
        records, audit = m.extract_ppr(ppr_book())
        self.assertEqual(audit["sensor_quantity"], 8)
        self.assertEqual(audit["distinct_dismantling_windows"], 1)
        self.assertEqual([record["commission_on"] for record in records], ["2026-01-27", "2026-01-28"])
        self.assertEqual(records[1]["source_cells"]["dismantle_on"], "E10")
        self.assertEqual(records[1]["delivery_text"], "до 900 13.01.2026")
        self.assertEqual(records[1]["record_type"], "planned")

    def test_ppr_does_not_fill_beyond_merged_range(self):
        book = ppr_book()
        book["cells"].update(C12="Объект 3", D12="2", H12="46051")
        self.assertEqual(m.cell_value(book, "E12"), (None, None))
        with self.assertRaises(ValueError):
            m.extract_ppr(book)

    def test_ppr_rejects_reversed_stages_and_duplicate_object(self):
        book = ppr_book()
        book["cells"]["H11"] = "46033"
        with self.assertRaises(ValueError):
            m.extract_ppr(book)
        book = ppr_book()
        book["cells"]["C11"] = "Объект 1"
        with self.assertRaises(ValueError):
            m.extract_ppr(book)

    def test_excel_date_epochs_and_fictitious_leap_day(self):
        self.assertEqual(m.excel_date("1"), date(1900, 1, 1))
        self.assertEqual(m.excel_date("59"), date(1900, 2, 28))
        self.assertEqual(m.excel_date("61"), date(1900, 3, 1))
        self.assertEqual(m.excel_date("0", True), date(1904, 1, 1))
        self.assertEqual(m.excel_date(str((date(2026, 1, 12) - date(1904, 1, 1)).days), True),
                         date(2026, 1, 12))
        with self.assertRaises(ValueError):
            m.excel_date("60")

    def test_fractional_nonfinite_and_negative_dates_and_quantities_fail(self):
        for value in ("46034.5", "nan", "inf", "-1"):
            with self.subTest(date=value), self.assertRaises(ValueError):
                m.excel_date(value)
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(quantity=value), self.assertRaises(ValueError):
                m.quantity(value)

    def test_book_roundtrip_preserves_hash_merges_and_hidden_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, source in enumerate((monthly_book(), ppr_book())):
                path = Path(folder) / f"{index}.xlsx"
                write_book(source, path)
                loaded = m.read_book(path)
                self.assertEqual(loaded["cells"], source["cells"])
                self.assertEqual(loaded["hidden_rows"], source["hidden_rows"])
                self.assertEqual(loaded["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(loaded["merged"], source["merged"])

    def test_formula_cells_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "formula.xlsx"
            write_book(ppr_book(), path, formula=True)
            with self.assertRaises(ValueError):
                m.read_book(path)

    def test_object_number_does_not_bind_without_evidence(self):
        self.assertEqual(m.available_records(self.records, [], {"1", "2", "20"}, "2026-01-01T00:00:00Z"), [])
        self.assertTrue(all(record["object_id"] is None for record in self.records))

    def test_binding_is_version_specific_and_registry_id_is_required(self):
        for field, value in (("source_sha256", "c" * 64), ("object_key", "monthly:1"), ("object_id", "unknown")):
            binding = dict(self.binding, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.available_records(self.records, [binding], {"20"}, "2026-01-01T00:00:00Z")

    def test_duplicate_bindings_are_rejected(self):
        with self.assertRaises(ValueError):
            m.available_records(self.records, [self.binding, self.binding], {"20"}, "2026-01-01T00:00:00Z")

    def test_evidence_must_be_nonempty_for_both_facts(self):
        for field in ("mapping_evidence", "availability_evidence"):
            for value in (None, "", "  ", 1):
                binding = dict(self.binding, **{field: value})
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    m.available_records(self.records, [binding], {"20"}, "2026-01-01T00:00:00Z")

    def test_naive_available_time_and_cutoff_are_rejected(self):
        with self.assertRaises(ValueError):
            m.available_records(self.records, [self.binding], {"20"}, "2026-01-01")
        binding = dict(self.binding, available_at="2025-12-31T23:00:00")
        with self.assertRaises(ValueError):
            m.available_records(self.records, [binding], {"20"}, "2026-01-01T00:00:00Z")

    def test_future_version_is_excluded_at_one_second_boundary(self):
        binding = dict(self.binding, available_at="2026-01-01T03:00:00+03:00")
        self.assertEqual(m.available_records(self.records, [binding], {"20"}, "2025-12-31T23:59:59Z"), [])
        eligible = m.available_records(self.records, [binding], {"20"}, "2026-01-01T00:00:00Z")
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["object_id"], "20")

    def test_known_future_plan_is_allowed_without_changing_original(self):
        before = copy.deepcopy(self.records)
        eligible = m.available_records(self.records, [self.binding], {"20"}, "2026-01-01T00:00:00Z")
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["dismantle_on"], "2026-01-12")
        self.assertEqual(self.records, before)

    def test_cli_extracts_without_training_or_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            monthly, ppr, output = root / "monthly.xlsx", root / "ppr.xlsx", root / "result"
            write_book(monthly_book(), monthly)
            write_book(ppr_book(), ppr)
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (monthly, ppr)]
            run = subprocess.run([sys.executable, str(Path(m.__file__)), str(monthly), str(ppr), str(output)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            audit = json.loads(run.stdout)
            self.assertFalse(audit["training_performed"])
            self.assertFalse(audit["prediction_performed"])
            self.assertFalse(audit["actual_outcome_labels_created"])
            self.assertEqual(audit["eligible_equipment_or_ppr_rows"], 0)
            self.assertEqual(json.loads((output / "eligible.json").read_text()), [])
            self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in (monthly, ppr)], hashes)

    def test_cli_binding_accepts_channel_catalogue_and_rejects_facility_only_catalogue(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            monthly, ppr = root / "monthly.xlsx", root / "ppr.xlsx"
            write_book(monthly_book(), monthly)
            write_book(ppr_book(), ppr)
            binding = dict(self.binding, source_sha256=hashlib.sha256(ppr.read_bytes()).hexdigest())
            binding_path = root / "binding.json"
            binding_path.write_text(json.dumps([binding]), encoding="utf-8")
            for valid in (True, False):
                with self.subTest(channel_catalogue=valid):
                    registry, output = root / "registry.csv", root / str(valid)
                    registry.write_text("ид_канала_данных,ид_объект\n100,20\n" if valid else "ид_объект\n20\n",
                                        encoding="utf-8-sig")
                    run = subprocess.run([sys.executable, str(Path(m.__file__)), str(monthly), str(ppr), str(output),
                                          "--bindings", str(binding_path), "--registry", str(registry),
                                          "--as-of", "2026-01-01T00:00:00Z"], capture_output=True, text=True)
                    if valid:
                        self.assertEqual(run.returncode, 0, run.stderr)
                        self.assertEqual(json.loads((output / "eligible.json").read_text())[0]["object_id"], "20")
                    else:
                        self.assertNotEqual(run.returncode, 0)
                        self.assertIn("channel catalogue", run.stderr)
                        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
