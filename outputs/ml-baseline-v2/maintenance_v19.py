import argparse
import csv
import hashlib
import json
import math
import posixpath
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
WORK_KINDS = {"ТО", "ТР", "ТО+ТР"}
MONTH_COLUMNS = "GHIJKLMNOPQR"


def position(address):
    match = re.fullmatch(r"([A-Z]+)([1-9]\d*)", address)
    if not match:
        raise ValueError("Invalid cell address")
    column = 0
    for letter in match[1]:
        column = column * 26 + ord(letter) - 64
    return int(match[2]), column


def read_book(path):
    source_bytes = Path(path).read_bytes()
    with ZipFile(BytesIO(source_bytes)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 32_000_000:
            raise ValueError("Workbook is too large")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("m:sheets/m:sheet", NS)
        if len(sheets) != 1:
            raise ValueError("Expected one source worksheet")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation = next(item for item in relations if item.get("Id") == sheets[0].get(REL))
        if relation.get("TargetMode") == "External":
            raise ValueError("External worksheet is unsupported")
        target = relation.get("Target")
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        if not member.startswith("xl/"):
            raise ValueError("Invalid worksheet path")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(node.text or "" for node in item.findall(".//m:t", NS))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)
            ]
        worksheet = ET.fromstring(archive.read(member))
        cells = {}
        hidden_rows = set()
        for row in worksheet.findall("m:sheetData/m:row", NS):
            if row.get("hidden") == "1":
                hidden_rows.add(int(row.get("r")))
            for cell in row.findall("m:c", NS):
                if cell.find("m:f", NS) is not None:
                    raise ValueError("Formula cells require independent recalculation")
                address = cell.get("r")
                position(address)
                value = cell.find("m:v", NS)
                if cell.get("t") == "inlineStr":
                    text = "".join(node.text or "" for node in cell.findall("m:is//m:t", NS))
                elif value is None or value.text is None:
                    continue
                elif cell.get("t") == "s":
                    text = strings[int(value.text)]
                else:
                    text = value.text
                if address in cells:
                    raise ValueError("Duplicate source cell")
                cells[address] = text
        merged = []
        for item in worksheet.findall("m:mergeCells/m:mergeCell", NS):
            first, last = item.get("ref").split(":")
            merged.append((first, position(first), position(last)))
        properties = workbook.find("m:workbookPr", NS)
        return {
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "sheet": sheets[0].get("name"),
            "date1904": properties is not None and properties.get("date1904") in ("1", "true"),
            "cells": cells,
            "merged": merged,
            "hidden_rows": hidden_rows,
        }


def cell_value(book, address):
    if address in book["cells"]:
        return book["cells"][address], address
    row, column = position(address)
    for anchor, (first_row, first_column), (last_row, last_column) in book["merged"]:
        if first_row <= row <= last_row and first_column <= column <= last_column:
            return book["cells"].get(anchor), anchor
    return None, None


def quantity(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Equipment quantity must be positive and finite")
    return int(number) if number.is_integer() else number


def title_year(title):
    match = re.search(r"на\s+(20\d{2})\s*г", title or "", re.I)
    if not match:
        raise ValueError("Missing schedule year in title")
    return int(match[1])


def excel_date(value, date1904=False):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or ""):
        return date.fromisoformat(value)
    serial = float(value)
    if not math.isfinite(serial) or not serial.is_integer():
        raise ValueError("Expected an Excel calendar date")
    serial = int(serial)
    if date1904:
        if serial < 0:
            raise ValueError("Negative Excel date")
        return date(1904, 1, 1) + timedelta(days=serial)
    if serial < 1 or serial == 60:
        raise ValueError("Invalid 1900-system Excel date")
    epoch = date(1899, 12, 31) if serial < 60 else date(1899, 12, 30)
    return epoch + timedelta(days=serial)


def source_record(book, row, object_key, object_label):
    return {
        "source_sha256": book["sha256"],
        "sheet": book["sheet"],
        "row": row,
        "hidden_row": row in book["hidden_rows"],
        "object_key": object_key,
        "object_label": object_label,
        "record_type": "planned",
        "object_id": None,
        "available_at": None,
    }


def extract_monthly(book):
    year = title_year(book["cells"].get("G2"))
    records, consumed = [], set()
    object_key, object_label = None, None
    seen_objects = set()
    for row in sorted({position(address)[0] for address in book["cells"]}):
        ordinal = book["cells"].get(f"B{row}", "")
        if re.fullmatch(r"\d+", ordinal):
            object_key = f"monthly:{ordinal}"
            if object_key in seen_objects:
                raise ValueError("Duplicate monthly object section")
            seen_objects.add(object_key)
            object_label = cell_value(book, f"C{row}")[0]
        equipment = book["cells"].get(f"D{row}")
        if equipment is None or equipment.strip() == "Марка":
            continue
        if object_key is None:
            raise ValueError("Equipment without object section")
        record = source_record(book, row, object_key, object_label)
        record.update(equipment=equipment.strip(), quantity=quantity(book["cells"].get(f"E{row}")),
                      unit=book["cells"].get(f"F{row}"), year=year, works=[])
        for month, column in enumerate(MONTH_COLUMNS, 1):
            address = f"{column}{row}"
            value = book["cells"].get(address)
            if value is None or not value.strip():
                continue
            kind = re.sub(r"\s+", "", value)
            if kind not in WORK_KINDS:
                raise ValueError(f"Unknown monthly work kind at {address}")
            start = date(year, month, 1)
            end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            record["works"].append({"cell": address, "kind": kind, "start_on": start.isoformat(),
                                    "end_before": end.isoformat(), "precision": "month"})
            consumed.add(address)
        records.append(record)
    marks = {
        address for address, value in book["cells"].items()
        if position(address)[1] in range(7, 19) and re.sub(r"\s+", "", value or "") in WORK_KINDS
    }
    if not records or not consumed <= marks:
        raise ValueError("Monthly schedule did not reconcile")
    return records, {
        "objects": len(seen_objects), "equipment_rows": len(records),
        "work_marks": len(consumed), "all_work_marks": len(marks),
        "work_kind_counts": {kind: sum(work["kind"] == kind for record in records for work in record["works"])
                             for kind in sorted(WORK_KINDS)},
        "unassigned_work_cells": sorted(marks - consumed, key=position),
        "rows_without_work": sum(not record["works"] for record in records),
        "hidden_equipment_rows": sum(record["hidden_row"] for record in records),
        "hidden_work_marks": sum(len(record["works"]) for record in records if record["hidden_row"]),
        "title_year": year,
        "sheet_year_conflict": any(int(value) != year for value in re.findall(r"20\d{2}", book["sheet"])),
    }


def extract_ppr(book):
    year = title_year(book["cells"].get("A7"))
    records, seen_objects = [], set()
    for address, label in sorted(book["cells"].items(), key=lambda item: position(item[0])):
        if position(address)[1] != 3:
            continue
        match = re.fullmatch(r"Объект\s+(\d+)", label.strip())
        if not match:
            continue
        row = position(address)[0]
        key = f"ppr:{match[1]}"
        if key in seen_objects:
            raise ValueError("Duplicate PPR object")
        seen_objects.add(key)
        record = source_record(book, row, key, label)
        record.update(quantity=quantity(book["cells"].get(f"D{row}")), year=year,
                      month_label=cell_value(book, f"B{row}")[0], source_cells={})
        for column, field in (("E", "dismantle_on"), ("F", "deliver_by"), ("G", "pickup_on"),
                              ("H", "commission_on")):
            value, anchor = cell_value(book, f"{column}{row}")
            if value is None:
                raise ValueError(f"Missing PPR stage at {column}{row}")
            record["source_cells"][field] = anchor
            if column == "F":
                record["delivery_text"] = value
                found = re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", value)
                if len(found) != 1:
                    raise ValueError("Ambiguous PPR delivery date")
                parsed = datetime.strptime(found[0], "%d.%m.%Y").date()
            else:
                parsed = excel_date(value, book["date1904"])
            record[field] = parsed.isoformat()
        stages = [record[field] for field in ("dismantle_on", "deliver_by", "pickup_on", "commission_on")]
        if stages != sorted(stages):
            raise ValueError("PPR stages are out of order")
        records.append(record)
    if not records:
        raise ValueError("No PPR objects found")
    return records, {
        "objects": len(records), "sensor_quantity": sum(record["quantity"] for record in records),
        "distinct_dismantling_windows": len({(record["dismantle_on"], record["pickup_on"]) for record in records}),
        "first_dismantle_on": min(record["dismantle_on"] for record in records),
        "last_commission_on": max(record["commission_on"] for record in records),
        "title_year": year,
        "date_year_conflicts": sum(int(record[field][:4]) != year for record in records
                                   for field in ("dismantle_on", "deliver_by", "pickup_on", "commission_on")),
    }


def aware_time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return parsed


def available_records(records, bindings, registry_ids, as_of):
    cutoff = aware_time(as_of)
    valid_keys = {(record["source_sha256"], record["object_key"]) for record in records}
    indexed = {}
    for binding in bindings:
        key = binding["source_sha256"], binding["object_key"]
        if key not in valid_keys or key in indexed:
            raise ValueError("Unknown source version, object key, or duplicate binding")
        if binding["object_id"] not in registry_ids:
            raise ValueError("Unknown registry object ID")
        for field in ("mapping_evidence", "availability_evidence"):
            if not isinstance(binding.get(field), str) or not binding[field].strip():
                raise ValueError("Binding and availability require evidence")
        available_at = aware_time(binding["available_at"])
        indexed[key] = binding, available_at
    eligible = []
    for record in records:
        bound = indexed.get((record["source_sha256"], record["object_key"]))
        if bound is not None and bound[1] <= cutoff:
            eligible.append(dict(record, object_id=bound[0]["object_id"], available_at=bound[0]["available_at"]))
    return eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("monthly", type=Path)
    parser.add_argument("ppr", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    if any((args.bindings, args.registry, args.as_of)) and not all((args.bindings, args.registry, args.as_of)):
        parser.error("--bindings, --registry and --as-of must be specified together")
    monthly_book, ppr_book = read_book(args.monthly), read_book(args.ppr)
    monthly, monthly_audit = extract_monthly(monthly_book)
    ppr, ppr_audit = extract_ppr(ppr_book)
    records = monthly + ppr
    eligible = []
    if args.bindings:
        with args.registry.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"ид_канала_данных", "ид_объект"} <= set(reader.fieldnames or []):
                raise ValueError("--registry must be a channel catalogue")
            registry_ids = {row["ид_объект"] for row in reader}
        eligible = available_records(records, json.loads(args.bindings.read_text()), registry_ids, args.as_of)
    audit = {
        "monthly": dict(monthly_audit, source_sha256=monthly_book["sha256"], sheet=monthly_book["sheet"]),
        "ppr": dict(ppr_audit, source_sha256=ppr_book["sha256"], sheet=ppr_book["sheet"]),
        "eligible_equipment_or_ppr_rows": len(eligible),
        "binding_supplied": args.bindings is not None,
        "as_of": args.as_of,
        "actual_outcome_labels_created": False,
        "training_performed": False,
        "prediction_performed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in (("monthly.json", monthly), ("ppr.json", ppr), ("eligible.json", eligible), ("audit.json", audit)):
        path = args.output / name
        if path.resolve() in {args.monthly.resolve(), args.ppr.resolve(), args.bindings.resolve() if args.bindings else None,
                             args.registry.resolve() if args.registry else None}:
            raise ValueError("Output would overwrite an input")
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
