#!/usr/bin/env python3
"""
===============================================================================
KestrelEye — Unified Grapevine Disease Dataset Builder
===============================================================================

Scopo:
    Scaricare, normalizzare e unificare i dataset pubblici Tier S di malattie
    della vite in un unico dataset YOLOv8-ready, con tassonomia a 9 classi
    focalizzata su malattie diffuse in area FVG / Europa.

Dataset integrati:
    1. Bacterial-Fungicidal Vine Disease (10k UAV images)
       - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11301173/
       - Licenza: CC-BY 4.0
    2. Tardif et al. Expertized Grapevine Disease DB (1.483 RGB, focus FD)
       - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10294005/
       - Licenza: CC-BY 4.0
    3. LDD Leaf Diseases Dataset (1.092 RGB + segmentation)
       - https://arxiv.org/pdf/2206.10192
       - Licenza: Academic / Open

    Opzionali (integrabili con pochi adjustments):
    4. Saccuti UniPR Lambrusco Multispectral (172 images, 6 bande)
       - https://zenodo.org/records/14936376
       - Licenza: CC-BY 4.0
       - Nota: multispettrale, gestione a parte (pipeline multi-channel)

Output:
    data/
    └── grape_unified/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        ├── data.yaml                 ← file pronto per YOLOv8
        ├── CITATIONS.md              ← citazioni CC-BY obbligatorie
        └── dataset_report.json       ← statistiche unificazione

Autore: KestrelEye / HLD
Versione: 1.0 (2026-04)
===============================================================================
"""

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Dipendenze esterne (installare con: pip install pyyaml tqdm requests pillow)
try:
    import requests
    import yaml
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERRORE] Dipendenza mancante: {e.name}")
    print("Installa con: pip install pyyaml tqdm requests pillow roboflow")
    sys.exit(1)


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

# Tassonomia KestrelEye v1 — 9 classi
KE_CLASSES = {
    0: "healthy_leaf",
    1: "flavescence_doree_leaf",
    2: "esca_leaf",
    3: "downy_mildew_leaf",      # peronospora
    4: "powdery_mildew_leaf",    # oidio
    5: "black_rot_leaf",
    6: "healthy_bunch",
    7: "damaged_bunch",
    8: "shoot",                  # tralcio
}

# Mapping da classi sorgente (dataset pubblici) → classi KE
# Arricchibile man mano che si scoprono nomi nuovi nei dataset scaricati.
SOURCE_CLASS_MAPPING = {
    # Bacterial-Fungicidal (Romania) — naming approssimato, da validare dopo download
    "healthy": 0, "healthy_leaf": 0, "leaf_healthy": 0,
    "downy_mildew": 3, "downymildew": 3, "peronospora": 3, "mildew_downy": 3,
    "powdery_mildew": 4, "powderymildew": 4, "oidio": 4, "mildew_powdery": 4,
    "black_rot": 5, "blackrot": 5, "black-rot": 5,
    "healthy_bunch": 6, "bunch_healthy": 6, "grape_cluster_healthy": 6,
    "damaged_bunch": 7, "bunch_damaged": 7, "rotten_bunch": 7,

    # Tardif et al. (focus FD)
    "fd": 1, "flavescence_doree": 1, "flavescence": 1, "fd_symptomatic": 1,
    "esca": 2, "esca_symptomatic": 2, "ed": 2, "mal_esca": 2,
    "confounding": 0,  # "confounding leaves" = foglie sane con sintomi ambigui

    # LDD dataset
    "leaf_blight": 5,  # mappiamo leaf blight su black_rot come fallback
    "shoot": 8, "tralcio": 8,
}

# URL sorgenti (alcuni richiedono interazione manuale — lo script lo segnala)
DATASETS = {
    "bacterial_fungicidal": {
        "name": "Bacterial-Fungicidal Vine Disease (Romania, 10k UAV)",
        "license": "CC-BY 4.0",
        "citation": "Marian et al. (2024). Bacterial-fungicidal vine disease detection with proximal aerial images. Heliyon.",
        "doi": "10.1016/j.heliyon.2024.e34919",
        "url": "https://www.sciencedirect.com/science/article/pii/S2405844024100485",
        "download_mode": "manual",  # richiede form su ScienceDirect
        "expected_format": "yolo_or_coco",
    },
    "tardif": {
        "name": "Tardif et al. Expertized Grapevine Disease DB (France, 1483 RGB)",
        "license": "CC-BY 4.0",
        "citation": "Tardif M. et al. (2023). An expertized grapevine disease image database... Data in Brief 48:109276.",
        "doi": "10.1016/j.dib.2023.109276",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10294005/",
        "download_mode": "manual",  # link dataset sul paper
        "expected_format": "coco_or_pascal_voc",
    },
    "ldd": {
        "name": "LDD Grape Leaf Diseases Dataset (1092 RGB + masks)",
        "license": "Academic / Open",
        "citation": "Pinto et al. (2022). LDD: A Dataset for Grape Diseases Object Detection and Instance Segmentation. arXiv:2206.10192.",
        "doi": "10.48550/arXiv.2206.10192",
        "url": "https://arxiv.org/abs/2206.10192",
        "download_mode": "manual",
        "expected_format": "coco",
    },
}

# Split train/valid/test (ratios)
SPLIT_RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}

# Estensioni immagini accettate
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logger con output sia su file che su console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "unification.log"

    logger = logging.getLogger("KE_Unifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DatasetStats:
    """Statistiche per dataset processato."""
    name: str
    images_found: int = 0
    images_processed: int = 0
    labels_processed: int = 0
    classes_seen: dict = field(default_factory=dict)
    unmapped_classes: set = field(default_factory=set)
    errors: list = field(default_factory=list)


@dataclass
class UnificationReport:
    """Report finale unificazione."""
    total_images: int = 0
    total_labels: int = 0
    per_dataset: dict = field(default_factory=dict)
    per_class_count: dict = field(default_factory=dict)
    split_counts: dict = field(default_factory=dict)


# =============================================================================
# DOWNLOAD MODULE
# =============================================================================

def print_manual_download_instructions(logger: logging.Logger, output_dir: Path):
    """Stampa istruzioni download manuale per dataset che non supportano auto."""
    logger.warning("=" * 70)
    logger.warning("DOWNLOAD MANUALE RICHIESTO")
    logger.warning("=" * 70)
    logger.warning("")
    logger.warning("I dataset Tier S richiedono download manuale dal paper originale.")
    logger.warning("Questo perché alcuni hanno form di consenso licenza o link")
    logger.warning("dinamici che cambiano periodicamente.")
    logger.warning("")
    logger.warning(f"Crea la cartella: {output_dir / 'raw'}")
    logger.warning("Poi scarica i seguenti dataset in sottocartelle dedicate:")
    logger.warning("")

    for key, ds in DATASETS.items():
        logger.warning(f"  [{key}]")
        logger.warning(f"    Nome   : {ds['name']}")
        logger.warning(f"    DOI    : {ds['doi']}")
        logger.warning(f"    URL    : {ds['url']}")
        logger.warning(f"    Licenza: {ds['license']}")
        logger.warning(f"    Cartella: {output_dir / 'raw' / key}/")
        logger.warning("")

    logger.warning("Una volta scaricati, rilancia lo script con il flag --process-only")
    logger.warning("=" * 70)


def check_raw_datasets_present(raw_dir: Path, logger: logging.Logger) -> dict:
    """Verifica quali dataset sono già stati scaricati manualmente."""
    found = {}
    for key in DATASETS.keys():
        ds_dir = raw_dir / key
        if ds_dir.exists() and any(ds_dir.iterdir()):
            # Cerca immagini ricorsivamente
            imgs = [p for p in ds_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
            found[key] = {
                "path": ds_dir,
                "image_count": len(imgs),
                "present": len(imgs) > 0,
            }
            logger.info(f"  [✓] {key}: trovate {len(imgs)} immagini in {ds_dir}")
        else:
            found[key] = {"path": ds_dir, "image_count": 0, "present": False}
            logger.warning(f"  [✗] {key}: non trovato in {ds_dir}")
    return found


# =============================================================================
# LABEL CONVERSION MODULE
# =============================================================================

def map_source_class_to_ke(source_class: str) -> Optional[int]:
    """Mappa nome classe sorgente → indice classe KE. Ritorna None se non mappato."""
    normalized = source_class.lower().strip().replace(" ", "_").replace("-", "_")
    return SOURCE_CLASS_MAPPING.get(normalized)


def yolo_bbox_to_absolute(yolo_bbox, img_w, img_h):
    """Converte bbox YOLO normalizzato in assoluto [xmin, ymin, xmax, ymax]."""
    xc, yc, w, h = yolo_bbox
    xmin = (xc - w / 2) * img_w
    ymin = (yc - h / 2) * img_h
    xmax = (xc + w / 2) * img_w
    ymax = (yc + h / 2) * img_h
    return [xmin, ymin, xmax, ymax]


def absolute_to_yolo_bbox(abs_bbox, img_w, img_h):
    """Converte bbox assoluto [xmin, ymin, xmax, ymax] in YOLO normalizzato."""
    xmin, ymin, xmax, ymax = abs_bbox
    xc = (xmin + xmax) / 2 / img_w
    yc = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return [xc, yc, w, h]


def convert_yolo_labels(src_label_path: Path, src_classes_names: list,
                         img_path: Path) -> tuple[list, set]:
    """
    Converte un file YOLO (.txt) dal mapping sorgente a quello KE.

    Returns:
        (lines_ke_format, unmapped_classes_seen)
    """
    out_lines = []
    unmapped = set()

    if not src_label_path.exists():
        return [], unmapped

    with open(src_label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            src_cls_idx = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]

            if src_cls_idx >= len(src_classes_names):
                continue
            src_cls_name = src_classes_names[src_cls_idx]
            ke_idx = map_source_class_to_ke(src_cls_name)

            if ke_idx is None:
                unmapped.add(src_cls_name)
                continue

            out_lines.append(f"{ke_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

    return out_lines, unmapped


def convert_coco_to_yolo(coco_json_path: Path, images_dir: Path,
                          output_labels_dir: Path) -> tuple[int, int, set]:
    """
    Converte annotazioni COCO JSON → file YOLO .txt + mapping classi a KE.

    Returns:
        (num_images, num_labels, unmapped_classes)
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup: image_id → filename, width, height
    img_lookup = {img["id"]: img for img in coco.get("images", [])}
    cat_lookup = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    # Raggruppa annotazioni per image_id
    ann_by_img = {}
    for ann in coco.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    unmapped = set()
    num_labels = 0

    for img_id, anns in ann_by_img.items():
        img_info = img_lookup.get(img_id)
        if not img_info:
            continue
        img_w, img_h = img_info.get("width"), img_info.get("height")
        if not img_w or not img_h:
            # Fallback: apri immagine per leggere dimensioni
            img_file = images_dir / img_info["file_name"]
            if not img_file.exists():
                continue
            try:
                with Image.open(img_file) as im:
                    img_w, img_h = im.size
            except Exception:
                continue

        lines = []
        for ann in anns:
            cat_name = cat_lookup.get(ann["category_id"], "")
            ke_idx = map_source_class_to_ke(cat_name)
            if ke_idx is None:
                unmapped.add(cat_name)
                continue

            # COCO bbox: [xmin, ymin, w, h] in absolute
            x, y, w, h = ann["bbox"]
            abs_bbox = [x, y, x + w, y + h]
            yolo_bbox = absolute_to_yolo_bbox(abs_bbox, img_w, img_h)

            # Clip a [0, 1]
            yolo_bbox = [max(0.0, min(1.0, v)) for v in yolo_bbox]
            lines.append(f"{ke_idx} {' '.join(f'{v:.6f}' for v in yolo_bbox)}")
            num_labels += 1

        if lines:
            label_file = output_labels_dir / (Path(img_info["file_name"]).stem + ".txt")
            with open(label_file, "w", encoding="utf-8") as fo:
                fo.write("\n".join(lines))

    return len(ann_by_img), num_labels, unmapped


# =============================================================================
# PROCESSING MODULE
# =============================================================================

def hash_filename(src_path: Path, dataset_key: str) -> str:
    """Genera nome file univoco per evitare collisioni tra dataset."""
    h = hashlib.md5(f"{dataset_key}_{src_path.name}".encode()).hexdigest()[:8]
    return f"{dataset_key}_{h}_{src_path.stem}{src_path.suffix.lower()}"


def process_dataset(dataset_key: str, raw_path: Path, staging_dir: Path,
                    logger: logging.Logger) -> DatasetStats:
    """
    Processa un singolo dataset: trova immagini + labels, converte in formato KE,
    stage in directory temporanea prima dello split finale.
    """
    stats = DatasetStats(name=dataset_key)
    stage_img_dir = staging_dir / "images"
    stage_lbl_dir = staging_dir / "labels"
    stage_img_dir.mkdir(parents=True, exist_ok=True)
    stage_lbl_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"→ Processing {dataset_key} da {raw_path}")

    # Cerca tutte le immagini ricorsivamente
    all_images = [p for p in raw_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
    stats.images_found = len(all_images)

    if not all_images:
        logger.warning(f"  Nessuna immagine trovata in {raw_path}")
        return stats

    # Strategia: cerca file data.yaml, classes.txt, annotations.json ricorsivamente
    # per capire il formato del dataset
    yaml_files = list(raw_path.rglob("data.yaml")) + list(raw_path.rglob("data.yml"))
    classes_txt = list(raw_path.rglob("classes.txt"))
    coco_jsons = [p for p in raw_path.rglob("*.json")
                  if any(kw in p.name.lower() for kw in ["annotation", "coco", "instances", "train", "valid", "test"])]

    src_class_names = []

    # Detect formato
    if yaml_files:
        logger.info(f"  Rilevato formato YOLO (data.yaml trovato)")
        with open(yaml_files[0], "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        src_class_names = yaml_data.get("names", [])
        if isinstance(src_class_names, dict):
            src_class_names = [src_class_names[i] for i in sorted(src_class_names.keys())]

        for img_path in tqdm(all_images, desc=f"  {dataset_key} (YOLO)"):
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                # Prova nella cartella labels/ parallela
                alt_label = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
                if alt_label.exists():
                    label_path = alt_label

            new_name = hash_filename(img_path, dataset_key)
            dst_img = stage_img_dir / new_name
            dst_lbl = stage_lbl_dir / (Path(new_name).stem + ".txt")

            try:
                shutil.copy2(img_path, dst_img)
                stats.images_processed += 1
            except Exception as e:
                stats.errors.append(f"Copy img fail: {img_path}: {e}")
                continue

            if label_path.exists():
                lines, unmapped = convert_yolo_labels(label_path, src_class_names, img_path)
                if lines:
                    with open(dst_lbl, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    stats.labels_processed += 1
                    for line in lines:
                        cls_idx = int(line.split()[0])
                        cls_name = KE_CLASSES[cls_idx]
                        stats.classes_seen[cls_name] = stats.classes_seen.get(cls_name, 0) + 1
                stats.unmapped_classes.update(unmapped)

    elif coco_jsons:
        logger.info(f"  Rilevato formato COCO ({len(coco_jsons)} JSON trovati)")

        # Prima copia tutte le immagini
        img_name_map = {}
        for img_path in tqdm(all_images, desc=f"  {dataset_key} (copying images)"):
            new_name = hash_filename(img_path, dataset_key)
            dst_img = stage_img_dir / new_name
            try:
                shutil.copy2(img_path, dst_img)
                img_name_map[img_path.name] = new_name
                stats.images_processed += 1
            except Exception as e:
                stats.errors.append(f"Copy img fail: {img_path}: {e}")

        # Poi processa i JSON COCO
        for coco_path in coco_jsons:
            try:
                # Trova la directory immagini associata al JSON
                possible_img_dirs = [coco_path.parent, coco_path.parent / "images",
                                     coco_path.parent.parent / "images"]
                img_dir = next((d for d in possible_img_dirs if d.exists()), coco_path.parent)

                n_imgs, n_lbls, unmapped = convert_coco_to_yolo(
                    coco_path, img_dir, stage_lbl_dir
                )
                stats.labels_processed += n_lbls
                stats.unmapped_classes.update(unmapped)
            except Exception as e:
                stats.errors.append(f"COCO parse fail: {coco_path}: {e}")

    else:
        logger.warning(f"  Formato non riconosciuto per {dataset_key}. "
                       f"Copio solo le immagini (nessuna label).")
        for img_path in tqdm(all_images, desc=f"  {dataset_key} (images only)"):
            new_name = hash_filename(img_path, dataset_key)
            dst_img = stage_img_dir / new_name
            try:
                shutil.copy2(img_path, dst_img)
                stats.images_processed += 1
            except Exception as e:
                stats.errors.append(f"Copy fail: {img_path}: {e}")

    logger.info(f"  {dataset_key}: {stats.images_processed} immagini, "
                f"{stats.labels_processed} file label processati")
    if stats.unmapped_classes:
        logger.warning(f"  Classi non mappate (ignorate): {sorted(stats.unmapped_classes)}")
        logger.warning(f"  → Aggiungi questi nomi a SOURCE_CLASS_MAPPING in alto allo script per recuperarli")

    return stats


# =============================================================================
# SPLIT MODULE
# =============================================================================

def split_dataset(staging_dir: Path, output_dir: Path,
                  ratios: dict, seed: int, logger: logging.Logger) -> dict:
    """Splitta il dataset stage in train/valid/test con ratio configurabile."""
    random.seed(seed)

    stage_img = staging_dir / "images"
    stage_lbl = staging_dir / "labels"

    all_images = sorted([p for p in stage_img.iterdir() if p.suffix.lower() in IMG_EXTS])
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * ratios["train"])
    n_valid = int(n * ratios["valid"])

    splits = {
        "train": all_images[:n_train],
        "valid": all_images[n_train:n_train + n_valid],
        "test": all_images[n_train + n_valid:],
    }

    counts = {}
    for split_name, imgs in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(imgs, desc=f"  Split → {split_name}"):
            shutil.move(str(img_path), img_out / img_path.name)
            lbl_path = stage_lbl / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.move(str(lbl_path), lbl_out / lbl_path.name)

        counts[split_name] = len(imgs)
        logger.info(f"  {split_name}: {len(imgs)} immagini")

    return counts


# =============================================================================
# OUTPUT FILES
# =============================================================================

def write_data_yaml(output_dir: Path):
    """Genera data.yaml pronto per training YOLOv8/v11."""
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(KE_CLASSES),
        "names": [KE_CLASSES[i] for i in sorted(KE_CLASSES.keys())],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def write_citations(output_dir: Path):
    """Genera CITATIONS.md con le citazioni obbligatorie CC-BY."""
    lines = [
        "# Citations — KestrelEye Unified Grapevine Dataset",
        "",
        "Questo dataset unificato integra dati provenienti da fonti scientifiche",
        "pubbliche con licenza CC-BY 4.0 o equivalente. Le citazioni sottostanti",
        "sono obbligatorie in qualsiasi pubblicazione, documento tecnico, o",
        "deliverable che usi modelli addestrati su questi dati.",
        "",
        "## Dataset sources",
        "",
    ]
    for key, ds in DATASETS.items():
        lines.append(f"### {ds['name']}")
        lines.append(f"- **Citation**: {ds['citation']}")
        lines.append(f"- **DOI**: {ds['doi']}")
        lines.append(f"- **License**: {ds['license']}")
        lines.append(f"- **URL**: {ds['url']}")
        lines.append("")

    lines += [
        "## Note d'uso",
        "",
        "Il **modello addestrato** su questi dati è di proprietà dell'utilizzatore",
        "(KestrelEye / HLD). La licenza CC-BY 4.0 impone solo l'attribuzione della",
        "fonte dati originale, non vincoli sul modello risultante.",
        "",
        "Per uso commerciale, mantenere queste citazioni nella documentazione",
        "tecnica del prodotto e/o nei paper associati.",
    ]

    cit_path = output_dir / "CITATIONS.md"
    with open(cit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return cit_path


def write_report(output_dir: Path, report: UnificationReport):
    """Scrive report JSON con statistiche di unificazione."""
    report_data = {
        "total_images": report.total_images,
        "total_labels": report.total_labels,
        "split_counts": report.split_counts,
        "per_class_count": report.per_class_count,
        "per_dataset": {
            k: {
                "images_found": v.images_found,
                "images_processed": v.images_processed,
                "labels_processed": v.labels_processed,
                "classes_seen": v.classes_seen,
                "unmapped_classes": sorted(list(v.unmapped_classes)),
                "errors_count": len(v.errors),
            }
            for k, v in report.per_dataset.items()
        },
        "ke_classes": KE_CLASSES,
    }
    report_path = output_dir / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return report_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KestrelEye Unified Grapevine Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:

  # Mostra istruzioni download manuale
  python build_unified_dataset.py --output ./data --show-instructions

  # Processa dataset già scaricati manualmente in ./data/raw/
  python build_unified_dataset.py --output ./data --process

  # Custom split e seed
  python build_unified_dataset.py --output ./data --process \\
         --train-ratio 0.75 --valid-ratio 0.15 --seed 42
"""
    )
    parser.add_argument("--output", "-o", type=str, default="./data",
                        help="Directory output base (default: ./data)")
    parser.add_argument("--process", action="store_true",
                        help="Processa dataset già scaricati in {output}/raw/")
    parser.add_argument("--show-instructions", action="store_true",
                        help="Mostra istruzioni per download manuale")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.output).resolve()
    raw_dir = base_dir / "raw"
    output_dir = base_dir / "grape_unified"
    staging_dir = base_dir / "_staging"

    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("KestrelEye — Unified Grapevine Dataset Builder v1.0")
    logger.info("=" * 70)

    # Branch 1: mostra solo istruzioni
    if args.show_instructions or (not args.process):
        logger.info("")
        logger.info("Verifico stato dataset raw...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        found = check_raw_datasets_present(raw_dir, logger)

        if not any(v["present"] for v in found.values()):
            print_manual_download_instructions(logger, base_dir)
            logger.info("")
            logger.info("Scarica i dataset indicati sopra, poi rilancia con --process")
            return 0
        else:
            logger.info("")
            logger.info(f"Trovati {sum(1 for v in found.values() if v['present'])}/{len(found)} dataset.")
            logger.info("Per processarli, usa: python build_unified_dataset.py --process")
            return 0

    # Branch 2: process mode
    logger.info("")
    logger.info("Modalità: PROCESS (processamento dataset scaricati)")
    logger.info(f"Raw dir   : {raw_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("")

    # Check split ratios
    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if test_ratio < 0 or test_ratio > 1:
        logger.error("Split ratios non validi (somma > 1 o < 0)")
        return 1
    ratios = {"train": args.train_ratio, "valid": args.valid_ratio, "test": test_ratio}
    logger.info(f"Split: train={ratios['train']:.0%}, valid={ratios['valid']:.0%}, test={ratios['test']:.0%}")

    # Verifica dataset presenti
    found = check_raw_datasets_present(raw_dir, logger)
    present = [k for k, v in found.items() if v["present"]]
    if not present:
        logger.error("Nessun dataset trovato in ./raw/. Scarica prima i dati.")
        print_manual_download_instructions(logger, base_dir)
        return 1

    logger.info(f"\nDataset disponibili: {present}")

    # Clean output se esiste
    if output_dir.exists():
        logger.warning(f"Output dir esistente: rimuovo {output_dir}")
        shutil.rmtree(output_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Reinit logger dopo eventuale pulizia
    logger = setup_logging(output_dir)

    # Processa ogni dataset presente
    report = UnificationReport()
    for key in present:
        ds_stats = process_dataset(key, found[key]["path"], staging_dir, logger)
        report.per_dataset[key] = ds_stats

    # Split
    logger.info("\nSplit train/valid/test...")
    split_counts = split_dataset(staging_dir, output_dir, ratios, args.seed, logger)
    report.split_counts = split_counts

    # Totali
    report.total_images = sum(split_counts.values())
    for ds_stats in report.per_dataset.values():
        for cls_name, count in ds_stats.classes_seen.items():
            report.per_class_count[cls_name] = report.per_class_count.get(cls_name, 0) + count
        report.total_labels += ds_stats.labels_processed

    # Cleanup staging
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # Genera file output
    logger.info("\nGenerazione file output...")
    yaml_path = write_data_yaml(output_dir)
    logger.info(f"  data.yaml     : {yaml_path}")
    cit_path = write_citations(output_dir)
    logger.info(f"  CITATIONS.md  : {cit_path}")
    report_path = write_report(output_dir, report)
    logger.info(f"  dataset_report.json: {report_path}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("UNIFICAZIONE COMPLETATA")
    logger.info("=" * 70)
    logger.info(f"  Immagini totali : {report.total_images}")
    logger.info(f"  Label totali    : {report.total_labels}")
    logger.info(f"  Split           : {split_counts}")
    logger.info(f"  Classi presenti : {list(report.per_class_count.keys())}")
    logger.info("")
    logger.info("Per lanciare il training:")
    logger.info(f"  cd {output_dir.parent}")
    logger.info(f"  yolo detect train data={yaml_path} model=yolov8s.pt epochs=100 imgsz=640 device=mps")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
