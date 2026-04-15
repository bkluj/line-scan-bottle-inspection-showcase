from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"

MAX_DEPTH = 3
MAX_IMAGE_FILES_PER_DIR = 2
COMMENT_SPACING = 4

IGNORE_NAMES = {
    ".git",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".DS_Store",
    "artifacts",
    "results",
    "results_tiled",
    "dataset_anomalib_tiled",
    "dataset_anomalib",
    "dataset_preprocessed",
    "t_artifacts",
    "t_results",
    "t_results_tiled",
    "t_dataset_anomalib_tiled",
    "t_dataset_anomalib",
    "t_dataset_preprocessed",
}

DESCRIPTIONS = {
    "config/": "central configuration",
    "data/": "datasets, artifacts and model outputs",
    "gui/": "desktop GUI",
    "gui/auth/": "authentication and local user access",
    "gui/controllers/": "GUI workflow and business logic",
    "gui/views/": "application windows and settings dialogs",
    "gui/widgets/": "reusable GUI widgets",
    "scripts/": "thin entrypoints",
    "tests/": "unit tests",
    "vision/": "core computer vision logic",
    "vision/calibration/": "tile-level calibration and score normalization",
    "vision/dataset/": "dataset preparation and tiling builders",
    "vision/defects/": "synthetic defect generators",
    "vision/inference/": "tile-based inference pipeline",
    "vision/preprocessing/": "image resizing, padding and tiling utilities",
    "vision/visualisation/": "stitching and result visualisation",
}

IGNORE_PREFIXES = (".coverage",)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

TREE_START = "<!-- TREE_START -->"
TREE_END = "<!-- TREE_END -->"


def debug(msg: str) -> None:
    print(f"[DEBUG] {msg}")


def should_ignore(path: Path) -> bool:
    name = path.name

    if name in IGNORE_NAMES:
        return True

    if any(name.startswith(prefix) for prefix in IGNORE_PREFIXES):
        return True

    if name.endswith(".pyc"):
        return True

    return False


def safe_iterdir(path: Path) -> list[Path]:
    try:
        items = list(path.iterdir())
        debug(f"Listed {len(items)} items in: {path}")
        return items
    except Exception as exc:
        print(f"[ERROR] Cannot list directory: {path}")
        print(f"[ERROR] Reason: {exc}")
        return []


def split_entries(path: Path) -> tuple[list[Path], list[Path], list[Path]]:
    dirs: list[Path] = []
    image_files: list[Path] = []
    other_files: list[Path] = []

    entries = sorted(
        safe_iterdir(path),
        key=lambda p: (not p.is_dir(), p.name.lower())
    )

    for entry in entries:
        if should_ignore(entry):
            debug(f"Ignored: {entry}")
            continue

        if entry.is_dir():
            dirs.append(entry)
        elif entry.suffix.lower() in IMAGE_EXTS:
            image_files.append(entry)
        else:
            other_files.append(entry)

    debug(
        f"Split for {path}: "
        f"{len(dirs)} dirs, {len(image_files)} image files, {len(other_files)} other files"
    )
    return dirs, image_files, other_files


def build_tree(path: Path, prefix: str = "", depth: int = 0) -> list[tuple[str, str]]:
    """
    Returns list of tuples:
    (rendered_tree_line_without_comment, relative_path_for_description_or_empty)
    """
    debug(f"build_tree(path={path}, depth={depth})")

    if depth > MAX_DEPTH:
        debug(f"Max depth exceeded at: {path}")
        return []

    dirs, image_files, other_files = split_entries(path)

    shown_images = image_files[:MAX_IMAGE_FILES_PER_DIR]
    hidden_image_count = max(0, len(image_files) - MAX_IMAGE_FILES_PER_DIR)

    entries = dirs + other_files + shown_images
    lines: list[tuple[str, str]] = []

    total_items = len(entries) + (1 if hidden_image_count > 0 else 0)
    debug(f"Total visible items for {path}: {total_items}")

    for idx, entry in enumerate(entries):
        is_last = idx == len(entries) - 1 and hidden_image_count == 0
        connector = "└── " if is_last else "├── "

        name = entry.name + ("/" if entry.is_dir() else "")
        rel_path = entry.relative_to(ROOT).as_posix()
        if entry.is_dir():
            rel_path += "/"

        line = prefix + connector + name
        lines.append((line, rel_path))
        debug(f"Added line: {line} | rel_path={rel_path}")

        if entry.is_dir() and depth < MAX_DEPTH:
            extension = "    " if is_last else "│   "
            child_lines = build_tree(entry, prefix + extension, depth + 1)
            lines.extend(child_lines)

    if hidden_image_count > 0:
        line = prefix + "└── ..."
        lines.append((line, ""))
        debug(f"Added ellipsis line for {path}: {line}")

    return lines


def align_comments(lines_with_paths: list[tuple[str, str]]) -> list[str]:
    """
    Align inline comments automatically using spaces.
    """
    comment_candidates = [
        (line, rel_path, DESCRIPTIONS[rel_path])
        for line, rel_path in lines_with_paths
        if rel_path in DESCRIPTIONS
    ]

    if not comment_candidates:
        return [line for line, _ in lines_with_paths]

    max_line_len = max(len(line) for line, _, _ in comment_candidates)

    output: list[str] = []
    for line, rel_path in lines_with_paths:
        desc = DESCRIPTIONS.get(rel_path)
        if desc:
            padding = " " * (max_line_len - len(line) + COMMENT_SPACING)
            output.append(f"{line}{padding}# {desc}")
        else:
            output.append(line)

    return output


def generate_tree_text() -> str:
    debug(f"Generating tree from ROOT: {ROOT}")

    if not ROOT.exists():
        raise FileNotFoundError(f"Project root does not exist: {ROOT}")

    root_line = ROOT.name + "/"
    built = build_tree(ROOT)
    aligned = align_comments(built)

    lines = [root_line] + aligned
    tree_text = "\n".join(lines)

    debug("Tree generation finished.")
    return tree_text


def read_readme() -> str:
    debug(f"Reading README: {README_PATH}")

    if not README_PATH.exists():
        raise FileNotFoundError(f"README.md does not exist: {README_PATH}")

    content = README_PATH.read_text(encoding="utf-8")
    debug(f"README length: {len(content)} chars")
    return content


def update_readme(tree_text: str) -> None:
    content = read_readme()

    if TREE_START not in content:
        raise RuntimeError(f"Missing marker in README: {TREE_START}")

    if TREE_END not in content:
        raise RuntimeError(f"Missing marker in README: {TREE_END}")

    start_index = content.index(TREE_START)
    end_index = content.index(TREE_END)

    debug(f"TREE_START index: {start_index}")
    debug(f"TREE_END index: {end_index}")

    new_block = f"{TREE_START}\n```text\n{tree_text}\n```\n{TREE_END}"

    before = content[:start_index]
    after = content[end_index + len(TREE_END):]
    updated = before + new_block + after

    README_PATH.write_text(updated, encoding="utf-8")
    debug("README updated successfully.")


def main() -> None:
    print("[INFO] Starting README tree update...")
    print(f"[INFO] ROOT: {ROOT}")
    print(f"[INFO] README: {README_PATH}")
    print(f"[INFO] MAX_DEPTH: {MAX_DEPTH}")
    print(f"[INFO] MAX_IMAGE_FILES_PER_DIR: {MAX_IMAGE_FILES_PER_DIR}")

    try:
        tree_text = generate_tree_text()
        print("\n[INFO] Generated tree preview:\n")
        print(tree_text[:2000])
        if len(tree_text) > 2000:
            print("\n[INFO] ... preview truncated ...\n")

        update_readme(tree_text)
        print("[INFO] README.md updated.")
    except Exception as exc:
        print(f"[ERROR] Failed to update README: {exc}")
        raise


if __name__ == "__main__":
    main()