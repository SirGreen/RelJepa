"""Extract TVE's released data.zip (Google Drive, see TVE README) into the data directory.

Only the requested datasets are extracted; macOS metadata (__MACOSX/, .DS_Store) is skipped.
Usage: python extract_tve_data.py <data.zip> <dest_dir> [--datasets rel-hm ...]
"""
import argparse
import os
import shutil
import zipfile


def dataset_of(rel_path):
    """Return the RelBench dataset a zip member belongs to, or None for shared files."""
    parts = rel_path.split("/")
    if parts[0].startswith("rel-"):
        return parts[0]
    if parts[0] == "relbench" and len(parts) > 1:
        return parts[1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("dest")
    parser.add_argument("--datasets", nargs="+", default=["rel-hm"])
    args = parser.parse_args()

    n_files = n_bytes = 0
    with zipfile.ZipFile(args.zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            assert name.startswith("data/"), f"unexpected member: {name}"
            rel = name[len("data/"):]
            ds = dataset_of(rel)
            if ds is not None and ds not in args.datasets:
                continue
            out = os.path.join(args.dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            n_files += 1
            n_bytes += info.file_size
            print(f"{info.file_size:>14,d}  {rel}")
    print(f"extracted {n_files} files, {n_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
