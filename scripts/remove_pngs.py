"""Cleanup script: remove PNG images from static exports and sanitize JSON indexes.

Usage:
    python scripts/remove_pngs.py --site-dir scripts/static_site

Behavior:
- Recursively deletes all .png files under <site-dir>/images
- For each JSON file under <site-dir>, removes list entries or dict values that point to .png files that no longer exist.
- Makes a .bak copy of modified JSON files.
"""

import argparse
from pathlib import Path
import json
import shutil


def delete_pngs(images_dir: Path):
    removed = 0
    if not images_dir.exists():
        print(f"Images dir {images_dir} does not exist, nothing to do.")
        return removed
    for p in images_dir.rglob('*.png'):
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print(f"Warning: could not remove {p}: {e}")
    print(f"Deleted {removed} .png files under {images_dir}")
    return removed


def sanitize_json_files(site_dir: Path):
    json_files = list(site_dir.rglob('*.json'))
    changed = 0
    for jf in json_files:
        try:
            text = jf.read_text(encoding='utf-8')
            if '.png' not in text:
                continue
            # Load JSON
            data = json.loads(text)
        except Exception as e:
            print(f"Skipping {jf} (not JSON or read error): {e}")
            continue

        orig = json.dumps(data, ensure_ascii=False)
        modified = _remove_png_references(data, site_dir)
        if json.dumps(modified, ensure_ascii=False) != orig:
            bak = jf.with_suffix(jf.suffix + '.bak')
            try:
                shutil.copy(jf, bak)
            except Exception:
                pass
            jf.write_text(json.dumps(modified, ensure_ascii=False, indent=2), encoding='utf-8')
            changed += 1
            print(f"Sanitized JSON file: {jf} (backup -> {bak})")
    print(f"Sanitized {changed} JSON files")
    return changed


def _remove_png_references(obj, site_dir: Path):
    """Recursively remove or replace PNG references in JSON-compatible objects.

    Rules:
    - If a string endswith .png and a corresponding .webp of same basename exists, replace with .webp
    - If a string endswith .png and no .webp exists, remove the value (if in list) or delete key (if in dict)
    - For nested lists/dicts, operate recursively
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            nv = _remove_png_references(v, site_dir)
            # If value reduced to None or empty list/dict, keep it (we only remove explicit png strings)
            if nv is not None:
                new[k] = nv
        return new
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            nv = _remove_png_references(item, site_dir)
            if nv is None:
                continue
            # Skip empty strings
            if nv == '':
                continue
            new_list.append(nv)
        return new_list
    elif isinstance(obj, str):
        s = obj
        if s.lower().endswith('.png'):
            base = Path(s)
            webp_path = base.with_suffix('.webp')
            # If the referenced path is relative inside site_dir, check existence
            candidate = (site_dir / webp_path) if not webp_path.is_absolute() else webp_path
            if candidate.exists():
                return str(webp_path)
            # Windows-safe fallback: check for 'upper-{basename}.webp' or 'lower-{basename}.webp'
            try:
                name_no_ext = base.stem
                # build upper-prefixed candidate
                upper_name = f"upper-{name_no_ext}.webp"
                upper_candidate = (site_dir / base.parent / upper_name) if not base.is_absolute() else Path(base.parent) / upper_name
                if upper_candidate.exists():
                    rel = upper_candidate.relative_to(site_dir)
                    return str(rel)
                # build lower-prefixed candidate
                lower_name = f"lower-{name_no_ext}.webp"
                lower_candidate = (site_dir / base.parent / lower_name) if not base.is_absolute() else Path(base.parent) / lower_name
                if lower_candidate.exists():
                    rel = lower_candidate.relative_to(site_dir)
                    return str(rel)

                # Also check font-prefixed variants like 'italic_upper-{basename}.webp' and 'roman_lower-{basename}.webp'
                for ft in ('roman', 'italic'):
                    ft_up = f"{ft}_upper-{name_no_ext}.webp"
                    ft_up_candidate = (site_dir / base.parent / ft_up) if not base.is_absolute() else Path(base.parent) / ft_up
                    if ft_up_candidate.exists():
                        rel = ft_up_candidate.relative_to(site_dir)
                        return str(rel)
                    ft_low = f"{ft}_lower-{name_no_ext}.webp"
                    ft_low_candidate = (site_dir / base.parent / ft_low) if not base.is_absolute() else Path(base.parent) / ft_low
                    if ft_low_candidate.exists():
                        rel = ft_low_candidate.relative_to(site_dir)
                        return str(rel)
            except Exception:
                pass
            # No webp available -> remove reference
            return None
        else:
            return obj
    else:
        return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove PNGs & sanitize JSON references in a static site')
    parser.add_argument('--site-dir', default='scripts/static_site', help='Path to the static site directory')
    args = parser.parse_args()

    site = Path(args.site_dir)
    images_dir = site / 'images'

    delete_pngs(images_dir)
    sanitize_json_files(site)
