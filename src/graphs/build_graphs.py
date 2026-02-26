import argparse
import json
from pathlib import Path

from src.graphs.ckg import build_ckg
from src.graphs.cpg import build_cpg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CKG and CPG graphs")
    parser.add_argument("--meta-dir", default="data/corpus_meta")
    parser.add_argument("--graphs-dir", default="graphs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckg = build_ckg(meta_dir=Path(args.meta_dir), out_dir=Path(args.graphs_dir))
    cpg = build_cpg(chunk_manifest_path=Path(args.meta_dir) / "chunk_manifest.json", out_dir=Path(args.graphs_dir))
    summary = {
        "ckg_file": str((Path(args.graphs_dir) / "ckg.json").as_posix()),
        "cpg_file": str((Path(args.graphs_dir) / "cpg.json").as_posix()),
        "ckg_node_count": len(ckg.get("nodes", [])),
        "ckg_edge_count": len(ckg.get("edges", [])),
        "cpg_node_count": len(cpg.get("nodes", [])),
        "cpg_edge_count": len(cpg.get("edges", [])),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
