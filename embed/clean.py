"""
Remove rich nodes that lack required attributes, then prune orphaned sparse nodes.

Supports the project's internal JSON format (json_data/) and GraphML files.
Prints a before/after report. Writes the filtered graph only when --output is given.

Usage
-----
    # Internal JSON format
    python -m embed.clean json_data/dagstuhl.json --require data title

    # GraphML — specify the partition attribute value that identifies rich nodes
    python -m embed.clean dagstuhl.graphml --require seminar_keywords --rich-value event

    # Save the result
    python -m embed.clean dagstuhl.graphml --require seminar_keywords \\
        --rich-value event --output dagstuhl_clean.graphml

    # CSV — drop rows missing any of the listed columns
    python -m embed.clean netflix_enriched.csv --require overview genres poster_path \\
        --output netflix_clean.csv
"""
import argparse
import json
import os

import networkx as nx


# ---------------------------------------------------------------------------
# JSON dict format  (json_data/*.json)
# ---------------------------------------------------------------------------

def filter_graph(
    data: dict,
    required_attrs: list[str],
    rich_key: str = 'rich',
    sparse_key: str = 'sparse',
) -> tuple[dict, dict]:
    """
    Remove rich nodes missing required attributes, then prune orphaned sparse nodes.

    A rich node is removed if any attribute in required_attrs is absent, None, or an
    empty string / empty collection. Sparse nodes with no remaining rich neighbour are
    also removed.

    Parameters
    ----------
    data : dict
        Graph in the json_data format: {rich_key: [...], sparse_key: [...]}.
        Each rich node dict must carry an 's_ids' list (plain IDs or [id, weight] pairs).
    required_attrs : list[str]
        Attribute keys every rich node must have a non-empty value for.
    rich_key : str
        Top-level key for the rich node list (default 'rich').
    sparse_key : str
        Top-level key for the sparse node list (default 'sparse').

    Returns
    -------
    (filtered_data, report)
    """
    def _is_missing(v) -> bool:
        if isinstance(v, str):
            return not v.strip()
        return v is None or v == [] or v == {}

    def _edge_count(nodes: list) -> int:
        return sum(len(n.get('s_ids', [])) for n in nodes)

    def _sparse_id(s) -> str:
        return s[0] if isinstance(s, list) else s

    report = {
        'rich_before':   len(data[rich_key]),
        'sparse_before': len(data[sparse_key]),
        'edges_before':  _edge_count(data[rich_key]),
        'removed_per_attr': {},
    }

    kept_rich = list(data[rich_key])
    for attr in required_attrs:
        before = len(kept_rich)
        kept_rich = [n for n in kept_rich if not _is_missing(n.get(attr))]
        report['removed_per_attr'][attr] = before - len(kept_rich)

    connected_ids = {
        _sparse_id(s)
        for n in kept_rich
        for s in n.get('s_ids', [])
    }
    kept_sparse = [n for n in data[sparse_key] if n['id'] in connected_ids]

    report['rich_after']     = len(kept_rich)
    report['sparse_after']   = len(kept_sparse)
    report['edges_after']    = _edge_count(kept_rich)
    report['sparse_removed'] = report['sparse_before'] - report['sparse_after']

    filtered = {**data, rich_key: kept_rich, sparse_key: kept_sparse}
    return filtered, report


# ---------------------------------------------------------------------------
# NetworkX / GraphML format
# ---------------------------------------------------------------------------

def filter_nx_graph(
    G: nx.Graph,
    required_attrs: list[str],
    required_values: dict[str, str] | None = None,
    rich_value: str = 'rich',
    partition_key: str = 'type',
    min_sparse_degree: int = 1,
) -> tuple[nx.Graph, dict]:
    """
    Remove rich nodes missing required attributes or not matching required values,
    then prune isolated nodes.

    A node is "rich" when G.nodes[n][partition_key] == rich_value. After rich nodes
    are removed, any remaining node with degree 0 (orphaned sparse node) is also
    removed. Modifies G in place.

    Parameters
    ----------
    G : nx.Graph
        Bipartite NetworkX graph with a node attribute identifying partitions.
    required_attrs : list[str]
        Attribute keys every rich node must have a non-empty value for.
    required_values : dict[str, str] or None
        Attribute keys mapped to exact required values. A rich node is removed if
        its value for that attribute does not equal the specified string.
    rich_value : str
        Value of partition_key that identifies rich nodes (e.g. 'event', 'rich').
    partition_key : str
        Node attribute key used to identify which partition a node belongs to.
    min_sparse_degree : int
        Minimum degree a sparse node must have after rich-node filtering to be kept.
        Default 1 removes only isolates. Set to 2 to also remove degree-1 sparse nodes.

    Returns
    -------
    (G, report)
    """
    def _is_missing(v) -> bool:
        if isinstance(v, str):
            return not v.strip()
        return v is None

    rich_nodes = [n for n, d in G.nodes(data=True) if d.get(partition_key) == rich_value]

    report = {
        'rich_before':   len(rich_nodes),
        'sparse_before': G.number_of_nodes() - len(rich_nodes),
        'edges_before':  G.number_of_edges(),
        'removed_per_attr': {},
    }

    # Build removal set attribute by attribute for accurate per-attr counts
    remaining = set(rich_nodes)
    for attr in required_attrs:
        before = len(remaining)
        remaining = {n for n in remaining if not _is_missing(G.nodes[n].get(attr))}
        report['removed_per_attr'][attr] = before - len(remaining)

    for attr, expected in (required_values or {}).items():
        before = len(remaining)
        remaining = {n for n in remaining if G.nodes[n].get(attr) == expected}
        report['removed_per_attr'][f'{attr}={expected}'] = before - len(remaining)

    G.remove_nodes_from(set(rich_nodes) - remaining)

    low_degree_sparse = [
        n for n, d in G.nodes(data=True)
        if d.get(partition_key) != rich_value and G.degree(n) < min_sparse_degree
    ]
    G.remove_nodes_from(low_degree_sparse)

    report['rich_after']     = sum(1 for n in G.nodes() if G.nodes[n].get(partition_key) == rich_value)
    report['sparse_after']   = G.number_of_nodes() - report['rich_after']
    report['edges_after']    = G.number_of_edges()
    report['sparse_removed'] = len(low_degree_sparse)

    return G, report


# ---------------------------------------------------------------------------
# CSV format  (flat row lists)
# ---------------------------------------------------------------------------

def filter_csv(
    rows: list[dict],
    required_attrs: list[str],
) -> tuple[list[dict], dict]:
    """
    Remove rows where any required attribute is absent, None, or blank/whitespace.

    Parameters
    ----------
    rows : list[dict]
        Rows as returned by csv.DictReader.
    required_attrs : list[str]
        Column names every row must have a non-empty value for.

    Returns
    -------
    (filtered_rows, report)
    """
    def _is_missing(v) -> bool:
        if isinstance(v, str):
            return not v.strip()
        return v is None

    report: dict = {
        'rows_before': len(rows),
        'removed_per_attr': {},
    }

    kept = list(rows)
    for attr in required_attrs:
        before = len(kept)
        kept = [r for r in kept if not _is_missing(r.get(attr))]
        report['removed_per_attr'][attr] = before - len(kept)

    report['rows_after'] = len(kept)
    return kept, report


def _print_csv_report(path: str, report: dict) -> None:
    print(f"\nFile   : {os.path.basename(path)}")
    print(f"\nBefore : {report['rows_before']:>6,} rows")
    print(f"\nRemoved")
    any_removed = False
    for attr, count in report['removed_per_attr'].items():
        if count:
            print(f"  {count:>4,} row{'s' if count != 1 else ''} missing '{attr}'")
            any_removed = True
    if not any_removed:
        print("  (nothing removed)")
    print(f"\nAfter  : {report['rows_after']:>6,} rows")
    print()


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------

def _print_report(
    path: str,
    required_attrs: list[str],
    report: dict,
    rich_label: str = 'Rich',
    sparse_label: str = 'Sparse',
) -> None:
    print(f"\nGraph  : {os.path.basename(path)}")
    print(f"Attrs  : {required_attrs or '(none)'}")

    print(f"\nBefore")
    print(f"  {rich_label + ' nodes':<14}: {report['rich_before']:>6,}")
    print(f"  {sparse_label + ' nodes':<14}: {report['sparse_before']:>6,}")
    print(f"  {'Edges':<14}: {report['edges_before']:>6,}")

    print(f"\nRemoved")
    any_removed = False
    for attr, count in report['removed_per_attr'].items():
        if count:
            print(f"  {count:>4,} {rich_label.lower()} node{'s' if count != 1 else ''} missing '{attr}'")
            any_removed = True
    n = report['sparse_removed']
    if n:
        print(f"  {n:>4,} orphaned {sparse_label.lower()} node{'s' if n != 1 else ''}")
        any_removed = True
    if not any_removed:
        print(f"  (nothing removed)")

    print(f"\nAfter")
    print(f"  {rich_label + ' nodes':<14}: {report['rich_after']:>6,}")
    print(f"  {sparse_label + ' nodes':<14}: {report['sparse_after']:>6,}")
    print(f"  {'Edges':<14}: {report['edges_after']:>6,}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Remove rich nodes missing required attributes and prune orphaned '
            'sparse nodes. Prints a before/after report.'
        )
    )
    parser.add_argument(
        'input',
        help='Path to the graph file (.json or .graphml).',
    )
    parser.add_argument(
        '--require', nargs='+', metavar='ATTR', default=[],
        help='Attribute keys every rich node must have, e.g. --require data title.',
    )
    parser.add_argument(
        '--require-value', nargs='+', metavar='ATTR=VALUE', default=[],
        help=(
            'Exact value constraints in ATTR=VALUE form. A rich node is removed if '
            'its attribute does not equal VALUE. E.g. --require-value seminar_type="Dagstuhl Seminar".'
        ),
    )
    parser.add_argument(
        '--output', default=None, metavar='PATH',
        help='Write the cleaned graph here. If omitted, only the report is printed.',
    )
    parser.add_argument(
        '--rich-value', default=None, metavar='VALUE',
        help=(
            'Node attribute value identifying rich nodes. '
            'Defaults to "rich" for .json and must be set for .graphml '
            '(e.g. --rich-value event for Dagstuhl).'
        ),
    )
    parser.add_argument(
        '--partition-key', default='type',
        help='Node attribute key that identifies partitions (default: type). GraphML only.',
    )
    parser.add_argument(
        '--min-sparse-degree', type=int, default=1, metavar='N',
        help='Remove sparse nodes with degree below N after rich-node filtering (default: 1, keeps all connected nodes).',
    )
    parser.add_argument('--rich-key',   default='rich',   help='Top-level JSON key for rich nodes (default: rich).')
    parser.add_argument('--sparse-key', default='sparse', help='Top-level JSON key for sparse nodes (default: sparse).')

    args = parser.parse_args()
    ext = os.path.splitext(args.input)[1].lower()

    # Parse --require-value ATTR=VALUE pairs
    required_values: dict[str, str] = {}
    for item in args.require_value:
        if '=' not in item:
            parser.error(f"--require-value items must be in ATTR=VALUE form, got: {item!r}")
        attr, _, val = item.partition('=')
        required_values[attr] = val

    if ext == '.graphml':
        rich_value = args.rich_value
        if rich_value is None:
            parser.error('--rich-value is required for GraphML inputs (e.g. --rich-value event).')

        G = nx.read_graphml(args.input)
        G, report = filter_nx_graph(
            G,
            required_attrs=args.require,
            required_values=required_values,
            rich_value=rich_value,
            partition_key=args.partition_key,
            min_sparse_degree=args.min_sparse_degree,
        )
        _print_report(args.input, args.require + [f'{k}={v}' for k, v in required_values.items()], report,
                      rich_label=rich_value.capitalize(),
                      sparse_label='Sparse')
        if args.output:
            dirpath = os.path.dirname(args.output)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            nx.write_graphml(G, args.output)
            print(f"Saved to: {args.output}")
        else:
            print("(dry run — pass --output PATH to save the cleaned graph)")

    elif ext == '.json':
        rich_value = args.rich_value or 'rich'

        with open(args.input, 'r') as f:
            data = json.load(f)

        filtered, report = filter_graph(
            data,
            required_attrs=args.require,
            rich_key=args.rich_key,
            sparse_key=args.sparse_key,
        )
        _print_report(args.input, args.require, report,
                      rich_label=rich_value.capitalize(),
                      sparse_label=args.sparse_key.capitalize())
        if args.output:
            dirpath = os.path.dirname(args.output)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(filtered, f, indent=4)
            print(f"Saved to: {args.output}")
        else:
            print("(dry run — pass --output PATH to save the cleaned graph)")

    elif ext == '.csv':
        import csv

        with open(args.input, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        filtered, report = filter_csv(rows, required_attrs=args.require)
        _print_csv_report(args.input, report)

        if args.output:
            dirpath = os.path.dirname(args.output)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(args.output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered)
            print(f"Saved to: {args.output}")
        else:
            print("(dry run — pass --output PATH to save the cleaned CSV)")

    else:
        parser.error(f"Unsupported file format '{ext}'. Expected .json, .graphml, or .csv.")
