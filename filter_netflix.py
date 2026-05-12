"""
Downsample an existing Netflix GraphML to ~max_movies + ~max_users for browser use.

Selection strategy (greedy, 2 streaming passes):
  Pass 1 — count movie degrees from edges; select top --max-movies by degree.
  Pass 2 — buffer edges for selected movies, count user degrees within them;
            select top --max-users by degree; write filtered GraphML.
  If the resulting edge count exceeds --max-edges, edges are randomly sampled
  down to that limit (uniform over movie-user pairs).

Usage
-----
    python filter_netflix.py
    python filter_netflix.py --input netflix-filtered.graphml --max-movies 1000 --max-users 5000 --max-edges 10000
    python filter_netflix.py --output netflix-tiny.graphml --max-movies 500 --max-users 2000 --max-edges 5000
"""
import argparse
import html
import random
from collections import defaultdict
from xml.etree.ElementTree import iterparse

NS = '{http://graphml.graphdrawing.org/graphml}'

MOVIE_ATTRS = [
    'title', 'year', 'data', 'genres', 'poster_path',
    'tmdb_id', 'cast', 'director', 'keywords',
    'tmdb_vote_average', 'tmdb_vote_count',
]


def _parse_node(elem) -> tuple[str, dict]:
    nid = elem.get('id', '')
    data = {}
    for child in elem:
        if child.tag == f'{NS}data':
            data[child.get('key', '')] = child.text or ''
    return nid, data


def pass1_movie_degrees(path: str) -> tuple[dict[str, dict], dict[str, int]]:
    """Collect movie metadata and count edges per movie."""
    movie_meta: dict[str, dict] = {}
    movie_degrees: dict[str, int] = defaultdict(int)
    n_edges = 0

    print(f"Pass 1: counting movie degrees from {path} ...", flush=True)
    for event, elem in iterparse(path, events=('end',)):
        tag = elem.tag
        if tag == f'{NS}node':
            nid, data = _parse_node(elem)
            if data.get('type') == 'movie':
                movie_meta[nid] = data
            elem.clear()
        elif tag == f'{NS}edge':
            src = elem.get('source', '')
            if src.startswith('r_'):          # movie→user direction
                movie_degrees[src] += 1
            elif elem.get('target', '').startswith('r_'):
                movie_degrees[elem.get('target')] += 1
            n_edges += 1
            if n_edges % 10_000_000 == 0:
                print(f"  {n_edges:,} edges scanned", flush=True)
            elem.clear()

    print(f"  {len(movie_meta):,} movies, {n_edges:,} edges total")
    return movie_meta, dict(movie_degrees)


def pass2_collect_and_write(
    path: str,
    selected_movies: set[str],
    movie_meta: dict[str, dict],
    max_users: int,
    max_edges: int,
    output: str,
) -> tuple[int, int, int]:
    """Buffer edges for selected movies, pick top users, write GraphML."""
    user_degrees: dict[str, int] = defaultdict(int)
    buffered_edges: list[tuple[str, str]] = []
    n_scanned = 0

    print(f"Pass 2: collecting edges for {len(selected_movies):,} selected movies ...", flush=True)
    for event, elem in iterparse(path, events=('end',)):
        if elem.tag == f'{NS}edge':
            src = elem.get('source', '')
            tgt = elem.get('target', '')
            movie_id = src if src in selected_movies else (tgt if tgt in selected_movies else None)
            if movie_id is not None:
                user_id = tgt if movie_id == src else src
                user_degrees[user_id] += 1
                buffered_edges.append((movie_id, user_id))
            n_scanned += 1
            if n_scanned % 10_000_000 == 0:
                print(f"  {n_scanned:,} edges scanned, {len(buffered_edges):,} buffered", flush=True)
            elem.clear()

    print(f"  {len(user_degrees):,} distinct users reached; selecting top {max_users:,} ...")
    selected_users = set(
        uid for uid, _ in sorted(user_degrees.items(), key=lambda x: -x[1])[:max_users]
    )

    final_edges = [(m, u) for m, u in buffered_edges if u in selected_users]
    if len(final_edges) > max_edges:
        print(f"  {len(final_edges):,} edges → sampling down to {max_edges:,} ...")
        final_edges = random.sample(final_edges, max_edges)

    # Only keep users that still appear after edge sampling
    selected_users = {u for _, u in final_edges}

    print(f"Writing {output} ...")
    with open(output, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        f.write('  <key id="type" for="node" attr.name="type" attr.type="string"/>\n')
        for attr in MOVIE_ATTRS:
            f.write(f'  <key id="{attr}" for="node" attr.name="{attr}" attr.type="string"/>\n')
        f.write('  <graph id="G" edgedefault="undirected">\n')

        for movie_id in sorted(selected_movies, key=lambda x: int(x[2:])):
            f.write(f'    <node id="{html.escape(movie_id)}">\n')
            f.write('      <data key="type">movie</data>\n')
            for attr in MOVIE_ATTRS:
                val = movie_meta.get(movie_id, {}).get(attr, '')
                if val:
                    f.write(f'      <data key="{attr}">{html.escape(val)}</data>\n')
            f.write('    </node>\n')

        for user_id in sorted(selected_users, key=lambda x: int(x[2:])):
            f.write(f'    <node id="{html.escape(user_id)}">\n')
            f.write('      <data key="type">user</data>\n')
            f.write('    </node>\n')

        edge_count = 0
        for movie_id, user_id in final_edges:
            f.write(f'    <edge source="{html.escape(movie_id)}" target="{html.escape(user_id)}"/>\n')
            edge_count += 1

        f.write('  </graph>\n')
        f.write('</graphml>\n')

    return len(selected_movies), len(selected_users), edge_count


def main() -> None:
    parser = argparse.ArgumentParser(description='Downsample a Netflix GraphML for browser use.')
    parser.add_argument('--input',      default='netflix-filtered.graphml', metavar='PATH')
    parser.add_argument('--output',     default='netflix-small.graphml',    metavar='PATH')
    parser.add_argument('--max-movies', type=int, default=1000,             metavar='N')
    parser.add_argument('--max-users',  type=int, default=5000,             metavar='N')
    parser.add_argument('--max-edges',  type=int, default=40000,            metavar='N')
    args = parser.parse_args()

    movie_meta, movie_degrees = pass1_movie_degrees(args.input)

    print(f"\nSelecting top {args.max_movies:,} movies by degree ...")
    selected_movies = set(
        mid for mid, _ in sorted(movie_degrees.items(), key=lambda x: -x[1])[:args.max_movies]
    )
    min_deg = min(movie_degrees[m] for m in selected_movies)
    max_deg = max(movie_degrees[m] for m in selected_movies)
    print(f"  Degree range of selected movies: {min_deg:,} – {max_deg:,}")

    n_movies, n_users, n_edges = pass2_collect_and_write(
        args.input, selected_movies, movie_meta, args.max_users, args.max_edges, args.output,
    )

    print(f"\nDone. Written to {args.output}")
    print(f"  Movies : {n_movies:>6,}")
    print(f"  Users  : {n_users:>6,}")
    print(f"  Edges  : {n_edges:>6,}")


if __name__ == '__main__':
    main()
