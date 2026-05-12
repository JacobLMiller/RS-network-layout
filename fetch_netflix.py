"""
Build a bipartite Netflix graph (movies × users) from the TMDB-enriched CSV and
the raw rating zip files, then reduce to the (p, q)-core.

Memory-efficient streaming design — edges are never held in memory:
  Pre-filter  raw rating counts (any user)  →  fast candidate set reduction
  Core        alternate between movie-degree and user-degree passes until stable:
                (A) drop movies with fewer than p valid-user ratings
                (B) drop users who rated fewer than q valid movies
  Write       stream-write GraphML without building a graph in memory

  p = --min-movie-ratings (default 0, which enforces ≥1 to remove orphans)
  q = --min-user-degree   (default 1)

Output is a GraphML file matching the structure of dagstuhl-filtered.graphml:
movie nodes carry type='movie' plus metadata; user nodes carry type='user'.

Usage
-----
    python fetch_netflix.py --top-genres 8 --min-movie-ratings 500 --min-user-degree 5
    python fetch_netflix.py --top-genres 8 --min-movie-ratings 500 --min-user-degree 5 \\
        --output netflix.graphml
"""
import argparse
import csv
import html
import os
import zipfile
from collections import Counter, defaultdict

import numpy as np

# Largest user ID in the Netflix Prize dataset (user IDs are 1-indexed integers).
MAX_USER_ID = 2_649_429

REQUIRED_ATTRS = ['overview', 'genres', 'poster_path']
RATING_FILES = [
    'combined_data_1.txt.zip',
    'combined_data_2.txt.zip',
    'combined_data_3.txt.zip',
    'combined_data_4.txt.zip',
]
MOVIE_ATTRS = [
    'title', 'year', 'data', 'genres', 'poster_path',
    'tmdb_id', 'cast', 'director', 'keywords',
    'tmdb_vote_average', 'tmdb_vote_count',
]


# ---------------------------------------------------------------------------
# Movie loading
# ---------------------------------------------------------------------------

def _top_genres(movies: dict[str, dict], n: int) -> set[str]:
    counter: Counter = Counter()
    for meta in movies.values():
        for genre in meta['genres'].split(','):
            genre = genre.strip()
            if genre:
                counter[genre] += 1
    top = {genre for genre, _ in counter.most_common(n)}
    print(f"  Top {n} genres: {', '.join(sorted(top))}")
    return top


def _load_clean_movies(csv_path: str, top_genres: set[str] | None) -> dict[str, dict]:
    movies: dict[str, dict] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if any(not row.get(attr, '').strip() for attr in REQUIRED_ATTRS):
                continue
            if top_genres is not None:
                row_genres = {g.strip() for g in row.get('genres', '').split(',')}
                if not row_genres & top_genres:
                    continue
            movies[row['movie_id']] = {
                'title':             row.get('title', ''),
                'year':              row.get('year', ''),
                'data':              row.get('overview', ''),
                'genres':            row.get('genres', ''),
                'poster_path':       row.get('poster_path', ''),
                'tmdb_id':           row.get('tmdb_id', ''),
                'cast':              row.get('cast', ''),
                'director':          row.get('director', ''),
                'keywords':          row.get('keywords', ''),
                'tmdb_vote_average': row.get('tmdb_vote_average', ''),
                'tmdb_vote_count':   row.get('tmdb_vote_count', ''),
            }
    return movies


# ---------------------------------------------------------------------------
# Streaming edge reader (used by every pass)
# ---------------------------------------------------------------------------

def _stream_edges(raw_dir: str, valid_ids: set[str], label: str = ''):
    for filename in RATING_FILES:
        path = os.path.join(raw_dir, filename)
        print(f"  {label + ' ' if label else ''}reading {filename} ...", flush=True)
        with zipfile.ZipFile(path) as z:
            with z.open(z.namelist()[0]) as f:
                current_movie: str | None = None
                in_valid = False
                for raw_line in f:
                    line = raw_line.decode().strip()
                    if line.endswith(':'):
                        current_movie = line[:-1]
                        in_valid = current_movie in valid_ids
                    elif in_valid:
                        yield current_movie, line.split(',')[0]


# ---------------------------------------------------------------------------
# Passes 1-3: counting only, no edge storage
# ---------------------------------------------------------------------------

def _count_movie_ratings(
    raw_dir: str,
    candidate_ids: set[str],
    valid_user_mask: np.ndarray | None = None,
    label: str = '[pre-filter]',
) -> dict[str, int]:
    """Count ratings per candidate movie, optionally restricted to valid users."""
    counts: dict[str, int] = defaultdict(int)
    n = 0
    for movie_id, user_id in _stream_edges(raw_dir, candidate_ids, label=label):
        if valid_user_mask is None or valid_user_mask[int(user_id)]:
            counts[movie_id] += 1
        n += 1
        if n % 5_000_000 == 0:
            print(f"    {n:,} ratings scanned", flush=True)
    return dict(counts)


def _count_user_degrees(
    raw_dir: str,
    valid_movies: set[str],
    label: str = '[user degrees]',
) -> np.ndarray:
    """Count how many valid movies each user rated.

    Returns an int32 array of length MAX_USER_ID + 1 where index i is the
    degree of user i.  ~10 MB vs ~150 MB for an equivalent dict[str, int].
    """
    counts = np.zeros(MAX_USER_ID + 1, dtype=np.int32)
    for _, user_id in _stream_edges(raw_dir, valid_movies, label=label):
        counts[int(user_id)] += 1
    return counts


def _core_decomposition(
    raw_dir: str,
    candidate_movies: set[str],
    min_movie_degree: int,
    min_user_degree: int,
) -> tuple[set[str], np.ndarray]:
    """Iteratively compute the (min_movie_degree, min_user_degree)-core.

    Each iteration:
      (A) Re-count movie degrees counting only currently valid users;
          drop movies whose degree falls below the threshold.
      (B) Re-count user degrees counting only currently valid movies;
          drop users whose degree falls below the threshold.
    Repeats until neither set changes.

    The movie threshold is treated as 'strictly greater than min_movie_degree'
    (same semantics as --min-movie-ratings), so the default of 0 enforces ≥1
    and automatically removes any movie left with no valid-user edge.
    """
    valid_movies    = candidate_movies
    valid_user_mask = np.ones(MAX_USER_ID + 1, dtype=bool)
    valid_user_mask[0] = False   # user IDs are 1-indexed

    for iteration in range(1, 1000):
        n_movies_before = len(valid_movies)
        n_users_before  = int(valid_user_mask.sum())

        # (A) Movie pass: count only edges to currently valid users
        movie_degrees = _count_movie_ratings(
            raw_dir, valid_movies, valid_user_mask,
            label=f'[core {iteration}a]',
        )
        valid_movies = {mid for mid, c in movie_degrees.items()
                        if c > min_movie_degree}

        # (B) User pass: count only edges to currently valid movies
        user_degree_arr = _count_user_degrees(
            raw_dir, valid_movies,
            label=f'[core {iteration}b]',
        )
        valid_user_mask = user_degree_arr >= min_user_degree
        valid_user_mask[0] = False
        del user_degree_arr

        dropped_movies = n_movies_before - len(valid_movies)
        dropped_users  = n_users_before  - int(valid_user_mask.sum())
        print(
            f"  [core {iteration}] "
            f"−{dropped_movies:,} movies, −{dropped_users:,} users"
            f"  →  {len(valid_movies):,} movies, {int(valid_user_mask.sum()):,} users",
            flush=True,
        )

        if dropped_movies == 0 and dropped_users == 0:
            break

    return valid_movies, valid_user_mask


# ---------------------------------------------------------------------------
# Pass 4: stream-write GraphML
# ---------------------------------------------------------------------------

def _write_graphml(
    path: str,
    movies: dict[str, dict],
    final_movies: set[str],
    valid_user_mask: np.ndarray,
    raw_dir: str,
) -> int:
    """Pass 4 — write nodes then stream edges directly to GraphML without building a graph."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')

        # Key declarations
        f.write('  <key id="type" for="node" attr.name="type" attr.type="string"/>\n')
        for attr in MOVIE_ATTRS:
            f.write(f'  <key id="{attr}" for="node" attr.name="{attr}" attr.type="string"/>\n')

        f.write('  <graph id="G" edgedefault="undirected">\n')

        # Movie nodes
        for movie_id in sorted(final_movies, key=int):
            nid = html.escape(f'r_{movie_id}')
            f.write(f'    <node id="{nid}">\n')
            f.write('      <data key="type">movie</data>\n')
            for attr in MOVIE_ATTRS:
                val = movies[movie_id].get(attr, '')
                if val:
                    f.write(f'      <data key="{attr}">{html.escape(str(val))}</data>\n')
            f.write('    </node>\n')

        # User nodes — np.where returns sorted indices naturally
        for user_id in np.where(valid_user_mask)[0]:
            nid = html.escape(f's_{user_id}')
            f.write(f'    <node id="{nid}">\n')
            f.write('      <data key="type">user</data>\n')
            f.write('    </node>\n')

        # Edges — streamed, never accumulated
        print("  [pass 4] streaming edges ...", flush=True)
        edge_count = 0
        for movie_id, user_id in _stream_edges(raw_dir, final_movies, label='[pass 4]'):
            if valid_user_mask[int(user_id)]:
                src = html.escape(f'r_{movie_id}')
                tgt = html.escape(f's_{user_id}')
                f.write(f'    <edge source="{src}" target="{tgt}"/>\n')
                edge_count += 1

        f.write('  </graph>\n')
        f.write('</graphml>\n')

    return edge_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build Netflix bipartite GraphML from enriched CSV + rating zip files.'
    )
    parser.add_argument('--csv',      default='netflix_enriched.csv', metavar='PATH')
    parser.add_argument('--raw-dir',  default='raw_data/netflix',     metavar='DIR')
    parser.add_argument('--output',   default='netflix.graphml',      metavar='PATH')
    parser.add_argument('--top-genres', type=int, default=0, metavar='N',
                        help='Keep movies with at least one of the N most frequent genres (0=all).')
    parser.add_argument('--min-movie-ratings', type=int, default=0, metavar='K',
                        help='Drop movies with K or fewer ratings.')
    parser.add_argument('--min-user-degree', type=int, default=1, metavar='N',
                        help='Drop users who rated fewer than N remaining movies (default: 1).')
    args = parser.parse_args()

    # --- Load + genre-filter movies ---
    print(f"Loading movies from {args.csv} ...")
    if args.top_genres > 0:
        all_movies = _load_clean_movies(args.csv, top_genres=None)
        genre_set = _top_genres(all_movies, args.top_genres)
        movies = _load_clean_movies(args.csv, top_genres=genre_set)
    else:
        movies = _load_clean_movies(args.csv, top_genres=None)
    print(f"  {len(movies):,} movies after metadata + genre filter")

    # --- Raw pre-filter: drop movies below the threshold on total ratings ---
    # Any movie failing this on raw (unfiltered) counts cannot survive the core,
    # so this pass cheaply shrinks the candidate set before iteration begins.
    print("\nPre-filter: counting raw ratings per movie ...")
    raw_counts = _count_movie_ratings(args.raw_dir, set(movies))
    candidate_movies = {mid for mid, c in raw_counts.items()
                        if c > args.min_movie_ratings}
    del raw_counts
    print(f"  {len(movies) - len(candidate_movies):,} movies dropped by pre-filter")
    print(f"  {len(candidate_movies):,} candidates entering core decomposition")

    # --- (p, q)-core decomposition ---
    print(f"\nComputing ({args.min_movie_ratings}, {args.min_user_degree})-core ...")
    final_movies, valid_user_mask = _core_decomposition(
        args.raw_dir, candidate_movies,
        min_movie_degree=args.min_movie_ratings,
        min_user_degree=args.min_user_degree,
    )
    del candidate_movies

    # --- Pass 4: write GraphML ---
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"\nPass 4: writing {args.output} ...")
    edge_count = _write_graphml(args.output, movies, final_movies, valid_user_mask, args.raw_dir)

    n_users = int(valid_user_mask.sum())
    print(f"\nDone.")
    print(f"  Movies : {len(final_movies):>8,}")
    print(f"  Users  : {n_users:>8,}")
    print(f"  Edges  : {edge_count:>8,}")


if __name__ == '__main__':
    main()
