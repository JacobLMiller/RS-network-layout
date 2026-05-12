"""
enrich_netflix_with_tmdb.py
----------------------------
Enriches the Netflix Prize movie_titles.csv with metadata from the TMDB API.

Netflix Prize movie_titles.csv format (no header):
    movie_id,year,title
    e.g.  1,2003,Dinosaur Planet

TMDB API (free): https://www.themoviedb.org/settings/api
    - Rate limit: ~40 requests / 10 seconds (the script respects this).

Usage:
    pip install requests pandas tqdm
    python enrich_netflix_with_tmdb.py \
        --input  movie_titles.csv \
        --output netflix_enriched.csv \
        --api_key YOUR_TMDB_API_KEY

The script saves a checkpoint after every batch so it can be safely
interrupted and resumed without re-fetching already-processed movies.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# TMDB helpers
# ---------------------------------------------------------------------------

TMDB_BASE = "https://api.themoviedb.org/3"
REQUESTS_PER_SECOND = 4          # conservative — TMDB allows ~40 / 10 s
CHECKPOINT_EVERY    = 500         # rows between checkpoint saves
MAX_RETRIES         = 3


def tmdb_get(endpoint: str, api_key: str, params: dict | None = None) -> dict | None:
    """GET a TMDB endpoint, retrying on transient errors."""
    url = f"{TMDB_BASE}{endpoint}"
    base_params = {"api_key": api_key}
    if params:
        base_params.update(params)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=base_params, timeout=10)
            if resp.status_code == 429:          # rate-limited — back off
                retry_after = int(resp.headers.get("Retry-After", 5))
                time.sleep(retry_after)
                continue
            if resp.status_code == 200:
                return resp.json()
            # Non-retryable HTTP errors (404, 401 …) — just return None
            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def search_movie(title: str, year: str | None, api_key: str) -> dict | None:
    """
    Search TMDB for a movie by title (+ optional year).
    Returns the best-matching result dict or None.
    """
    params: dict = {"query": title, "include_adult": "false"}
    if year and str(year).isdigit():
        params["year"] = year

    data = tmdb_get("/search/movie", api_key, params)
    if not data or not data.get("results"):
        # Retry without year constraint in case the year is slightly off
        if year:
            params.pop("year", None)
            data = tmdb_get("/search/movie", api_key, params)

    if not data or not data.get("results"):
        return None

    return data["results"][0]          # TMDB returns results by relevance


def fetch_details(tmdb_id: int, api_key: str) -> dict:
    """
    Fetch full movie details + credits in one call using append_to_response.
    Returns a dict with overview, genres, cast, crew, runtime, etc.
    """
    data = tmdb_get(
        f"/movie/{tmdb_id}",
        api_key,
        {"append_to_response": "credits,keywords"},
    )
    if not data:
        return {}

    # Flatten genres
    genres = ", ".join(g["name"] for g in data.get("genres", []))

    # Top-5 cast members
    cast = data.get("credits", {}).get("cast", [])
    top_cast = ", ".join(c["name"] for c in cast[:5])

    # Director(s)
    crew = data.get("credits", {}).get("crew", [])
    directors = ", ".join(
        c["name"] for c in crew if c.get("job") == "Director"
    )

    # Keywords
    keywords = ", ".join(
        k["name"] for k in data.get("keywords", {}).get("keywords", [])[:10]
    )

    return {
        "tmdb_id":           tmdb_id,
        "tmdb_title":        data.get("title"),
        "original_language": data.get("original_language"),
        "overview":          data.get("overview"),
        "tagline":           data.get("tagline"),
        "genres":            genres,
        "cast":              top_cast,
        "director":          directors,
        "keywords":          keywords,
        "runtime_min":       data.get("runtime"),
        "tmdb_vote_average": data.get("vote_average"),
        "tmdb_vote_count":   data.get("vote_count"),
        "popularity":        data.get("popularity"),
        "tmdb_release_date": data.get("release_date"),
        "poster_path": (
            f"https://image.tmdb.org/t/p/w342{data['poster_path']}"
            if data.get("poster_path") else None
        ),
    }


# ---------------------------------------------------------------------------
# Main enrichment logic
# ---------------------------------------------------------------------------

EMPTY_RECORD = {
    "tmdb_id": None, "tmdb_title": None, "original_language": None,
    "overview": None, "tagline": None, "genres": None, "cast": None,
    "director": None, "keywords": None, "runtime_min": None,
    "tmdb_vote_average": None, "tmdb_vote_count": None,
    "popularity": None, "tmdb_release_date": None, "poster_path": None,
}


def load_netflix_titles(path: str) -> pd.DataFrame:
    """Parse movie_titles.csv (no header, comma-separated, title may contain commas)."""
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            # Format: id,year,title  — year field is always 4 digits or NULL
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            movie_id, year, title = parts
            rows.append({
                "movie_id": int(movie_id.strip()),
                "year":     year.strip() if year.strip() != "NULL" else None,
                "title":    title.strip(),
            })
    return pd.DataFrame(rows)


def enrich(input_path: str, output_path: str, api_key: str) -> None:
    checkpoint_path = output_path + ".checkpoint.json"

    # Load Netflix titles
    print(f"Loading {input_path} …")
    df = load_netflix_titles(input_path)
    print(f"  {len(df):,} movies found.")

    # Load checkpoint if it exists (so we can resume)
    enriched: dict[int, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            enriched = {int(k): v for k, v in json.load(f).items()}
        print(f"  Resuming from checkpoint — {len(enriched):,} already fetched.")

    delay = 1.0 / REQUESTS_PER_SECOND

    todo = df[~df["movie_id"].isin(enriched.keys())]
    print(f"  {len(todo):,} movies left to fetch.\n")

    for i, row in enumerate(tqdm(todo.itertuples(), total=len(todo), unit="movie")):
        movie_id = row.movie_id
        title    = row.title
        year     = row.year

        # 1. Search TMDB
        result = search_movie(title, year, api_key)
        time.sleep(delay)

        if result is None:
            enriched[movie_id] = EMPTY_RECORD.copy()
        else:
            tmdb_id = result["id"]
            # 2. Fetch detailed info
            details = fetch_details(tmdb_id, api_key)
            time.sleep(delay)
            enriched[movie_id] = details if details else EMPTY_RECORD.copy()

        # Checkpoint periodically
        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(enriched, f)
            tqdm.write(f"  ✓ Checkpoint saved ({len(enriched):,} records).")

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(enriched, f)

    # Merge back with original dataframe
    enriched_df = pd.DataFrame.from_dict(enriched, orient="index")
    enriched_df.index.name = "movie_id"
    enriched_df = enriched_df.reset_index()
    enriched_df["movie_id"] = enriched_df["movie_id"].astype(int)

    merged = df.merge(enriched_df, on="movie_id", how="left")

    # Save final output
    merged.to_csv(output_path, index=False)
    print(f"\n✅ Done! Enriched data saved to: {output_path}")

    # Print a quick summary
    matched = merged["tmdb_id"].notna().sum()
    print(f"   {matched:,} / {len(merged):,} movies matched on TMDB "
          f"({matched / len(merged) * 100:.1f}%)")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("   Checkpoint file removed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enrich Netflix Prize movie_titles.csv with TMDB metadata."
    )
    p.add_argument(
        "--input",  default="movie_titles.csv",
        help="Path to the Netflix Prize movie_titles.csv (default: movie_titles.csv)"
    )
    p.add_argument(
        "--output", default="netflix_enriched.csv",
        help="Path for the output CSV (default: netflix_enriched.csv)"
    )
    p.add_argument(
        "--api_key", required=True,
        help="Your TMDB API key (get one free at https://www.themoviedb.org/settings/api)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(
        input_path  = args.input,
        output_path = args.output,
        api_key     = args.api_key,
    )