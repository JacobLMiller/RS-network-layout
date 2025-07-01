import os
import csv
import json

raw_data_dir = "raw_data/netflix"
combined_files = [
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
]
movie_titles_file = "movie_titles.csv"

# Step 1: Read movie_titles.csv into a dict
"""
movie_info[movie_id] = (movie_id in the raw files)
{
    'year' : year,
    'titles' : list of titles, (usually only one title, but some movies have a few titles)
    'data' : outline of movie plot
}
"""
movie_info = {}
with open(os.path.join(raw_data_dir, movie_titles_file), encoding="ISO-8859-1") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        movie_id, year, = row[:2]
        titles = row[2:]
        titles = [title for t in titles for title in t.split('/')] # TODO Some titles contain commas (e.g. 20,000 Leagues Under the Sea) so CSV reader splits the title

        movie_info[movie_id] = {"year": year, "titles": list(titles)}

# Step 2: Build the result dictionary
"""
movie_data[movie_id] = (here, movie_id is f"r_{original_movie_id"})
    {
        'year': year,
        'titles' : list of titles,
        's_ids' : list of customer ids
    }
"""

for i, filename in enumerate(combined_files):
    movie_data = {}
    with open(os.path.join(raw_data_dir, filename), "r") as f:
        current_movie_id = None
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                current_movie_id = line[:-1]
                key = f"r_{current_movie_id}"
                if current_movie_id in movie_info:
                    movie_data[key] = {
                        "year": movie_info[current_movie_id]["year"],
                        "titles": movie_info[current_movie_id]["titles"],
                        "s_ids": []
                    }
                else:
                    # If movie_id not in movie_titles.csv, skip or handle as needed
                    movie_data[key] = {
                        "year": None,
                        "titles": None,
                        "s_ids": []
                    }
            elif current_movie_id:
                person_id = line.split(",")[0]
                movie_data[f"r_{current_movie_id}"]["s_ids"].append("s_" + person_id)

    rich_data: list = [{'id': r_id} | r_data for r_id, r_data in movie_data.items()]
    del movie_data

    unique_s_ids = sorted(set(s_id for r_data in rich_data for s_id in r_data['s_ids']),
                        key=lambda x: int(x.split('_')[1]))
    sparse_data: list = [{'id': s_id} for s_id in unique_s_ids]


    data_dict = {
        'rich' : rich_data,
        'sparse' : sparse_data
    }

    file_name = f'netflix_graph_{i + 1}.json'
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
    print(f"Dictionary saved to {file_name}") 
