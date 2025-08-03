def save_movie_plot(raw_data_dir="raw_data/netflix", write_file_name="movie_info.json"):
    """
    Saves movie data to a JSON file in the specified directory.

    The function writes a dictionary called `movie_info` to a JSON file.
    Each entry in `movie_info` is keyed by `movie_id` (as found in the raw files)
    and contains the following structure:
    
        {
            'year': <str>,                # Release year of the movie
            'titles': <list of str>,      # List of movie titles (usually one, but may include alternate titles)
            'data': <str>                 # Plot outline or summary of the movie
        }

    Parameters:
        raw_data_dir (str): Path to the directory where the output JSON file will be saved.
        write_file_name (str): Name of the JSON file to write the data to.

    Returns:
        None
    """
    import os
    import csv
    import json
    from imdb import Cinemagoer
    from tqdm import tqdm

    movie_titles_file = "movie_titles.csv"

    movie_info = dict()

    with open(os.path.join(raw_data_dir, movie_titles_file), encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        for row in reader:
            movie_id, year, = row[:2]
            titles = row[2:]
            # titles = [title for t in titles for title in t.split('/')]
            title = ", ".join(titles) # Because maybe commas were part of the movie - but if they are actually two different titles, hopefully still won't be a problem? ¯\_(ツ)_/¯
            movie_info[movie_id] = {"year": year, "title": [title]}

    ia = Cinemagoer()

    with tqdm(movie_info) as pbar:
        for movie_id in pbar:
            titles = movie_info[movie_id]['title']
            pbar.set_postfix(id=movie_id, titles=titles)

            if not titles:
                print(f"No titles found for movie_id: {movie_id}")
                continue


            for title in titles:
                try:
                    search_results = ia.search_movie(title)
                    if search_results:
                        # Get the first result (most likely match)
                        movie = search_results[0]
                        ia.update(movie)

                        plot = movie.get('plot')
                        if plot:
                            # Plot is usually a list, take the first one
                            description = plot[0] if isinstance(plot, list) else str(plot)
                            # Clean up the plot (remove "::Author Name" suffix if present)
                            if '::' in description:
                                description = description.split('::')[0].strip()
                        else:
                            description = movie.get('plot outline', '') # Empty if no plot outline
                        # if not description: continue
  
                        movie_info[movie_id]['data'] = description
                        print(movie_id, description[:100] + "...")

                        break

                except Exception as e:
                    print(f"Error processing {title}: {str(e)}")

    with open(os.path.join(raw_data_dir, write_file_name), 'w', encoding="utf-8") as f:
        json.dump(movie_info, f, indent=4, ensure_ascii=False)

    return movie_info

def save_netflix_as_json(movie_info, raw_data_dir, write_dir, min_sparse_degree=300):
    import json
    import os
    from collections import defaultdict

    combined_files = [
        "combined_data_1.txt",
        "combined_data_2.txt",
        "combined_data_3.txt",
        "combined_data_4.txt",
    ]

    for i, filename in enumerate(combined_files):
        movie_data = {}
        movies_wo_info = []
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
                            "title": movie_info[current_movie_id]["title"],
                            "data": movie_info[current_movie_id]["data"],
                            "s_ids": []
                        }
                    else:
                        # If movie_id not in movie_titles.csv just skip this node
                        movies_wo_info.append(key)
                        movie_data[key] = {
                            "year": None,
                            "title": None,
                            "data": None,
                            "s_ids": []
                        }
                elif current_movie_id:
                    person_id = line.split(",")[0]
                    movie_data[f"r_{current_movie_id}"]["s_ids"].append("s_" + person_id)

        # Remove movies without data
        for movie_id in movies_wo_info:
            movie_data.pop(movie_id)


        # Remove s_ids that have degree lower than min_sparse_degree
        # And remove movies that get a 0 degree as a result of this
        s_id_degree = defaultdict(int)
        for data in movie_data.values():
            for s_id in data['s_ids']:
                s_id_degree[s_id] += 1

        free_node_movies = list()
        for movie_id, data in movie_data.items():
            data['s_ids'] = [s_id for s_id in data['s_ids'] if s_id_degree[s_id] > min_sparse_degree]
            if not data['s_ids']:
                free_node_movies.append(movie_id)
        for movie_id in free_node_movies:
            movie_data.pop(movie_id)


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
        with open(os.path.join(write_dir, file_name), 'w', encoding='utf-8') as json_file:
            json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
        print(f"Dictionary saved to {file_name}") 


def convert_netflix(target_dir="json_data"):
    raw_data_dir = "raw_data/netflix"
    movie_info_file = "movie_info.json"
    movie_info = save_movie_plot(raw_data_dir, movie_info_file)

    save_netflix_as_json(movie_info, raw_data_dir, target_dir)

if __name__ == "__main__":
    import os
    import json

    raw_data_dir = "raw_data/netflix"
    movie_info_file = "movie_info.json"

    with open(os.path.join(raw_data_dir, movie_info_file), 'r') as f:
        movie_info = json.load(f)

    movie_info = {id: info for id, info in movie_info.items() if (('data' in info) and (info['data'] not in [None, ""]))}

    target_dir = "json_data"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    save_netflix_as_json(movie_info, raw_data_dir, target_dir)