from embed.common_dist import UnionGraph
from embed.two_layer_naive import box_procrustes_layout
from embed.two_layer_fructermann import two_layer_layout
from embed._utils import read_json_reprsentation
from embed.poster_method import Poster

from tqdm import tqdm

import networkx as nx


if __name__ == "__main__":
    import json
    import os 

    for graphname in tqdm(os.listdir("json_data")):
        graphname = graphname.replace(".json", "")
        if not os.path.isdir(f"outputs/{graphname}"): os.mkdir(f"outputs/{graphname}")

        with open(f"json_data/{graphname}.json", 'r') as fdata:
            data = json.load(fdata)

        G = read_json_reprsentation(data,graphname)
        print(graphname)

        attrs = {
            "paper_type": "rich",
            "author_type": "sparse",
            "embed_papers_by": "data",
            "paper_embed_data_loc": "data"
        }

        pos_layouts = {
            'Semantic' : lambda G: Poster(G, **attrs).layout(),
            'Union_FR' : lambda G : UnionGraph(G, **attrs).fruchtermann(),
            'Union_UMAP' : lambda G : UnionGraph(G, **attrs).dim_reduction('UMAP'),
            'Independent_R' : lambda G : box_procrustes_layout(G, transform_author=True, **attrs),
            'Independent_S' : lambda G : box_procrustes_layout(G, transform_author=False, **attrs),
            'Naive_FR': nx.spring_layout
        }

        for name, algo in pos_layouts.items():
            pos = algo(G)
            for v in G.nodes():
                print(G.nodes[v], pos[v])

                G.nodes[v]['x'] = float(pos[v][0])
                G.nodes[v]['y'] = float(pos[v][1])
            
            
            jsdata = nx.json_graph.node_link_data(G,edges='links')
            
            with open(f"outputs/{graphname}/{name}" + ".json", 'w') as fdata:
                json.dump(jsdata, fdata,indent=4)

    