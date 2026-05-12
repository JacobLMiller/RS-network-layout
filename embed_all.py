from embed.common_dist import UnionGraph
from embed.independent import box_procrustes_layout
from embed._utils import read_json_reprsentation
from embed.semantic_layout import Semantic
from embed.preprocess import preprocess

from tqdm import tqdm

import networkx as nx


if __name__ == "__main__":
    import json
    import os

    for graphname in tqdm(os.listdir("json_data")):
        graphname = graphname.replace(".json", "")
        os.makedirs(f"outputs/{graphname}", exist_ok=True)

        with open(f"json_data/{graphname}.json", 'r') as fdata:
            data = json.load(fdata)

        G = read_json_reprsentation(data, graphname)
        print(graphname)

        preprocess(G, rich_type='rich', data_key='data', data_format='text', vector_key='vector')

        semantic_attrs = {
            "rich_type": "rich",
            "sparse_type": "sparse",
            "vector_key": "vector",
        }

        legacy_attrs = {
            "rich_type": "rich",
            "sparse_type": "sparse",
            "embed_rich_by": "data",
            "rich_embed_data_loc": "data",
        }

        pos_layouts = {
            'Semantic':      lambda G: Semantic(G, **semantic_attrs).layout(),
            'Union_FR':      lambda G: UnionGraph(G, **legacy_attrs).fruchtermann(),
            'Union_UMAP':    lambda G: UnionGraph(G, **legacy_attrs).dim_reduction('UMAP'),
            'Independent_R': lambda G: box_procrustes_layout(G, transform_author=True, **legacy_attrs),
            'Independent_S': lambda G: box_procrustes_layout(G, transform_author=False, **legacy_attrs),
            'Naive_FR':      nx.spring_layout,
        }

        for name, algo in pos_layouts.items():
            pos = algo(G)
            for v in G.nodes():
                G.nodes[v]['x'] = float(pos[v][0])
                G.nodes[v]['y'] = float(pos[v][1])

            jsdata = nx.json_graph.node_link_data(G, edges='links')

            with open(f"outputs/{graphname}/{name}.json", 'w') as fdata:
                json.dump(jsdata, fdata, indent=4)
