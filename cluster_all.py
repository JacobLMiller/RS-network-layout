from embed.common_dist import UnionGraph
from embed.two_layer_naive import box_procrustes_layout
from embed.two_layer_fructermann import two_layer_layout
from embed._utils import read_json_reprsentation
from embed.semantic_layout import Semantic
import numpy as np

from tqdm import tqdm

import networkx as nx


from embed.cluster import get_clustering
from collections import defaultdict, Counter

import pylab as plt
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union 
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, Delaunay



def compute_countries(G,X,r = 0.1,cluster="cluster",memberships=None):
    np.random.seed(42)


    from scipy.spatial.distance import cdist

    clusters = dict()
    cid = dict()
    for i,v in enumerate(X):
        if memberships[i] not in clusters:
            clusters[memberships[i]] = [i]
        else:
            clusters[memberships[i]].append(i)
        cid[i] = memberships[i]
    
        
    diam = np.max(np.linalg.norm(X,axis=1))
    max_d = np.max(X)
    min_d = np.min(X)

    print(diam, min_d, max_d)

    if abs(diam - 1) <= 0.5:
        X /= diam

        R = np.random.uniform(-2, 2,(10000,2))
    else:
        R = np.random.uniform(min_d - diam, max_d + diam,(100000,2))

    dist = cdist(R,X,"euclidean") # Pairwise distance between random points and graph vertex positions
    dist = np.min(dist,axis=1)
    R = R[(r <= dist)]            # Throw out all random points with distance to any vertex < r

    vor = Voronoi(np.concatenate((X,R),axis=0)) #Voronoi diagram of all points (the remaining random and vertex positions)
    cluster_points = dict()
    for c in clusters:
        verts = clusters[c]

        #Extract voronoi vertices of bounding region of each graph vertex. 
        #Convert to shapely polygon and take the unary union to get outer boundaries
        pt = [Polygon([(x[0], x[1]) for x in vor.vertices[vor.regions[vor.point_region[v]]]]) for v in verts ]
        pt = shapely.get_coordinates(unary_union(pt))
        
        #TODO: Make sure disjoint regions are properly wound

        #Assume shapely represents union of disjoint triangles ABC and XYZ as the sequence ABCAXYZX
        #Whenever we revisit a point, we have a closed polygon
        visited = set()
        breakpoints = list()
        start = 0
        for i,point in enumerate(pt):
            if tuple(point) in visited:
                breakpoints.append( (start, i+1) )
                start = i+1
            visited.add(tuple(point))
        
        polys = [pt[start:end] for start, end in breakpoints]
        #Flatten array, delimiter of single space for points, /// for disjoint polygons
        cluster_points[c] = "///".join(" ".join( str(num) for num in np.around(poly.reshape(-1), 3) ) for poly in polys )
        # cluster_points[c] = " ".join([str(num) for num in np.around(pt.reshape(-1), 3)])
        
    #Flatten array, delimiter of ";"
    return ";".join([cluster_points[i] for i in range(len(cluster_points.keys()))])



if __name__ == "__main__":
    import json
    import os 

    datasets = ["dagstuhl",] 
                # "dagstuhl-after2015", "dagstuhl-after2015", "InfoVis_graph", 
                # "MC3_VAST2023", "SciVis_graph", "VisPub_2015-19_graph", "VisPub_2020-24_graph", "VisPub_graph"]
    algorithms = ["Semantic", "Independent_RA", "Independent_SA", "Union_FR", "Union_UMAP"]

    for dataset in datasets:
        if not os.path.isdir(f"outputs_new/{dataset}"):
            os.mkdir(f"outputs_new/{dataset}")

        for alg in algorithms:

            with open(f"outputs/{dataset}/{alg}.json", 'r') as fdata:
                data = json.load(fdata)

            print(alg, dataset)


            X = np.array([[v['x'], v['y']] for v in data['nodes']])

            full = data 
            data = full['nodes']

            rich_indices = [i for i, v in enumerate(data) if v['type'] == 'rich']
            if dataset == "dagstuhl": rich_indices = [i for i, v in enumerate(data) if v['type'] == 'rich' and v['keywords']]            
            rich_map = {local: gglobal for local, gglobal in enumerate(rich_indices)}

            Xp = X[list(rich_indices)]
            c1, c2 = get_clustering(Xp)


            c1collect = defaultdict(Counter)
            c2collect = defaultdict(Counter)

            c1map = {i: val for i,val in enumerate(c1)}
            c2map = {i: val for i,val in enumerate(c2)}


            if dataset == "dagstuhl":
                for ind in range(Xp.shape[0]):
                    c1collect[c1[ind]].update(data[ind]['keywords'].split("; "))
                    c2collect[c2[ind]].update(data[ind]['keywords'].split("; "))

                # for collect in c1collect.values(): 
                #     collect.discard("")
                # for collect in c2collect.values(): 
                #     collect.discard("")        

    
                #Ask an LLM for labels?

                full['cluster_labels_1'] = ";".join([c1collect[i].most_common(2)[1][0] if c1collect[i] else None for i in range(len(c1collect))])
                full['cluster_labels_2'] = ";".join([c2collect[i].most_common(2)[1][0] if c2collect[i] else None for i in range(len(c2collect))])
                full["cluster_labels"] = [full['cluster_labels_1'], full['cluster_labels_2']]
                full['alts'] = [[f"{c[0]}_{c[1]}" for c in c2collect[i].most_common(10)] for i in range(len(c2collect))]
            else:
                full["cluster_labels_1"] = ";".join([str(x) for x in np.unique(c1)])
                full["cluster_labels_2"] = ";".join([str(x) for x in np.unique(c2)])
                full["cluster_labels"] = [full['cluster_labels_1'], full['cluster_labels_2']]



            for local, gglobal in rich_map.items():
                data[gglobal]['c1'] = int(c1map[local])
                data[gglobal]['c2'] = int(c2map[local])

            c2 = compute_countries(full,Xp,r=0.6, cluster='c2',memberships=c2map)
            c1 = compute_countries(full,Xp,r=0.6, cluster='c1',memberships=c1map)
            full["cluster_polygons"] = [c1, c2]
            full['divisor'] = float(np.max(np.linalg.norm(Xp,axis=1)))

            with open(f"outputs_new/{dataset}/{alg}.json", 'w') as fdata:
                json.dump(full,fdata,indent=4)