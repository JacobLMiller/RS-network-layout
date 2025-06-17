import networkx as nx 
import json




def _convert_vispub(G):

    Authors = set()
    Papers  = list()

    for v in G.nodes():
        if "Paper" in v: 
            nodeobj = G.nodes[v]
            paperobj = {
                'title': nodeobj['Title'],
                'data': nodeobj['Abstract'],
                'url': nodeobj['Link'],
                's_names': nodeobj['AuthorNames-Deduped'].split(";")
            }
            Authors = Authors.union(set(nodeobj['AuthorNames-Deduped'].split(";")))
            Papers.append(paperobj)


    Authors = [(author, i) for i,author in enumerate(Authors)]

    authorlookup = dict(Authors)

    for i,v in enumerate(Papers):
        v['id'] = f'r_{i}'
        v['s_ids'] = [f"s_{authorlookup[a]}" for a in v['s_names']]
        del v['s_names']


    Authors = [{'id': f"s_{i}", 'name': name} for name, i in Authors]



    graphobj = {
        "rich": Papers, 
        "sparse": Authors
    }

    return graphobj

def convert_vispub():
    import os 
    for gname in os.listdir("raw_data/vispub"):
        if "graphml" not in gname: continue

        G = nx.read_graphml(f"raw_data/vispub/{gname}")

        graphobj = _convert_vispub(G)

        with open(f"json_data/{gname.replace('.graphml', '')}.json", 'w') as fdata:
            json.dump(graphobj, fdata, indent=4)

def convert_dagstuhl(start=1990, end=2030,outname="dagstuhl"):
    G = nx.read_graphml("raw_data/dagstuhl_filtered_1CC.graphml")

    Seminars = nx.subgraph(G,[v for v in G.nodes() if G.nodes[v]['type'] == "event"])

    in_range = set()
    for v in Seminars.nodes():
        if "seminar_start" in Seminars.nodes[v]:
            year = int(Seminars.nodes[v]['seminar_start'].split("-")[0])
            if start <= year <= end: 
                in_range.add(v)

    Seminars = nx.subgraph(Seminars,in_range)

    Rich = list()
    Sparse = set() 

    for v in Seminars.nodes():
        participants = set()
        for p in G[v]:
            participants.add(p)
        Sparse = Sparse.union(participants)

        nodeobj = Seminars.nodes[v]
        if "seminar_summary" in nodeobj:
            semobj = {
                "title": nodeobj['seminar_name'],
                "data": nodeobj['seminar_summary'],
                "keywords": nodeobj['seminar_keywords'] if "seminar_keywords" in nodeobj else "",
                "seminar_number": nodeobj['seminar_number'],
                "s_names": list(participants)
            }
            Rich.append(semobj)

    Sparse = [(part, i) for i,part in enumerate(Sparse)]

    authorlookup = dict(Sparse)

    for i,v in enumerate(Rich):
        v['id'] = f'r_{i}'
        v['s_ids'] = [f"s_{authorlookup[a]}" for a in v['s_names']]
        del v['s_names']

    Sparse = [{'id': f"s_{i}", 'name': name} for name, i in Sparse]

    graphobj = {
        "rich": Rich, 
        "sparse": Sparse
    }


    with open(f"json_data/{outname}.json", 'w') as fdata:
        json.dump(graphobj, fdata, indent=4)



def convert_vast():

    G = nx.read_graphml("raw_data/MC3_processed.graphml")

    Companies = nx.subgraph(G,{v for v in G.nodes() if 'type' in G.nodes[v] and G.nodes[v]['type'] in ["Company"]})
    # Persons   = nx.subgraph(G,{v for v in G.nodes() if v not in Companies})
    
    Rich = list() 
    Sparse = set()
    for v in Companies.nodes():
        if not "product_services" in G.nodes[v]: continue
        if G.nodes[v]["product_services"].lower() == "unknown": continue

        contacts = {u for u in G[v]}
        if len(contacts) < 3: continue

        Sparse = Sparse.union(contacts)

        nodeobj = G.nodes[v]
        Rich.append({
            "title": str(v),
            "data": nodeobj["product_services"], 
            "revenue": nodeobj["revenue_omu"],
            "country": nodeobj["country"],
            "s_names": list(contacts)
        })

    Sparse = [(part, i) for i,part in enumerate(Sparse)]

    authorlookup = dict(Sparse)

    for i,v in enumerate(Rich):
        v['id'] = f'r_{i}'
        v['s_ids'] = [f"s_{authorlookup[a]}" for a in v['s_names']]
        del v['s_names']

    Sparse = [{'id': f"s_{i}", 'name': name} for name, i in Sparse]

    graphobj = {
        "rich": Rich, 
        "sparse": Sparse
    }

    with open("json_data/MC3_VAST2023.json", 'w') as fdata:
        json.dump(graphobj,fdata,indent=4)        
    

if __name__ == "__main__":
    import os 
    if not os.path.isdir("json_data"):
        os.mkdir("json_data")

    convert_dagstuhl()
    convert_dagstuhl(1990,2015,"dagstuhl-before2015")
    convert_dagstuhl(2016,2030,"dagstuhl-after2015")

    convert_vispub()
    convert_vast()