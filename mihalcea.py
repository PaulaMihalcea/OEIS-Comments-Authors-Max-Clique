import argparse
import itertools as its
import json
import networkx as nx
import os
import random
import re
import sys
import tqdm


def load_json(file_path, print_content=False):
    """
    Load a JSON file as a dictionary and optionally print its content.

    Parameters
    ----------
    file_path : string
        Absolute or relative path of the JSON file to be loaded.

    print_content : boolean, (optional, default: False)
        Specifies whether the file's content should be printed (True) or not (False).

    Examples
    --------
    Load and print file "my_json_file.json" in the "path/to/" directory:

        file = load_json('path/to/my_json_file.json', True)
    """

    try:
        file = open(file_path, 'r')
    except OSError:
        print('Could not open file: {}, exiting program.'.format(file_path.split('/')[-1]))
        sys.exit()
    with file:
        raw_data = json.load(file)
        if print_content:
            print('File ' + file_path.split('/')[-1] + ' contents:')
            print()
            print(json.dumps(raw_data, indent=True))
            print()
            print('The \'json\' Python module returns a dictionary, which can be confirmed by invoking the \'type\' function on the loaded data: ' + str(type(raw_data)) + '.')
            print('This dictionary\'s keys are: ' + str(raw_data.keys()).replace('dict_keys([', '').replace('])', '') + '.')
        return raw_data


def parse_authors_from_comments(raw_data):
    """
    Parse (almost) all unique author names from the comments of an OEIS JSON sequence file.

    Uses regular expressions.

    Parameters
    ----------
    raw_data : dict
       OEIS sequence file content loaded as a Python dictionary.

    Examples
    --------
    Parse authors from dict variable named "A000001":

    authors = parse_authors_from_comments(A000001)
    """

    # Regex pattern
    common_pattern = r'[A-Z](?!=[A-Z])[^0-9+\(\)\[\]\{\}\\\/_:;""]{2,}?'
    pattern_list = [('(?<=_)', '(?=_)'), ('(?<=\[)', '(?=\])'), ('(?<=- )', '(?= \(|, )'), ('(?<=\()', '(?=,)')]
    pattern = re.compile('|'.join([start + common_pattern + end for start, end in pattern_list]))

    # Comment parsing
    comment_list = raw_data.get('results')[0].get('comment')
    if comment_list:
        authors = set()
        for comment in comment_list:
            authors.update([n for names in re.findall(pattern, comment) for n in names.split(', ')])
        return authors
    return


def build_graph_from_directory(dir_path, save=False, filename='comments_authors_graph'):
    """
    Build a NetworkX graph with data from OEIS sequences.

    Each node represents an authors of each comment in every sequence;

    Nodes represent all unique authors that can be found in each comment of every sequence.
    Edges link two authors who have commented the same sequence.

    Parameters
    ----------
    dir_path : string
        Absolute or relative path of the directory containing the OEIS JSON sequence files to be loaded.

    save : boolean, (optional, default: False)
        Specifies whether the graph should be saved to disk as JSON (True) or not (False).

    filename : string, (optional, default: 'comments_authors_graph')
        If the graph should be saved to disk, specifies the file name (JSON extension excluded).

    Examples
    --------
    Build a graph g from the sequence files in the "data/sequences" directory and save it to disk as "my_graph.json":

        g = build_graph_from_directory('data/sequences', save=True, 'my_graph.json')
    """

    # Get file list
    if dir_path[-1] != '/':
        dir_path += '/'
    file_list = [json_file for json_file in os.listdir(dir_path) if json_file.endswith('.json')]

    # Prepare variables
    g = nx.Graph()
    progress_bar = tqdm.tqdm(total=len(file_list))

    # Parse all JSON files
    for f in file_list:
        progress_bar.set_description('Parsing file {}'.format(f))
        file_path = dir_path + f
        raw_data = load_json(file_path)

        authors = parse_authors_from_comments(raw_data)
        if authors:
            # g.add_nodes_from(authors)
            g.add_edges_from(list(its.combinations(authors, 2)))
        progress_bar.update(1)

    # Save graph
    if save:
        with open(dir_path.split('/')[0] + '/' + filename + '.json', 'w') as out_file:
            json.dump(nx.readwrite.json_graph.node_link_data(g), out_file)

    return g


def load_json_graph(file_path):
    """
    Load a NetworkX graph from a JSON file.

    Parameters
    ----------
    file_path : string
        Absolute or relative path of the JSON file containing hte graph to be loaded.

    Examples
    --------
    Load graph into variable g from a JSON file named "comments_authors_graph.json":

        g = load_json_graph('data/comments_authors_graph.json')
    """
    with open(file_path) as file:
        return nx.readwrite.json_graph.node_link_graph(json.load(file))


def find_a_maximal_clique(g, tomita=True, print_result=True):
    if not isinstance(g, nx.classes.graph.Graph):
        print('The provided graph is not a valid NetworkX undirected graph.')
        return

    def bron_kerbosch(r, p, x):
        if not p and not x:
            if len(r) > 2:
                return r
        else:
            for v in {*p}:
                return bron_kerbosch(r | {v}, p & {*g.neighbors(v)}, x & {*g.neighbors(v)})

    def bron_kerbosch_tomita_pivoting(r, p, x):
        if not p and not x:
            if len(r) > 2:
                return r
        else:
            u = max({(v, len({n for n in g.neighbors(v) if n in p})) for v in p | x}, key=lambda v: v[1])[0]
            for v in p - {*g.neighbors(u)}:
                return bron_kerbosch_tomita_pivoting(r | {v}, p & {*g.neighbors(v)}, x & {*g.neighbors(v)})

    if g.nodes:
        # Initialization
        r = {*()}
        p = {*g.nodes}
        x = {*()}

        # Bron-Kerbosch algorithm
        if tomita:
            clique = bron_kerbosch_tomita_pivoting(r, p, x)
        else:
            clique = bron_kerbosch(r, p, x)

        # Printing
        if print_result:
            print(clique)

        return clique
    else:
        print('The provided graph is empty.')
        return


def find_all_maximal_cliques(g, tomita=True, print_result=False):
    if not isinstance(g, nx.classes.graph.Graph):
        print('The provided graph is not a valid NetworkX undirected graph.')
        return

    def bron_kerbosch(r, p, x):
        if not p and not x:
            if len(r) > 2:
                yield r
        else:
            for v in {*p}:
                yield from bron_kerbosch(r | {v}, p & {*g.neighbors(v)}, x & {*g.neighbors(v)})
                p = p - {v}
                x.add(v)

    def bron_kerbosch_tomita_pivoting(r, p, x):
        if not p and not x:
            if len(r) > 2:
                yield r
        else:
            u = max({(v, len({n for n in g.neighbors(v) if n in p})) for v in p | x}, key=lambda v: v[1])[0]
            for v in p - {*g.neighbors(u)}:
                yield from bron_kerbosch_tomita_pivoting(r | {v}, p & {*g.neighbors(v)}, x & {*g.neighbors(v)})
                p = p - {v}
                x.add(v)

    if g.nodes:
        # Initialization
        r = {*()}  # TODO https://stackoverflow.com/questions/6130374/empty-set-literal is faster
        p = {*g.nodes}
        x = {*()}

        # Bron-Kerbosch algorithm
        if tomita:
            cliques = bron_kerbosch_tomita_pivoting(r, p, x)
        else:
            cliques = bron_kerbosch(r, p, x)

        # Printing
        if print_result:
            print(*cliques, sep='\n')

        return list(cliques)

    else:
        print('The provided graph is empty.')
        return


def sample_random_subgraph(g, n):  # TODO
    return g.subgraph(random.sample(g.nodes, n))


######################################################################


def main(args):
    """
    Main function of the OEIS Comments Authors Max Clique project.

    Analyze a sample OEIS sequence file, then build a graph of all authors from the comments of all sequences in order to examine it with a series of graph algorithms.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments, usable as args.argument_name.
    """

    # Load sample sequence file
    print('\n' + 'Printing sample OEIS JSON file...')

    file = load_json('data/sequences/A000001.json', print_content=True)

    # Print 'results' section of the sample sequence
    print('\n' + 'Printing sample file "results" section...' + '\n')

    results = file.get('results')
    if results:
        print(json.dumps(results[0], indent=True))
    else:
        print('No "results" section found.')

    # Print 'comment' subsection of the sample sequence
    print('\n' + 'Printing sample file "comment" subsection...' + '\n')

    comment_list = results[0].get('comment')
    if comment_list:
        print(json.dumps(comment_list, indent=True))
    else:
        print('No "comments" subsection found.' + '\n')

    # Graph creation (either from raw data or existing JSON graph)
    if args.build_graph == 'True':  # Build graph and save to file
        print('\n' + 'Building graph g, where:')
        print('- nodes represent all unique authors that can be found in each comment of every sequence;')
        print('- edges link two authors who have commented the same sequence...')

        g = build_graph_from_directory('data/sequences', save=True)
    else:  # Load graph from disk
        print('\n' + 'Loading graph g from "data/comments_authors_graph.json", where:')
        print('- nodes represent all unique authors that can be found in each comment of every sequence;')
        print('- edges link two authors who have commented the same sequence.')

        g = load_json_graph('data/comments_authors_graph.json')

    print('\n' + 'Graph g has {} nodes and {} edges.'.format(len(g.nodes), len(g.edges)))



    # Find and print a maximal clique
    find_a_maximal_clique(g, print_result=True)

    # Find and print all maximal cliques in a random subgraph of 100 nodes
    subgraph = sample_random_subgraph(g, 100)
    find_all_maximal_cliques(subgraph, print_result=True)

    # TODO Timing
    '''
    def mio_no_pivot():
        return find_all_maximal_cliques(big_graph, False)

    def mio_si_pivot():
        return find_all_maximal_cliques(big_graph)

    def suo():
        return nx.find_cliques(big_graph)


    number=1000000
    print(timeit.timeit(mio_no_pivot, number=number))
    print(timeit.timeit(mio_si_pivot, number=number))
    print(timeit.timeit(suo, number=number))
    '''


######################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script for the OEIS Comments Authors Max Clique project.')
    parser.add_argument('-bg', '--build_graph', choices=('True', 'False'), default='True', help='Build graph from raw data, otherwise load it from the "data/comments_authors_graph.json" file.')

    args = parser.parse_args()

    main(args)
