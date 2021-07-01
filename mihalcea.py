import argparse
import itertools as its
import json
import networkx as nx
import numpy as np
import os
import random
import re
import timeit
import tqdm
import warnings


def load_json(file_path, print_result=False):
    """
    Loads a JSON file as a dictionary and optionally print its content.

    Parameters
    ----------
    file_path : str
        Absolute or relative path of the JSON file to be loaded.

    print_result : boolean, (optional, default: False)
        Specifies whether the file's content should be printed (True) or not (False).

    Returns
    ----------
    dict
        A Python dictionary containing the raw JSON data of the loaded file, or None.

    Examples
    --------
    Load and print file "my_json_file.json" in the "path/to/" directory:
        file = load_json('path/to/my_json_file.json', True)
    """

    try:
        with open(file_path, 'r') as file:
            raw_data = json.load(file)
            if print_result:
                print('File ' + file_path.split('/')[-1] + ' contents:')
                print()
                print(json.dumps(raw_data, indent=True))
                print()
                print(
                    'The \'json\' Python module returns a dictionary, which can be confirmed by invoking the \'type\' function on the loaded data: ' + str(
                        type(raw_data)) + '.')
                print('This dictionary\'s keys are: ' + str(raw_data.keys()).replace('dict_keys([', '').replace('])',
                                                                                                                '') + '.')
            return raw_data
    except OSError:
        print('Could not open file: {}, exiting program.'.format(file_path.split('/')[-1]))


def parse_authors_from_comments(raw_data):
    """
    Parses (almost) all unique author names from the comments of an OEIS JSON sequence file.

    Uses regular expressions.

    Parameters
    ----------
    raw_data : dict
       OEIS sequence file content loaded as a Python dictionary.

    Returns
    ----------
    set
        A set containing (almost) all unique author names from the comments of the given OEIS sequence.

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
    Builds a NetworkX graph with data from OEIS sequences, where:
        - nodes represent all unique authors that can be found in each comment of every sequence.
        - edges link two authors who have commented the same sequence.

    Parameters
    ----------
    dir_path : str
        Absolute or relative path of the directory containing the OEIS JSON sequence files to be loaded.

    save : boolean, (optional, default: False)
        Specifies whether the graph should be saved to disk as JSON (True) or not (False).

    filename : str, (optional, default: 'comments_authors_graph')
        If the graph should be saved to disk, specifies the file name (JSON extension excluded).

    Returns
    ----------
    networkx.classes.graph.Graph
        A NetworkX undirected graph built with data from OEIS sequences, where:
            - nodes represent all unique authors that can be found in each comment of every sequence.
            - edges link two authors who have commented the same sequence.

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
        try:
            with open(dir_path.split('/')[0] + '/' + filename + '.json', 'w') as out_file:
                json.dump(nx.readwrite.json_graph.node_link_data(g), out_file)
        except OSError:
            print('Could not save file: {}, exiting program.'.format(filename + '.json'))

    return g


def load_json_graph(file_path):
    """
    Loads a NetworkX graph from a JSON file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path of the JSON file containing hte graph to be loaded.

    Returns
    ----------
    networkx.classes.graph.Graph class
        A NetworkX undirected graph loaded from the given JSON file.

    Examples
    --------
    Load graph into variable g from a JSON file named "comments_authors_graph.json":
        g = load_json_graph('data/comments_authors_graph.json')
    """
    try:
        with open(file_path, 'r') as file:
            raw_data = json.load(file)
            return nx.readwrite.json_graph.node_link_graph(raw_data)
    except OSError:
        print('Could not open file: {}, exiting program.'.format(file_path.split('/')[-1]))


def get_degeneracy_ordering(graph):
    """
    Computes a degeneracy ordering for the given undirected graph.

    Parameters
    ----------
    graph : networkx.classes.graph.Graph
        A NetworkX undirected graph.

    Returns
    ----------
    list
        A list containing the degeneracy ordering of the given graph.

    Raises
    -------
    networkx.NetworkXError
       If g is not a NetworkX graph.

    networkx.NetworkXPointlessConcept
        If g is an empty graph.

    Examples
    --------
    Load graph into variable g from a JSON file named "comments_authors_graph.json":
        degeneracy_ordering = get_degeneracy_ordering(g)
    """

    # Check that g is a NetworkX graph
    if not isinstance(graph, nx.classes.graph.Graph):
        raise nx.NetworkXError('The provided graph is not a valid NetworkX undirected graph.')

    if graph.nodes:
        g = graph.copy()

        # Create and populate lists of lists
        max_degree = max([d for n, d in g.degree()])
        d = [[] for deg in range(max_degree + 1)]
        for node in g.degree():
            d[node[1]].append(node[0])

        # Degeneracy ordering
        degeneracy_ordering = []
        while d:
            # Get current node u
            u = next(i for i in d if i).pop()
            degeneracy_ordering.append(u)

            # Move neighbors of current node
            for v in {*g.neighbors(u)}:
                v_deg = g.degree(v)
                d[v_deg].remove(v)
                d[v_deg-1].append(v)

            # Remove current node from graph
            g.remove_node(u)

            # Remove last list of d if empty (ensure termination of while loop)
            if not d[len(d)-1]:
                d.pop()

        return degeneracy_ordering
    else:
        raise nx.NetworkXPointlessConcept('The provided graph is empty.')


def find_one_maximal_clique_greedy(g, print_result=False):
    """
    Greedy algorithm to find and optionally print one random maximal clique in the given graph.

    Parameters
    ----------
    g : networkx.classes.graph.Graph
        A NetworkX undirected graph.
    print_result : bool, (optional, default: False)
        Flag for specifying whether the result should be printed on screen or not

    Returns
    ----------
    set
        A set containing a random maximal clique which can be found in the given graph.

    Raises
    -------
    networkx.NetworkXError
       If g is not a NetworkX graph.

    networkx.NetworkXPointlessConcept
        If g is an empty graph.

    Examples
    --------
    Find a maximal clique in graph g:
        find_one_maximal_clique(g)

    Find and print a maximal clique in graph g:
        find_one_maximal_clique(g, print_result=True)
    """

    # Check that g is a NetworkX graph
    if not isinstance(g, nx.classes.graph.Graph):
        raise nx.NetworkXError('The provided graph is not a valid NetworkX undirected graph.')

    if g.nodes:
        # Initialization
        vertices = list(g.nodes)
        start_node = random.choice(vertices)
        vertices.remove(start_node)
        clique = {start_node}

        # Greedy algorithm
        for v in vertices:
            valid = True
            for u in clique:
                if not g.has_edge(v, u):
                    valid = False
                    break
            if valid:
                clique.add(v)

        # Result & printing
        if len(clique) > 2:
            if print_result:
                print(clique)
            return clique
        else:
            return
    else:
        raise nx.NetworkXPointlessConcept('The provided graph is empty.')


def find_all_maximal_cliques_bk(g, variant='degeneracy', print_result=False):
    """
    Finds and optionally prints all maximal cliques in the given graph.

    This function implements three variants of the Bron-Kerbosch algorithm, which can be optionally specified by the following keywords:
        - classic: classic Bron-Kerbosch algorithm, with no optimizations;
        - tomita: Bron-Kerbosch with Tomita pivoting;
        - degeneracy: Bron-Kerbosch with Tomita pivoting and degeneracy ordering.

    If no variant is specified, the degeneracy ordering algorithm is used by default.

    Parameters
    ----------
    g : networkx.classes.graph.Graph
        A NetworkX undirected graph.
    variant : str, (optional, default: "degeneracy"; alternative options: "classic", "tomita")
        Specifies the variant of the Bron-Kerbosch algorithm that should be used to find the maximal cliques.
    print_result : bool, (optional, default: False)
        Flag for specifying whether the result should be printed on screen or not

    Returns
    ----------
    set
        A set containing a random maximal clique which can be found in the given graph.

    Raises
    -------
    networkx.NetworkXError
       If g is not a NetworkX graph.

    networkx.NetworkXPointlessConcept
        If g is an empty graph.

    ValueError, in bron_kerbosch_tomita_pivot()
       If p | x is empty.

    Warnings
    -------
    UserWarning
        If the "variant" parameter has not been passed a valid argument.

    Examples
    --------
    Find all maximal cliques in graph g:
        find_all_maximal_cliques(g)

    Find and print all maximal cliques in graph g:
        find_all_maximal_cliques(g, print_result=True)

    Find all maximal cliques in graph g using the classic Bron-Kerbosch algorithm:
        find_all_maximal_cliques(g, variant='classic')

    Find all maximal cliques in graph g using the Bron-Kerbosch algorithm with Tomita pivoting:
        find_all_maximal_cliques(g, variant='tomita')

    Find all maximal cliques in graph g using the Bron-Kerbosch algorithm with degeneracy ordering:
        find_all_maximal_cliques(g, variant='degeneracy')

    Find and print all maximal cliques in graph g using the Bron-Kerbosch algorithm with degeneracy ordering:
        find_all_maximal_cliques(g, variant='degeneracy', print_result=True)
    """
    
    # Check that g is a NetworkX graph
    if not isinstance(g, nx.classes.graph.Graph):
        raise nx.NetworkXError('The provided graph is not a valid NetworkX undirected graph.')

    # Classic Bron-Kerbosch algorithm
    def bron_kerbosch(r, p, x):
        """
        Implementation of the classic Bron-Kerbosch recursive algorithm for finding all maximal cliques in a graph.

        Sets r, p and x must be initialized to empty set, set of all nodes and empty set, respectively.

        Reference: Coen Bron, Joep Kerbosch, Algorithm 457: finding all cliques of an undirected graph, Communications of the ACM, vol. 16, issue 9 (Sept. 1973), https://dl.acm.org/doi/10.1145/362342.362367

        Parameters
        ----------
        r : set
            The set to be extended or shrunk by one node Nodes that are eligible to extend r, i.e. that are connected to all points in r, are collected recursively in the remaining two sets.
        p : set
            The set of candidates, or of all nodes that will in due time serve as an extension to the present configuration of r.
        x : set
            The set of all nodes that have already served as an extension of the present configuration of r and are now explicitly excluded.

        Returns
        ----------
        iterator
            An iterator containing all maximal cliques, in random order. Each clique is a set of nodes in the graph.
        """
        if not p and not x:
            if len(r) > 2:
                yield r
        for v in {*p}:
            yield from bron_kerbosch(r | {v}, p & {*g.neighbors(v)}, x & {*g.neighbors(v)})
            p = p - {v}
            x.add(v)

    # Bron-Kerbosch algorithm with Tomita pivoting
    def bron_kerbosch_tomita_pivot(r, p, x):
        """
        Implementation of the Bron-Kerbosch recursive algorithm with Tomita pivoting for finding all maximal cliques in a graph.

        It consists of the classic Bron-Kerbosch algorithm with the addition of a pivot vertex before the for loop, in order to reduce the number of recursive calls; the Tomita pivoting chooses the vertex u in p | x with the most neighbors in p as pivot.

        Sets r, p and x must be initialized to empty set, set of all nodes and empty set, respectively.

        Reference: Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi, The worst-case time complexity for generating all maximal cliques and computational experiments, Theoretical Computer Science, 363 (1): 28–42, 2006, https://www.sciencedirect.com/science/article/pii/S0304397506003586

        Parameters
        ----------
        r : set
            The set to be extended or shrunk by one node Nodes that are eligible to extend r, i.e. that are connected to all points in r, are collected recursively in the remaining two sets.
        p : set
            The set of candidates, or of all nodes that will in due time serve as an extension to the present configuration of r.
        x : set
            The set of all nodes that have already served as an extension of the present configuration of r and are now explicitly excluded.

        Returns
        ----------
        iterator
            An iterator containing all maximal cliques, in random order. Each clique is a set of nodes in the graph.

        Raises
        -------
        ValueError
           If p | x is empty.
        """
        if not p and not x:
            if len(r) > 2:
                yield r
        try:
            u = max({(v, len({n for n in g.neighbors(v) if n in p})) for v in p | x}, key=lambda v: v[1])[0]
            for v in p - {*g.neighbors(u)}:
                yield from bron_kerbosch_tomita_pivot(r | {v}, p & {*g.neighbors(v)}, x & {*g.neighbors(v)})
                p = p - {v}
                x.add(v)
        except ValueError:
            pass

    # Bron-Kerbosch algorithm with Tomita pivoting & degeneracy ordering
    def bron_kerbosch_degeneracy(r, p, x):
        """
        Implementation of the Bron-Kerbosch recursive algorithm with Tomita pivoting and degeneracy ordering for finding all maximal cliques in a graph.

        It consists of the classic Bron-Kerbosch algorithm with Tomita pivoting further improved by ordering the vertices in a degeneracy ordering, a particular ordering of the vertices of a graph which minimizes the number of recursive calls.

        Reference: David Eppstein, Maarten Löffler, Darren Strash, Listing All Maximal Cliques in Sparse Graphs in Near-Optimal Time, Algorithms and Computation, ISAAC 2010, Lecture Notes in Computer Science (vol 6506), Springer, https://doi.org/10.1007/978-3-642-17517-6_36

        Parameters
        ----------
        r : set
            The set to be extended or shrunk by one node Nodes that are eligible to extend r, i.e. that are connected to all points in r, are collected recursively in the remaining two sets.
        p : set
            The set of candidates, or of all nodes that will in due time serve as an extension to the present configuration of r.
        x : set
            The set of all nodes that have already served as an extension of the present configuration of r and are now explicitly excluded.

        Returns
        ----------
        iterator
            An iterator containing all maximal cliques, in random order. Each clique is a set of nodes in the graph.

        Raises
        -------
        ValueError, in bron_kerbosch_tomita_pivot()
           If p | x is empty.
        """
        for v in get_degeneracy_ordering(g):
            yield from bron_kerbosch_tomita_pivot(r | {v}, p & {*g.neighbors(v)}, x & {*g.neighbors(v)})
            p = p - {v}
            x.add(v)

    # Main clique function
    if g.nodes:
        # Set initialization
        r = {*()}
        p = {*g.nodes}
        x = {*()}

        # Bron-Kerbosch algorithm
        if variant == 'classic':
            cliques = bron_kerbosch(r, p, x)
        elif variant == 'tomita':
            cliques = bron_kerbosch_tomita_pivot(r, p, x)
        elif variant == 'degeneracy':
            cliques = bron_kerbosch_degeneracy(r, p, x)
        else:
            warnings.warn('Invalid algorithm variant (\'{}\'). Using Bron-Kerbosch with degeneracy ordering as default.'.format(variant))
            cliques = bron_kerbosch_degeneracy(r, p, x)

        # Printing
        if print_result:
            print(*cliques, sep='\n')

        return cliques
    else:
        raise nx.NetworkXPointlessConcept('The provided graph is empty.')


def sample_random_subgraph(g, n):
    """
    Samples a subgraph of n nodes from the given NetworkX undirected graph g.

    Parameters
    ----------
    g : networkx.classes.graph.Graph
        A NetworkX undirected graph.

    n : int
        The number of nodes to be extracted from the graph.

    Returns
    ----------
    networkx.classes.graph.Graph
        A NetworkX undirected graph.

    Raises
    -------
    networkx.NetworkXError
       If g is not a NetworkX graph.

    networkx.NetworkXPointlessConcept
        If g is an empty graph.
    """

    # Check that g is a NetworkX graph
    if not isinstance(g, nx.classes.graph.Graph):
        raise nx.NetworkXError('The provided graph is not a valid NetworkX undirected graph.')

    # Check that g is not empty
    if g.nodes:
        return g.subgraph(random.sample(g.nodes, n))
    else:
        raise nx.NetworkXPointlessConcept('The provided graph is empty.')


def find_maximum_clique(x, print_result=False):
    """
    Returns and optionally prints the maximum clique (NOT maximal) from the given undirected NetworkX graph g, or a list of cliques.

    Parameters
    ----------
    x : networkx.classes.graph.Graph
        A NetworkX undirected graph.

    x : list
        A list of maximal cliques (containing either lists or sets of nodes).

    Returns
    ----------
    set
        A set containing the nodes in the maximum clique.

    Raises
    -------
    networkx.NetworkXError, in find_all_maximal_cliques_bk()
       If g is not a NetworkX graph.

    networkx.NetworkXPointlessConcept, in find_all_maximal_cliques_bk()
        If g is an empty graph.

    ValueError, in find_all_maximal_cliques_bk() [in bron_kerbosch_tomita_pivot()]
       If p | x is empty.

    Examples
    --------
    Find the maximum clique in graph g:
        find_maximum_clique(g)

    Find and print the maximum clique in graph g:
        find_maximum_clique(g, print_result=True)

    Find the maximum clique in list of cliques c:
        find_maximum_clique(c)

    Find and print the maximum clique in list of cliques c:
        find_maximum_clique(c, print_result=True)
    """

    # Check input type
    if isinstance(x, list):
        cliques = x
    else:
        cliques = list(find_all_maximal_cliques_bk(x))

    # Find maximum clique and convert to set (if input is a list of lists)
    maximum_clique = set(cliques[np.argmax(np.array([len(c) for c in cliques]))])

    # Printing
    if print_result:
        print(maximum_clique)

    return maximum_clique


######################################################################


def main(args):
    """
    Main function of the OEIS Comments Authors Max Clique project.

    Analyzes a sample OEIS sequence file, then builds a graph of all authors from the comments of all sequences in order to examine it with a series of graph algorithms.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments, usable as args.argument_name.
    """

    # Load sample sequence file
    print('\n' + 'Printing sample OEIS JSON file...')

    file = load_json('data/sequences/A000001.json', print_result=True)

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

    # Find and print one maximal clique
    find_one_maximal_clique_greedy(g, print_result=True)

    # Efficiency of set() vs. {*()}
    print('\n' + 'Efficiency of {*()} vs. set():')

    number_set = 1000000
    empty_literal_time = (timeit.timeit('{*()}', number=number_set)) / number_set
    set_time = (timeit.timeit('set()', number=number_set)) / number_set

    print('- empty literal execution time: {} s.'.format(empty_literal_time))
    print('- set constructor execution time: {} s.'.format(set_time))
    if empty_literal_time < set_time:
        print('Empty literal is faster than set constructor.')
    else:
        print('Set constructor is faster than empty literal.')

    # Find and print all maximal cliques in a random subgraph of 100 nodes
    subgraph = sample_random_subgraph(g, 100)
    find_all_maximal_cliques_bk(subgraph, print_result=True)

    # Efficiency of different Bron-Kerbosch variants
    print('\n' + 'Efficiency of different Bron-Kerbosch variants:')

    bk_classic_start = timeit.default_timer()
    bk_classic_cliques = find_all_maximal_cliques_bk(subgraph, variant='classic')
    bk_classic_end = timeit.default_timer()

    bk_tomita_start = timeit.default_timer()
    bk_tomita_cliques = find_all_maximal_cliques_bk(subgraph, variant='tomita')
    bk_tomita_end = timeit.default_timer()

    bk_degeneracy_start = timeit.default_timer()
    bk_degeneracy_cliques = find_all_maximal_cliques_bk(subgraph, variant='degeneracy')
    bk_degeneracy_end = timeit.default_timer()

    bk_classic_time = bk_classic_end - bk_classic_start
    bk_tomita_time = bk_tomita_end - bk_tomita_start
    bk_degeneracy_time = bk_degeneracy_end - bk_degeneracy_start

    print('- Bron-Kerbosch classic execution time: {} s.'.format(bk_classic_time))
    print('- Bron-Kerbosch with Tomita pivoting execution time: {} s.'.format(bk_tomita_time))
    print('- Bron-Kerbosch with degeneracy ordering execution time: {} s.'.format(bk_degeneracy_time))
    if bk_classic_time < bk_tomita_time and bk_classic_time < bk_degeneracy_time:
        print('Bron-Kerbosch classic is faster than the other two variants.')
    elif bk_tomita_time < bk_classic_time and bk_tomita_time < bk_degeneracy_time:
        print('Bron-Kerbosch with Tomita pivoting is faster than the other two variants.')
    elif bk_degeneracy_time < bk_classic_time and bk_degeneracy_time < bk_tomita_time:
        print('Bron-Kerbosch with degeneracy ordering is faster than the other two variants.')

    # Correctness of different Bron-Kerbosch variants
    print('\n' + 'Checking the correctness of different Bron-Kerbosch variants... ', end='')

    bk_classic_cliques = list(bk_classic_cliques)
    bk_tomita_cliques = list(bk_tomita_cliques)
    bk_degeneracy_cliques = list(bk_degeneracy_cliques)

    correctness_flag = False

    if list(filter(lambda c: c not in bk_classic_cliques, bk_tomita_cliques)) or list(filter(lambda c: c not in bk_tomita_cliques, bk_classic_cliques)):
        print('the cliques returned by the classic Bron-Kerbosch algorithm are different from those generated by the Tomita pivoting variant.')
    elif list(filter(lambda c: c not in bk_classic_cliques, bk_degeneracy_cliques)) or list(filter(lambda c: c not in bk_degeneracy_cliques, bk_classic_cliques)):
        print('the cliques returned by the classic Bron-Kerbosch algorithm are different from those generated by the degeneracy ordering variant.')
    elif list(filter(lambda c: c not in bk_classic_cliques, bk_degeneracy_cliques)) or list(filter(lambda c: c not in bk_degeneracy_cliques, bk_classic_cliques)):
        print('the cliques returned by the Tomita pivoting Bron-Kerbosch algorithm are different from those generated by the degeneracy ordering variant.')
    else:
        correctness_flag = True
        print('the cliques returned by all three algorithms are identical.')

    if correctness_flag:
        print('All implemeneted variants are correct.')
    else:
        print('There has been an error in the implementation of the Bron-Kerbosch algorithms.')

    # Find maximum clique
    maximum_clique = find_maximum_clique(bk_degeneracy_cliques, print_result=True)
    print('\n' + 'The maximum clique of the random subgraph has length {} and contains nodes: \n{}.'.format(len(maximum_clique), maximum_clique), end='')

######################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script for the OEIS Comments Authors Max Clique project.')
    parser.add_argument('-bg', '--build_graph', choices=('True', 'False'), default='False', help='Build graph from raw data, otherwise load it from the "data/comments_authors_graph.json" file.')

    args = parser.parse_args()

    main(args)
