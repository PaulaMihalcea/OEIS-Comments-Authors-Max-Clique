# OEIS: Maximal Cliques of Authors from Comments
## Author: Paula Mihalcea
#### Università degli Studi di Firenze

---

![](https://img.shields.io/github/repo-size/PaulaMihalcea/OEIS-Maximal-Cliques-of-Authors-from-Comments)

**[OEIS](https://oeis.org/)** is the online encyclopaedia of **integer sequences**. It lists thousands of number sequences in lexicographic order, such as the [prime numbers](http://oeis.org/A000040) or the [Fibonacci sequence](http://oeis.org/A000045), easing the work of countless researchers since 1964, its foundation year.

The OEIS is made of a series of **JSON files**, one for each integer sequence. Given their regular, human-readable format, these files can be easily manipulated in order to further analyze them. Indeed, each page of the OEIS not only lists the integers of the corresponding sequence, but also a series of information such as formulas, references, links and comments.

This work aims to create, step-by-step, a **[Python 3](https://www.python.org/)** script capable of loading these files and parsing their content in order to build a **graph** where:
- **nodes** represent all unique **authors** that can be found in each comment of every sequence, and
- **edges** link two authors who have **commented the same sequence**.

Three main algorithms are then implemented in order to find:
1. a **maximal clique**;
2. a list of **all maximal cliques**;
3. the **maximum clique**.

## Usage

The complete code for this project is provided as a standalone, fully documented module in the [`mihalcea.py`](mihalcea.py) Python file, which can be executed from the command line as follows:

- run the script and use the graph in [`data/comments_authors_graph.json`](data/comments_authors_graph.json):

```
python mihalcea.py
```

- run the script and build the graph from the raw JSON files in [`data/sequences`](data/sequences):

```
python mihalcea.py --build_graph True
```

Additional documentation can be found in the Jupyter notebook [`docs/mihalcea.ipynb`](docs/mihalcea.ipynb), which describes step-by-step the code of the main script and contains references to the resources used for its implementation. This guide is also provided as a PDF in [`docs/mihalcea.pdf`](docs/mihalcea.pdf).

### Requirements

The following Python packages are needed in order to run this project:

- [`argparse`](https://docs.python.org/3/library/argparse.html)
- [`itertools`](https://docs.python.org/3/library/itertools.html)
- [`json`](https://docs.python.org/3/library/json.html)
- [`networkx`](https://networkx.org/)
- [`numpy`](https://numpy.org/)
- [`os`](https://docs.python.org/3/library/os.html)
- [`random`](https://docs.python.org/3/library/random.html)
- [`re`](https://docs.python.org/3/library/re.html)
- [`timeit`](https://docs.python.org/3/library/timeit.html)
- [`tqdm`](https://tqdm.github.io/)
- [`warnings`](https://docs.python.org/3/library/warnings.html)

## License
This work is licensed under a [Creative Commons “Attribution-NonCommercial-ShareAlike 4.0 International”](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license. More details are available in the [LICENSE](./LICENSE) file.
