import os
import random
import re
import sys
from collections import defaultdict

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerate.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    df_inv = 1 - damping_factor
    prob_any_page = df_inv / len(corpus.keys())
    next_page_prob = damping_factor * (1/len(corpus[page]))
    res = dict()
    for key in corpus:
        res[key] = prob_any_page
        if key in corpus[page]:
            res[key] += next_page_prob
    return res


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    res = dict()
    for k in corpus:
        res[k] = 0

    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        res[page] += 1
        trans = transition_model(corpus, page, damping_factor)
        population = []
        weights = []
        for k, v in trans.items():
            population.append(k)
            weights.append(v)
        page = random.choices(population, weights=weights)[0]
    
    for k in res:
        res[k] = res[k] / n
        
    return res


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    linking = defaultdict(set)
    pr = dict()
    n = len(corpus.keys())
    for k, v in corpus.items():
        pr[k] = 1/n
        for page in v:
            linking[page].add(k)
    
    prob_of_any_page = (1-damping_factor) / n
    convergence_threshold = 0.001

    while True:

        converged = True

        for k in corpus:
            prev = pr[k]
            pr[k] = prob_of_any_page + sum( pr[page] / len(corpus[page]) for page in linking[k] ) * damping_factor

            if abs(prev - pr[k]) > convergence_threshold:
                converged = False

        if converged:
            break
        
    return pr


if __name__ == "__main__":
    #res = sample_pagerank({"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}},  DAMPING, SAMPLES)
    # res = iterate_pagerank({"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}, DAMPING)
    # print(res)
    main()
