#!/usr/bin/env python
# coding: utf-8

import click
import os
import sys
import codecs
import numpy
import json
import regex
import tfidf
from collections import Counter, OrderedDict
from lib.stopwords import StopWords



def compute_tfidf_matrix(corpus_dir, language):
    sw = set(StopWords.words(language))


    t = tfidf.tfidf()
    for path, subdirs, files in os.walk(corpus_dir):
        for name in files:
            f = os.path.join(path, name)
            with codecs.open(f, 'rb', 'utf-8') as i:
                tokens = []
                for line in i:
                    # Skip <doc> tags
                    if not regex.match(r'</?doc', line):
                        l_tokens = regex.split(r'[^\p{L}]+', line.lower())
                        tokens += [token for token in l_tokens if token and token not in sw]

                t.addDocument(f, tokens)
    return t


def dump_tfidf(ranking, outfile='tfidf.json'):
    with open(outfile, 'wb') as f:
        json.dump(ranking, f, indent=2)
    return 0


def get_distributions(tokens, tfidf_matrix, threshold):
    variances = {}
    stdevs = {}
    threshold_rank = {}
    tfidf_ranking = {}
    for token in tokens:
        relevance = 0
        ranking = tfidf_matrix.similarities([token])
        non_null = {doc: score for (doc, score) in ranking if score}
        ordered = OrderedDict(sorted(non_null.items(), key=lambda x: x[1], reverse=True))
        tfidf_ranking[token] = ordered
        scores = [pair[1] for pair in ranking]
        variances[token] = numpy.var(scores)
        stdevs[token] = numpy.std(scores)
        for score in scores:
            if score > threshold:
                relevance += 1
        threshold_rank[token] = relevance

    return (OrderedDict(sorted(variances.items(), key=lambda x: x[1], reverse=True)),
           OrderedDict(sorted(stdevs.items(), key=lambda x: x[1], reverse=True)),
           OrderedDict(sorted(threshold_rank.items(), key=lambda x: x[1], reverse=True)),
           tfidf_ranking)


@click.command()
@click.option('--language', default='english',
              help='Language for stopword filtering (default: english)')


@click.argument('corpus', type=click.Path(exists=True, file_okay=False))
@click.argument('tokens', type=click.File('r'))
@click.option('--threshold', '-t', default=0.6)
@click.option('--dump-tfidf/--no-dump-tfidf', default=False)
@click.option('--variance-out', type=click.File('w'), default='variances.json')
@click.option('--stdevs-out', type=click.File('w'), default='stdevs.json')
@click.option('--threshold-rank-out', type=click.File('w'), default='threshold_rank.json')
@click.option('--tfidf-rank-out', type=click.File('w'), default='tfidf.json')
def main(language, corpus, tokens, threshold, dump_tfidf,
         variance_out, stdevs_out, threshold_rank_out, tfidf_rank_out):


    print ("Loading tokens...")
    tokens = [token.strip() for token in tokens]

    print ("Building TF/IDF matrix against corpus %s ..." % corpus)
    t = compute_tfidf_matrix(corpus, language)


    print ("Computing variance, standard deviation, and ranking")
    vars, stdevs, threshold_rank, tfidf_ranking = get_distributions(tokens, t, threshold)

    print ("Dumping results to JSON ...")
    json.dump(vars, variance_out, indent=2)
    json.dump(stdevs, stdevs_out, indent=2)
    json.dump(threshold_rank, threshold_rank_out, indent=2)

    if dump_tfidf:
        json.dump(tfidf_ranking, tfidf_rank_out, indent=2)


if __name__ == "__main__":
    main()
