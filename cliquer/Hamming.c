//
// Created by 王台宜 on 4/17/20.
//
#include <stdio.h>
#include "cliquer.h"

/* Function for determining the Hamming distance between
   two words, c1 and c2, of length n */
int distance(int c1, int c2, int n) {
    int i, c, d;

    d = 0;
    c = c1 ^ c2;

    for (i = 0; i < n; i++) {
        if(c & 1) d++;
        c >>= 1;
    }
    return d;
}

int main(int argc, char *argv[]) {
    int i, j;
    int n, M, d;
    int order;
    int nr;
    graph_t *g;

    if (argc!=4) {
        fprintf(stderr,"%s <length> <size> <dist>\n",argv[0]);
        return 1;
    }
    n = atoi(argv[1]);
    M = atoi(argv[2]);
    d = atoi(argv[3]);

    /* no timer */
    clique_default_options->time_function = NULL;

    order = 1 << n;
    g = graph_new(order);
    for (i = 0; i < order; i++)
        for (j = i+1; j < order; j++)
            if(distance(i,j,n) >= d)
                GRAPH_ADD_EDGE(g,i,j);
    nr = clique_find_all(g,M,M,FALSE,clique_default_options);
    printf("Number of codes: %d\n",nr);
    return 0;
}
