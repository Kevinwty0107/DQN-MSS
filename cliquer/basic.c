//
// Created by 王台宜 on 4/17/20.
//


#include <stdio.h>
#include "cliquer.h"

int main(int argc, char *argv[]) {
    graph_t *g;
    set_t s;

    if (argc!=2) {
        fprintf(stderr,"%s <dimacs_file>\n",argv[0]);
        return 1;
    }
    g=graph_read_dimacs_file(argv[1]);
    if (g==NULL)
        return 1;

    ASSERT(graph_test(g,stderr));

    s=clique_find_single(g,0,0,FALSE,NULL);
    set_print(s);

    return 0;
}