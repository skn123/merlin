// Unit tests for merlin::graph (include/graph.h + src/graph.cpp).
// This TU requires src/graph.cpp to be linked (edge_id::NO_EDGE and the
// triangulate/init/maximal_cliques definitions live there).

#include <gtest/gtest.h>
#include "graph.h"

using merlin::graph;
using merlin::edge_id;

TEST(Graph, ConstructWithNodes) {
    graph g(4);
    EXPECT_EQ(g.num_nodes(), 4u);
    EXPECT_EQ(g.num_edges(), 0u);
}

TEST(Graph, AddNode) {
    graph g(0);
    size_t a = g.add_node();
    size_t b = g.add_node();
    EXPECT_NE(a, b);
    EXPECT_EQ(g.num_nodes(), 2u);
}

TEST(Graph, AddEdgeIsUndirectedAndIdempotent) {
    graph g(3);
    g.add_edge(0, 1);
    EXPECT_EQ(g.num_edges(), 1u);

    // Edge exists in both directions.
    EXPECT_NE(g.edge(0, 1), edge_id::NO_EDGE);
    EXPECT_NE(g.edge(1, 0), edge_id::NO_EDGE);

    // Adding the same edge again does not create a second edge.
    g.add_edge(0, 1);
    EXPECT_EQ(g.num_edges(), 1u);
}

TEST(Graph, MissingEdgeIsNoEdge) {
    graph g(3);
    g.add_edge(0, 1);
    EXPECT_EQ(g.edge(0, 2), edge_id::NO_EDGE);
}

TEST(Graph, RemoveEdge) {
    graph g(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    ASSERT_NE(g.edge(0, 1), edge_id::NO_EDGE);

    g.remove_edge(0, 1);
    EXPECT_EQ(g.edge(0, 1), edge_id::NO_EDGE);
    // The other edge is untouched.
    EXPECT_NE(g.edge(1, 2), edge_id::NO_EDGE);
}

TEST(Graph, NeighborsReflectAddedEdges) {
    graph g(4);
    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(0, 3);
    // Node 0 has three incident edges.
    EXPECT_EQ(g.neighbors(0).size(), 3u);
    // A node with no edges has no neighbors.
    graph h(2);
    EXPECT_EQ(h.neighbors(1).size(), 0u);
}

TEST(Graph, ClearEdges) {
    graph g(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.clear_edges();
    EXPECT_EQ(g.num_edges(), 0u);
    EXPECT_EQ(g.edge(0, 1), edge_id::NO_EDGE);
}
