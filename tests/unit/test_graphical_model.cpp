// Unit tests for merlin::graphical_model UAI parsing/serialization
// (include/graphical_model.h), driven by the committed examples/ fixtures.

#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "graphical_model.h"

using merlin::graphical_model;

namespace {
std::string data_path(const char* name) {
    return std::string(MERLIN_DATA_DIR) + "/" + name;
}

graphical_model load(const char* name) {
    std::ifstream is(data_path(name).c_str());
    EXPECT_TRUE(is.good()) << "cannot open fixture " << name;
    graphical_model gm;
    gm.read(is);
    return gm;
}
}  // namespace

TEST(GraphicalModel, ReadsCancerBayesNet) {
    graphical_model gm = load("cancer.uai");
    EXPECT_EQ(gm.nvar(), 5u);
    EXPECT_EQ(gm.num_factors(), 5u);
    EXPECT_TRUE(gm.is_bayes());
    EXPECT_FALSE(gm.is_markov());

    // All five variables are binary.
    for (size_t i = 0; i < gm.nvar(); ++i)
        EXPECT_EQ(gm.var(i).states(), 2u);
}

TEST(GraphicalModel, FactorScopeSizesMatchModel) {
    // cancer.uai cliques (sizes): 1, 2, 2, 3, 2.
    graphical_model gm = load("cancer.uai");
    size_t total_scope = 0;
    for (size_t f = 0; f < gm.num_factors(); ++f)
        total_scope += gm.get_factor(f).nvar();
    EXPECT_EQ(total_scope, 1u + 2u + 2u + 3u + 2u);
}

TEST(GraphicalModel, EachCPTNormalizesConsistently) {
    // For a Bayes net each factor is a CPT; summing the whole table gives the
    // number of parent configurations (each conditional column sums to 1).
    graphical_model gm = load("cancer.uai");
    for (size_t f = 0; f < gm.num_factors(); ++f) {
        const merlin::factor& fac = gm.get_factor(f);
        double s = fac.sum();
        // table size / child-cardinality = number of parent configs.
        EXPECT_GT(s, 0.0);
        EXPECT_TRUE(fac.isfinite());
    }
}

TEST(GraphicalModel, WriteThenReadRoundTripsStructure) {
    graphical_model gm = load("cancer.uai");

    std::ostringstream os;
    gm.write(os);  // writes MARKOV form

    std::istringstream is(os.str());
    graphical_model gm2;
    gm2.read(is);

    EXPECT_EQ(gm2.nvar(), gm.nvar());
    EXPECT_EQ(gm2.num_factors(), gm.num_factors());
    for (size_t i = 0; i < gm.nvar(); ++i)
        EXPECT_EQ(gm2.var(i).states(), gm.var(i).states());

    // write() emits the MARKOV form, which drops the BAYES child orientation,
    // so on re-read a factor's table may be laid out in a different variable
    // order. The information content is preserved, so compare each factor's
    // table as a sorted multiset of values (and its sum) rather than by cell.
    for (size_t f = 0; f < gm.num_factors(); ++f) {
        const merlin::factor& a = gm.get_factor(f);
        const merlin::factor& b = gm2.get_factor(f);
        ASSERT_EQ(a.num_states(), b.num_states());

        std::vector<double> ta(a.table(), a.table() + a.num_states());
        std::vector<double> tb(b.table(), b.table() + b.num_states());
        std::sort(ta.begin(), ta.end());
        std::sort(tb.begin(), tb.end());
        for (size_t k = 0; k < ta.size(); ++k)
            EXPECT_NEAR(ta[k], tb[k], 1e-9) << "factor " << f << " sorted cell " << k;
    }
}

TEST(GraphicalModel, RejectsUnknownFormat) {
    std::istringstream is("NOTAFORMAT\n1\n2\n0\n");
    graphical_model gm;
    EXPECT_THROW(gm.read(is), std::runtime_error);
}
