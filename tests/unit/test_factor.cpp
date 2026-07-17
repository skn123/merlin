// Unit tests for merlin::factor (include/factor.h).
// Only deterministic operations are exercised; randomize()/sample() are avoided.

#include <gtest/gtest.h>
#include "variable_set.h"
#include "factor.h"

using merlin::variable;
using merlin::variable_set;
using merlin::factor;

namespace {

// Build a factor over two binary variables (labels 0 and 1) with a known table.
// The table is laid out little-endian: index = x0 + x1 * dims[0], so with both
// binary the order is (x0,x1) = (0,0),(1,0),(0,1),(1,1).
factor make_2x2(double v00, double v10, double v01, double v11) {
    variable x0(0, 2), x1(1, 2);
    variable_set vs(x0, x1);
    factor f(vs, 0.0);
    f[0] = v00;  // x0=0, x1=0
    f[1] = v10;  // x0=1, x1=0
    f[2] = v01;  // x0=0, x1=1
    f[3] = v11;  // x0=1, x1=1
    return f;
}

}  // namespace

TEST(Factor, ScalarConstructor) {
    factor f(2.5);
    EXPECT_TRUE(f.isscalar());
    EXPECT_EQ(f.nvar(), 0u);
    EXPECT_EQ(f.num_states(), 1u);
    EXPECT_DOUBLE_EQ(f[0], 2.5);
}

TEST(Factor, ConstantOverScope) {
    variable_set vs(variable(0, 2), variable(1, 3));
    factor f(vs, 0.5);
    EXPECT_EQ(f.nvar(), 2u);
    EXPECT_EQ(f.num_states(), 6u);
    for (size_t i = 0; i < f.num_states(); ++i)
        EXPECT_DOUBLE_EQ(f[i], 0.5);
}

TEST(Factor, SumOverAll) {
    factor f = make_2x2(0.1, 0.2, 0.3, 0.4);
    EXPECT_DOUBLE_EQ(f.sum(), 1.0);
}

TEST(Factor, MaxMinArgmax) {
    factor f = make_2x2(0.1, 0.4, 0.3, 0.2);
    EXPECT_DOUBLE_EQ(f.max(), 0.4);
    EXPECT_DOUBLE_EQ(f.min(), 0.1);
    EXPECT_EQ(f.argmax(), 1u);  // value 0.4 sits at index 1
    EXPECT_EQ(f.argmin(), 0u);
}

TEST(Factor, Normalized) {
    factor f = make_2x2(1.0, 1.0, 1.0, 1.0);
    factor n = f.normalized();
    EXPECT_DOUBLE_EQ(n.sum(), 1.0);
    for (size_t i = 0; i < n.num_states(); ++i)
        EXPECT_DOUBLE_EQ(n[i], 0.25);
    // Original is unchanged.
    EXPECT_DOUBLE_EQ(f.sum(), 4.0);
}

TEST(Factor, LogPartition) {
    factor f = make_2x2(0.25, 0.25, 0.25, 0.25);
    EXPECT_NEAR(f.logpartition(), std::log(1.0), 1e-12);
}

TEST(Factor, MarginalToSingleVariable) {
    // table (x0,x1): (0,0)=0.1 (1,0)=0.2 (0,1)=0.3 (1,1)=0.4
    factor f = make_2x2(0.1, 0.2, 0.3, 0.4);

    // Marginal over x0: sum over x1.
    //   P(x0=0) = 0.1 + 0.3 = 0.4 ;  P(x0=1) = 0.2 + 0.4 = 0.6
    factor m0 = f.marginal(variable_set(variable(0, 2)));
    ASSERT_EQ(m0.num_states(), 2u);
    EXPECT_NEAR(m0[0], 0.4, 1e-12);
    EXPECT_NEAR(m0[1], 0.6, 1e-12);

    // Marginal over x1: sum over x0.
    //   P(x1=0) = 0.1 + 0.2 = 0.3 ;  P(x1=1) = 0.3 + 0.4 = 0.7
    factor m1 = f.marginal(variable_set(variable(1, 2)));
    ASSERT_EQ(m1.num_states(), 2u);
    EXPECT_NEAR(m1[0], 0.3, 1e-12);
    EXPECT_NEAR(m1[1], 0.7, 1e-12);
}

TEST(Factor, SumOutEqualsMarginalOfComplement) {
    factor f = make_2x2(0.1, 0.2, 0.3, 0.4);
    factor viaSumOut = f.sum(variable_set(variable(1, 2)));   // sum out x1
    factor viaMarginal = f.marginal(variable_set(variable(0, 2)));
    ASSERT_EQ(viaSumOut.num_states(), viaMarginal.num_states());
    for (size_t i = 0; i < viaSumOut.num_states(); ++i)
        EXPECT_NEAR(viaSumOut[i], viaMarginal[i], 1e-12);
}

TEST(Factor, ScalarArithmetic) {
    factor f = make_2x2(0.1, 0.2, 0.3, 0.4);
    factor g = f * 2.0;
    for (size_t i = 0; i < g.num_states(); ++i)
        EXPECT_NEAR(g[i], f[i] * 2.0, 1e-12);
    EXPECT_NEAR(g.sum(), 2.0, 1e-12);
}

TEST(Factor, ProductOverSameScope) {
    factor a = make_2x2(1.0, 2.0, 3.0, 4.0);
    factor b = make_2x2(2.0, 2.0, 2.0, 2.0);
    factor p = a * b;
    ASSERT_EQ(p.num_states(), 4u);
    for (size_t i = 0; i < p.num_states(); ++i)
        EXPECT_NEAR(p[i], a[i] * b[i], 1e-12);
}

TEST(Factor, ProductOfDisjointScopesIsOuterProduct) {
    // f(x0) over binary x0, g(x1) over binary x1 -> joint over {x0,x1}.
    variable_set vx0(variable(0, 2));
    variable_set vx1(variable(1, 2));
    factor f(vx0, 0.0); f[0] = 0.2; f[1] = 0.8;
    factor g(vx1, 0.0); g[0] = 0.5; g[1] = 0.5;

    factor joint = f * g;
    ASSERT_EQ(joint.nvar(), 2u);
    ASSERT_EQ(joint.num_states(), 4u);
    // joint(x0,x1) = f(x0)*g(x1), little-endian index = x0 + 2*x1.
    EXPECT_NEAR(joint[0], 0.2 * 0.5, 1e-12);  // (0,0)
    EXPECT_NEAR(joint[1], 0.8 * 0.5, 1e-12);  // (1,0)
    EXPECT_NEAR(joint[2], 0.2 * 0.5, 1e-12);  // (0,1)
    EXPECT_NEAR(joint[3], 0.8 * 0.5, 1e-12);  // (1,1)
}

TEST(Factor, DistanceToSelfIsZero) {
    factor f = make_2x2(0.1, 0.2, 0.3, 0.4);
    EXPECT_NEAR(f.distance(f, merlin::factor::Distance::L2), 0.0, 1e-12);
    EXPECT_NEAR(f.distance(f, merlin::factor::Distance::L1), 0.0, 1e-12);
}

TEST(Factor, L1DistanceKnownValue) {
    factor a = make_2x2(0.0, 0.0, 0.0, 0.0);
    factor b = make_2x2(0.1, 0.2, 0.3, 0.4);
    // L1 distance = sum of absolute differences = 1.0.
    EXPECT_NEAR(a.distance(b, merlin::factor::Distance::L1), 1.0, 1e-12);
}
