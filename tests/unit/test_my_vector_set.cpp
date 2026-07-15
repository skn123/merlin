// Unit tests for merlin::my_vector (include/vector.h) and merlin::my_set
// (include/set.h).

#include <gtest/gtest.h>
#include "vector.h"
#include "set.h"

using merlin::my_vector;
using merlin::my_set;

TEST(MyVector, ConstructionAndAccess) {
    my_vector<int> v(3, 7);
    EXPECT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 7);
    EXPECT_EQ(v[2], 7);
}

TEST(MyVector, LexicographicComparison) {
    my_vector<int> a(2, 1);
    my_vector<int> b(2, 1);
    b[1] = 2;
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a == b);

    my_vector<int> a_copy(a);
    EXPECT_TRUE(a == a_copy);
}

TEST(MySet, StartsEmpty) {
    my_set<int> s;
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0u);
}

TEST(MySet, AddKeepsSortedAndUnique) {
    my_set<int> s;
    s.add(5);
    s.add(1);
    s.add(3);
    s.add(1);  // duplicate ignored

    ASSERT_EQ(s.size(), 3u);
    EXPECT_EQ(s[0], 1);  // sorted ascending
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(s[2], 5);
    EXPECT_EQ(s.front(), 1);
    EXPECT_EQ(s.back(), 5);
}

TEST(MySet, RemoveAndFind) {
    my_set<int> s;
    s.add(2);
    s.add(4);
    s.add(6);

    EXPECT_NE(s.find(4), s.end());
    s.remove(4);
    EXPECT_EQ(s.find(4), s.end());
    EXPECT_EQ(s.size(), 2u);
}

TEST(MySet, UnionIntersectionDifference) {
    my_set<int> a;
    a.add(1); a.add(2); a.add(3);
    my_set<int> b;
    b.add(2); b.add(3); b.add(4);

    my_set<int> u = a | b;
    ASSERT_EQ(u.size(), 4u);
    EXPECT_EQ(u[0], 1); EXPECT_EQ(u[3], 4);

    my_set<int> inter = a & b;
    ASSERT_EQ(inter.size(), 2u);
    EXPECT_EQ(inter[0], 2); EXPECT_EQ(inter[1], 3);

    my_set<int> diff = a - b;
    ASSERT_EQ(diff.size(), 1u);
    EXPECT_EQ(diff[0], 1);

    my_set<int> sym = a ^ b;  // symmetric difference: {1, 4}
    ASSERT_EQ(sym.size(), 2u);
    EXPECT_EQ(sym[0], 1); EXPECT_EQ(sym[1], 4);
}

TEST(MySet, Comparison) {
    my_set<int> a; a.add(1); a.add(2);
    my_set<int> a_copy; a_copy.add(2); a_copy.add(1);  // insertion order differs
    EXPECT_TRUE(a == a_copy);  // sets are order-independent once sorted
}
