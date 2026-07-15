// Unit tests for merlin::indexed_heap (include/indexed_heap.h).
// The default constructor + insert() path is used; the templated range
// constructor is intentionally avoided (it reads an uninitialized max-id).

#include <gtest/gtest.h>
#include "indexed_heap.h"

using merlin::indexed_heap;

TEST(IndexedHeap, StartsEmpty) {
    indexed_heap h;
    EXPECT_TRUE(h.empty());
    EXPECT_EQ(h.size(), 0u);
}

TEST(IndexedHeap, TopReturnsHighestPriority) {
    indexed_heap h;
    h.insert(1.0, /*id=*/0);
    h.insert(5.0, /*id=*/1);
    h.insert(3.0, /*id=*/2);

    EXPECT_EQ(h.size(), 3u);
    std::pair<double, size_t> t = h.top();
    EXPECT_DOUBLE_EQ(t.first, 5.0);
    EXPECT_EQ(t.second, 1u);
}

TEST(IndexedHeap, PopReturnsInDescendingPriorityOrder) {
    indexed_heap h;
    h.insert(1.0, 0);
    h.insert(5.0, 1);
    h.insert(3.0, 2);
    h.insert(4.0, 3);

    double prev = 1e300;
    while (!h.empty()) {
        double p = h.top().first;
        EXPECT_LE(p, prev);
        prev = p;
        h.pop();
    }
    EXPECT_TRUE(h.empty());
}

TEST(IndexedHeap, ReprioritizeExistingId) {
    indexed_heap h;
    h.insert(1.0, 0);
    h.insert(2.0, 1);
    ASSERT_EQ(h.top().second, 1u);

    // Re-insert id 0 with a higher priority; it should now be on top.
    h.insert(9.0, 0);
    EXPECT_EQ(h.size(), 2u);
    EXPECT_EQ(h.top().second, 0u);
    EXPECT_DOUBLE_EQ(h.top().first, 9.0);
}

TEST(IndexedHeap, EraseById) {
    indexed_heap h;
    h.insert(1.0, 0);
    h.insert(5.0, 1);
    h.insert(3.0, 2);

    h.erase(1);  // erase the current max
    EXPECT_EQ(h.size(), 2u);
    EXPECT_EQ(h.top().second, 2u);  // next highest is id 2 (priority 3)

    // Erasing a missing id is a no-op.
    h.erase(42);
    EXPECT_EQ(h.size(), 2u);
}

TEST(IndexedHeap, Clear) {
    indexed_heap h;
    h.insert(1.0, 0);
    h.insert(2.0, 1);
    h.clear();
    EXPECT_TRUE(h.empty());
}
