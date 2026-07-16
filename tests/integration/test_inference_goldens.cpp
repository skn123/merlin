// End-to-end integration tests: drive the Merlin facade on the committed
// cancer.uai fixtures and compare against the golden outputs committed at the
// repository root (cancer.uai.MAR, cancer.uai.EM).
//
// Only the exact CTE algorithm is used, so results are deterministic.

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "merlin.h"
#include "base.h"
#include "graphical_model.h"

namespace {

std::string data_path(const char* name) {
    return std::string(MERLIN_DATA_DIR) + "/" + name;
}
std::string fixtures_path(const char* name) {
    return std::string(MERLIN_FIXTURES_DIR) + "/" + name;
}
std::string tmp_path(const char* name) {
    return std::string(MERLIN_TEST_TMP_DIR) + "/" + name;
}

// Read the whole file as a string.
std::string slurp(const std::string& path) {
    std::ifstream is(path.c_str());
    std::ostringstream ss;
    ss << is.rdbuf();
    return ss.str();
}

}  // namespace

// ---- MAR / PR against cancer.uai.MAR ----------------------------------------

TEST(Integration, CancerMARMatchesGolden) {
    const std::string out_base = tmp_path("cancer_mar_out");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAR);
    eng.set_algorithm(MERLIN_ALGO_CTE);  // exact inference
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    // Merlin appends ".MAR" to the output file for the MAR task.
    std::string produced = slurp(out_base + ".MAR");
    ASSERT_FALSE(produced.empty()) << "no MAR output produced";

    // Parse: token "PR" then the log-value, ... token "MAR" then
    // <nvar> then per-variable <states> <p0..p_{states-1}>.
    std::istringstream is(produced);
    std::string tok;
    double logZ = 0.0;
    bool got_pr = false;
    std::vector<double> marginals;
    size_t nvar = 0;

    while (is >> tok) {
        if (tok == "PR") {
            is >> logZ;
            got_pr = true;
        } else if (tok == "MAR") {
            is >> nvar;
            for (size_t v = 0; v < nvar; ++v) {
                size_t states = 0;
                is >> states;
                for (size_t k = 0; k < states; ++k) {
                    double p = 0.0;
                    is >> p;
                    marginals.push_back(p);
                }
            }
        }
    }

    ASSERT_TRUE(got_pr);
    ASSERT_EQ(nvar, 5u);
    ASSERT_EQ(marginals.size(), 10u);  // 5 binary variables

    // Golden values from cancer.uai.MAR.
    EXPECT_NEAR(logZ, -1.139434, 1e-4);
    const double expected[10] = {
        0.5,   0.5,     // var 0
        1.0,   0.0,     // var 1 (evidence)
        0.125, 0.875,   // var 2
        0.8,   0.2,     // var 3
        0.625, 0.375    // var 4
    };
    for (size_t i = 0; i < 10; ++i)
        EXPECT_NEAR(marginals[i], expected[i], 1e-4) << "marginal index " << i;
}

// ---- EM regression against a pinned snapshot -------------------------------
//
// NOTE: the golden committed at the repo root, cancer.uai.EM, has every CPT
// entry at 0.5. The current EM implementation does NOT reproduce it -- it
// learns real parameters from cancer.dat. That root file is a stale artifact
// predating the EM fixes in the git history ("Fixed a bug in EM virtual
// evidence"), so it is not a valid regression target.
//
// Instead we pin against tests/fixtures/cancer.expected.EM, a snapshot of the
// current (correct) EM output. This locks in behavior: if EM changes, this test
// flags it and the snapshot can be re-reviewed and updated deliberately.
TEST(Integration, CancerEMMatchesPinnedSnapshot) {
    const std::string out_base = tmp_path("cancer_em_out");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_dataset_file(data_path("cancer.dat"));
    eng.set_task(MERLIN_TASK_EM);
    eng.set_algorithm(MERLIN_ALGO_CTE);
    eng.set_iterations(10);
    eng.set_threshold(1e-6);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced_path = out_base + ".EM";
    std::ifstream produced_is(produced_path.c_str());
    ASSERT_TRUE(produced_is.good()) << "no EM output at " << produced_path;
    merlin::graphical_model learned;
    learned.read(produced_is);

    std::ifstream expected_is(fixtures_path("cancer.expected.EM").c_str());
    ASSERT_TRUE(expected_is.good());
    merlin::graphical_model expected;
    expected.read(expected_is);

    EXPECT_EQ(learned.nvar(), expected.nvar());
    ASSERT_EQ(learned.num_factors(), expected.num_factors());

    // Same model structure, so factors line up by index. Compare cell-for-cell.
    for (size_t f = 0; f < learned.num_factors(); ++f) {
        const merlin::factor& lf = learned.get_factor(f);
        const merlin::factor& ef = expected.get_factor(f);
        ASSERT_EQ(lf.num_states(), ef.num_states());
        // Tolerance is loose enough (5e-3) to absorb floating-point drift from
        // different optimization levels across the 10 EM iterations, while
        // still catching any behavioral change in the learning.
        for (size_t k = 0; k < lf.num_states(); ++k)
            EXPECT_NEAR(lf[k], ef[k], 5e-3) << "factor " << f << " cell " << k;
    }

    // Sanity: EM actually learned (did not leave everything at the trivial 0.5).
    bool learned_nontrivial = false;
    for (size_t f = 0; f < learned.num_factors() && !learned_nontrivial; ++f) {
        const merlin::factor& lf = learned.get_factor(f);
        for (size_t k = 0; k < lf.num_states(); ++k)
            if (std::fabs(lf[k] - 0.5) > 1e-3) { learned_nontrivial = true; break; }
    }
    EXPECT_TRUE(learned_nontrivial);
}

// ---- Regression: CTE MAR on a model with more variables than factors --------
//
// paskin.uai has 6 variables but only 5 factors. CTE previously sized its
// elimination graph by num_nodes() (= factor count = 5), so the variable with
// label 5 indexed one past the end of the adjacency table -- a heap overflow
// that crashed non-deterministically (SIGSEGV / std::length_error). This guards
// against a regression: the run must complete cleanly and yield 6 marginals.
TEST(Integration, PaskinCTEMARDoesNotCrash) {
    const std::string out_base = tmp_path("paskin_mar_out");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("paskin.uai"));
    eng.set_task(MERLIN_TASK_MAR);
    eng.set_algorithm(MERLIN_ALGO_CTE);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAR");
    ASSERT_FALSE(produced.empty()) << "no MAR output produced";

    std::istringstream is(produced);
    std::string tok;
    size_t nvar = 0;
    std::vector<double> marginals;
    while (is >> tok) {
        if (tok == "MAR") {
            is >> nvar;
            for (size_t v = 0; v < nvar; ++v) {
                size_t states = 0;
                is >> states;
                for (size_t k = 0; k < states; ++k) {
                    double p = 0.0;
                    is >> p;
                    marginals.push_back(p);
                }
            }
        }
    }

    EXPECT_EQ(nvar, 6u);
    ASSERT_EQ(marginals.size(), 12u);  // 6 binary variables
    // Every marginal is a valid probability, and each pair sums to 1.
    for (size_t v = 0; v < 6; ++v) {
        double p0 = marginals[2 * v], p1 = marginals[2 * v + 1];
        EXPECT_GE(p0, 0.0); EXPECT_LE(p0, 1.0);
        EXPECT_GE(p1, 0.0); EXPECT_LE(p1, 1.0);
        EXPECT_NEAR(p0 + p1, 1.0, 1e-6) << "variable " << v;
    }
}

// ---- AOBB (exact AND/OR branch-and-bound) MAP / MMAP ------------------------
//
// AOBB is an exact optimizer, so its reported value must equal the true optimum
// (verified here against values computed by brute force over the joint):
//   cancer.uai MAP  (evidence var1=0): log value -2.617844, config 1 0 1 0 0
//   simple5.uai MMAP (query 0 1 2)   : log value 11.285626, config 1 1 0

TEST(Integration, CancerMAPAobbIsExact) {
    const std::string out_base = tmp_path("cancer_map_aobb");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_AOBB);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP");
    ASSERT_FALSE(produced.empty()) << "no MAP output produced";

    // Parse: token "MAP", then <count> followed by one value per variable.
    std::istringstream is(produced);
    std::string tok;
    size_t count = 0;
    std::vector<size_t> cfg;
    while (is >> tok) {
        if (tok == "MAP") {
            is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); }
        }
    }
    ASSERT_EQ(count, 5u);
    ASSERT_EQ(cfg.size(), 5u);
    const size_t expected[5] = {1, 0, 1, 0, 0};  // brute-force MAP config
    for (size_t i = 0; i < 5; ++i)
        EXPECT_EQ(cfg[i], expected[i]) << "variable " << i;
}

TEST(Integration, Simple5MMAPAobbIsExact) {
    const std::string out_base = tmp_path("simple5_mmap_aobb");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("simple5.uai"));
    eng.set_query_file(data_path("simple5.map"));
    eng.set_task(MERLIN_TASK_MMAP);
    eng.set_algorithm(MERLIN_ALGO_AOBB);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MMAP");
    ASSERT_FALSE(produced.empty()) << "no MMAP output produced";

    std::istringstream is(produced);
    std::string tok;
    size_t count = 0;
    std::vector<size_t> cfg;
    while (is >> tok) {
        if (tok == "MMAP") {
            is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); }
        }
    }
    ASSERT_EQ(count, 3u);
    ASSERT_EQ(cfg.size(), 3u);
    const size_t expected[3] = {1, 1, 0};  // brute-force MMAP query config
    for (size_t i = 0; i < 3; ++i)
        EXPECT_EQ(cfg[i], expected[i]) << "query variable " << i;
}

// A generous time limit does not change the result: the search still completes
// and reports the proven optimum with an "optimal" status in the JSON output.
TEST(Integration, CancerMAPAobbTimeLimitStillOptimal) {
    const std::string out_base = tmp_path("cancer_map_aobb_tl");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_AOBB);
    eng.set_time_limit(60.0);                 // ample: search completes well within
    eng.set_output_format(MERLIN_OUTPUT_JSON);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP.json");
    ASSERT_FALSE(produced.empty()) << "no MAP JSON output produced";
    // Completed search => proven optimal.
    EXPECT_NE(produced.find("\"optimal\" : true"), std::string::npos)
        << "expected optimal:true in: " << produced;
    EXPECT_NE(produced.find("\"status\" : \"true\""), std::string::npos);
    // The optimum value is unchanged by the (unreached) time limit.
    EXPECT_NE(produced.find("\"value\" : -2.617844"), std::string::npos)
        << produced;
}

// ---- BRAOBB (breadth-rotating AND/OR BnB) MAP / MMAP -----------------------
//
// BRAOBB explores the same search space as AOBB and is exact, so it must return
// the same optima (verified against brute force in the AOBB tests): cancer MAP
// -> 1 0 1 0 0 and simple5 MMAP (query 0 1 2) -> 1 1 0. The result must also be
// invariant to the rotation limit (only exploration order / anytime quality
// changes).

TEST(Integration, CancerMAPBraobbIsExact) {
    const std::string out_base = tmp_path("cancer_map_braobb");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_BRAOBB);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP");
    ASSERT_FALSE(produced.empty()) << "no MAP output produced";
    std::istringstream is(produced);
    std::string tok; size_t count = 0; std::vector<size_t> cfg;
    while (is >> tok)
        if (tok == "MAP") { is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); } }
    ASSERT_EQ(count, 5u);
    const size_t expected[5] = {1, 0, 1, 0, 0};
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(cfg[i], expected[i]) << "variable " << i;
}

TEST(Integration, Simple5MMAPBraobbIsExact) {
    const std::string out_base = tmp_path("simple5_mmap_braobb");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("simple5.uai"));
    eng.set_query_file(data_path("simple5.map"));
    eng.set_task(MERLIN_TASK_MMAP);
    eng.set_algorithm(MERLIN_ALGO_BRAOBB);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MMAP");
    ASSERT_FALSE(produced.empty()) << "no MMAP output produced";
    std::istringstream is(produced);
    std::string tok; size_t count = 0; std::vector<size_t> cfg;
    while (is >> tok)
        if (tok == "MMAP") { is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); } }
    ASSERT_EQ(count, 3u);
    const size_t expected[3] = {1, 1, 0};
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(cfg[i], expected[i]) << "query variable " << i;
}

// The reported optimum must not depend on the rotation limit.
TEST(Integration, Simple5MMAPBraobbRotationInvariant) {
    const char* out1 = "simple5_mmap_br_r1";
    const char* out2 = "simple5_mmap_br_r1000";

    double vals[2];
    const size_t rot[2] = {1, 1000};
    for (int k = 0; k < 2; ++k) {
        Merlin eng;
        eng.set_use_files(true);
        eng.set_model_file(data_path("simple5.uai"));
        eng.set_query_file(data_path("simple5.map"));
        eng.set_task(MERLIN_TASK_MMAP);
        eng.set_algorithm(MERLIN_ALGO_BRAOBB);
        eng.set_rotate_limit(rot[k]);
        eng.set_output_format(MERLIN_OUTPUT_JSON);
        eng.set_output_file(tmp_path(k == 0 ? out1 : out2));
        ASSERT_TRUE(eng.init());
        ASSERT_EQ(eng.run(), 0);
        std::string produced = slurp(tmp_path(k == 0 ? out1 : out2) + ".MMAP.json");
        // Parse the "value" field.
        size_t p = produced.find("\"value\" : ");
        ASSERT_NE(p, std::string::npos);
        vals[k] = std::atof(produced.c_str() + p + 10);
    }
    EXPECT_NEAR(vals[0], vals[1], 1e-6);
    EXPECT_NEAR(vals[0], 11.285626, 1e-4);
}

// ---- AOBF (best-first AND/OR search / AO*) MAP / MMAP ----------------------
//
// AOBF searches the context-minimal AND/OR graph guided by the WMB heuristic and
// is exact, so it must return the same optima as AOBB / brute force: cancer MAP
// -> 1 0 1 0 0, simple5 MMAP (query 0 1 2) -> 1 1 0. Crucially, correctness must
// hold even when the WMB heuristic is APPROXIMATE (i-bound < treewidth): AOBF
// reconstructs an absolute per-node completion bound from the normalized WMB
// beliefs, so a small i-bound must not make it return a suboptimal answer.

TEST(Integration, CancerMAPAobfIsExact) {
    const std::string out_base = tmp_path("cancer_map_aobf");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_AOBF);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP");
    ASSERT_FALSE(produced.empty()) << "no MAP output produced";
    std::istringstream is(produced);
    std::string tok; size_t count = 0; std::vector<size_t> cfg;
    while (is >> tok)
        if (tok == "MAP") { is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); } }
    ASSERT_EQ(count, 5u);
    const size_t expected[5] = {1, 0, 1, 0, 0};
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(cfg[i], expected[i]) << "variable " << i;
}

TEST(Integration, Simple5MMAPAobfIsExact) {
    const std::string out_base = tmp_path("simple5_mmap_aobf");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("simple5.uai"));
    eng.set_query_file(data_path("simple5.map"));
    eng.set_task(MERLIN_TASK_MMAP);
    eng.set_algorithm(MERLIN_ALGO_AOBF);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MMAP");
    ASSERT_FALSE(produced.empty()) << "no MMAP output produced";
    std::istringstream is(produced);
    std::string tok; size_t count = 0; std::vector<size_t> cfg;
    while (is >> tok)
        if (tok == "MMAP") { is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); } }
    ASSERT_EQ(count, 3u);
    const size_t expected[3] = {1, 1, 0};
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(cfg[i], expected[i]) << "query variable " << i;
}

// Regression: simple5 has treewidth 3, so i-bound 2 gives an APPROXIMATE WMB
// heuristic. AOBF must still report the exact optimum (MAP value 10.982467),
// exercising the absolute-bound reconstruction. (A naive port that used the raw
// normalized WMB heuristic returned a suboptimal 7.18 here.)
TEST(Integration, Simple5MAPAobfApproxHeuristicIsExact) {
    const std::string out_base = tmp_path("simple5_map_aobf_ib2");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("simple5.uai"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_AOBF);
    eng.set_ibound(2);                        // < treewidth => approximate heuristic
    eng.set_output_format(MERLIN_OUTPUT_JSON);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP.json");
    ASSERT_FALSE(produced.empty()) << "no MAP JSON output produced";
    EXPECT_NE(produced.find("\"optimal\" : true"), std::string::npos) << produced;
    size_t p = produced.find("\"value\" : ");
    ASSERT_NE(p, std::string::npos);
    double value = std::atof(produced.c_str() + p + 10);
    EXPECT_NEAR(value, 10.982467, 1e-4) << produced;
}

// ---- RBFAOO (recursive best-first AND/OR search with overestimation) --------
//
// RBFAOO is a DFPN-style recursive best-first search over the context-minimal
// AND/OR graph with a memory-bounded transposition table. It is exact, so it
// must return the same optima as AOBB / AOBF / brute force: cancer MAP
// -> 1 0 1 0 0, simple5 MMAP (query 0 1 2) -> 1 1 0. Like the other best-first
// solver it reconstructs the optimum from the cache and needs the absolute-bound
// WMB heuristic to stay optimal even at an approximate i-bound. The MMAP test
// exercises the inline logsumexp SUM path.

TEST(Integration, CancerMAPRbfaooIsExact) {
    const std::string out_base = tmp_path("cancer_map_rbfaoo");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("cancer.uai"));
    eng.set_evidence_file(data_path("cancer.evid"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_RBFAOO);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP");
    ASSERT_FALSE(produced.empty()) << "no MAP output produced";
    std::istringstream is(produced);
    std::string tok; size_t count = 0; std::vector<size_t> cfg;
    while (is >> tok)
        if (tok == "MAP") { is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); } }
    ASSERT_EQ(count, 5u);
    const size_t expected[5] = {1, 0, 1, 0, 0};
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(cfg[i], expected[i]) << "variable " << i;
}

TEST(Integration, Simple5MMAPRbfaooIsExact) {
    const std::string out_base = tmp_path("simple5_mmap_rbfaoo");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("simple5.uai"));
    eng.set_query_file(data_path("simple5.map"));
    eng.set_task(MERLIN_TASK_MMAP);
    eng.set_algorithm(MERLIN_ALGO_RBFAOO);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MMAP");
    ASSERT_FALSE(produced.empty()) << "no MMAP output produced";
    std::istringstream is(produced);
    std::string tok; size_t count = 0; std::vector<size_t> cfg;
    while (is >> tok)
        if (tok == "MMAP") { is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); } }
    ASSERT_EQ(count, 3u);
    const size_t expected[3] = {1, 1, 0};
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(cfg[i], expected[i]) << "query variable " << i;
}

// Regression: simple5 has treewidth 3, so i-bound 2 gives an APPROXIMATE WMB
// heuristic. RBFAOO (best-first) must still report the exact optimum
// (MAP value 10.982467), exercising the absolute-bound reconstruction.
TEST(Integration, Simple5MAPRbfaooApproxHeuristicIsExact) {
    const std::string out_base = tmp_path("simple5_map_rbfaoo_ib2");

    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("simple5.uai"));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(MERLIN_ALGO_RBFAOO);
    eng.set_ibound(2);                        // < treewidth => approximate heuristic
    eng.set_output_format(MERLIN_OUTPUT_JSON);
    eng.set_output_file(out_base);

    ASSERT_TRUE(eng.init());
    ASSERT_EQ(eng.run(), 0);

    std::string produced = slurp(out_base + ".MAP.json");
    ASSERT_FALSE(produced.empty()) << "no MAP JSON output produced";
    EXPECT_NE(produced.find("\"optimal\" : true"), std::string::npos) << produced;
    size_t p = produced.find("\"value\" : ");
    ASSERT_NE(p, std::string::npos);
    double value = std::atof(produced.c_str() + p + 10);
    EXPECT_NEAR(value, 10.982467, 1e-4) << produced;
}

// ---- Zero-probability robustness (negative-log cost space) ------------------
//
// zeros.uai has hard-zero factor entries (deterministic constraints): only two
// of its 16 configurations have positive probability, so most search paths are
// +inf-cost dead ends. In negative-log cost space, mishandled +inf can produce
// NaN (from inf-inf in logsumexp / threshold arithmetic) or hangs/crashes (dead
// nodes never terminating). All four AND/OR solvers must return the exact
// optimum without NaN, crash, or timeout. Brute-forced optima:
//   MAP  -> log -0.867501, config [1,0,0,1]
//   MMAP (query x0) -> log -0.867501, config [1]

namespace {
// Run a search algorithm on zeros.uai and return the JSON value; assert no NaN.
double run_zeros(int task, int algorithm, const char* suffix, const char* tag) {
    const std::string out_base = tmp_path((std::string("zeros_") + tag).c_str());
    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path("zeros.uai"));
    if (task == MERLIN_TASK_MMAP) eng.set_query_file(data_path("zeros.map"));
    eng.set_task(task);
    eng.set_algorithm(algorithm);
    eng.set_ibound(2);
    eng.set_output_format(MERLIN_OUTPUT_JSON);
    eng.set_output_file(out_base);
    EXPECT_TRUE(eng.init());
    EXPECT_EQ(eng.run(), 0);
    std::string produced = slurp(out_base + suffix);
    EXPECT_FALSE(produced.empty());
    EXPECT_EQ(produced.find("nan"), std::string::npos) << "NaN in output: " << produced;
    size_t p = produced.find("\"value\" : ");
    EXPECT_NE(p, std::string::npos);
    return std::atof(produced.c_str() + p + 10);
}
}  // namespace

TEST(Integration, ZerosMAPAllSearchAlgosExact) {
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MAP, MERLIN_ALGO_AOBB,   ".MAP.json", "map_aobb"),   -0.867501, 1e-4);
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MAP, MERLIN_ALGO_BRAOBB, ".MAP.json", "map_braobb"), -0.867501, 1e-4);
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MAP, MERLIN_ALGO_AOBF,   ".MAP.json", "map_aobf"),   -0.867501, 1e-4);
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MAP, MERLIN_ALGO_RBFAOO, ".MAP.json", "map_rbfaoo"), -0.867501, 1e-4);
}

TEST(Integration, ZerosMMAPAllSearchAlgosExact) {
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MMAP, MERLIN_ALGO_AOBB,   ".MMAP.json", "mmap_aobb"),   -0.867501, 1e-4);
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MMAP, MERLIN_ALGO_BRAOBB, ".MMAP.json", "mmap_braobb"), -0.867501, 1e-4);
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MMAP, MERLIN_ALGO_AOBF,   ".MMAP.json", "mmap_aobf"),   -0.867501, 1e-4);
    EXPECT_NEAR(run_zeros(MERLIN_TASK_MMAP, MERLIN_ALGO_RBFAOO, ".MMAP.json", "mmap_rbfaoo"), -0.867501, 1e-4);
}

// ---- SLS (G+StS) and GLS+ local search for MAP ------------------------------
//
// SLS and GLS+ are stochastic (never prove optimality), but on these small/medium
// models they reliably reach the MAP optimum within a short time budget. Runs are
// deterministic given a fixed seed, so the tests assert the exact optimum:
//   cancer MAP (evidence) -> -2.617844, config [1,0,1,0,0]
//   simple5 MAP           -> 10.982467
//   zeros.uai MAP         -> -0.867501 (zero-probability robustness, no NaN)

namespace {
// Run a local-search algorithm on a model (with optional evidence) for a fixed
// seed + time budget; return the JSON value and assert no NaN.
double run_local_search(const char* model, const char* evid, int algorithm,
        size_t seed, double time_limit, const char* tag) {
    const std::string out_base = tmp_path((std::string("ls_") + tag).c_str());
    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path(model));
    if (evid) eng.set_evidence_file(data_path(evid));
    eng.set_task(MERLIN_TASK_MAP);
    eng.set_algorithm(algorithm);
    eng.set_seed(seed);
    eng.set_time_limit(time_limit);
    eng.set_output_format(MERLIN_OUTPUT_JSON);
    eng.set_output_file(out_base);
    EXPECT_TRUE(eng.init());
    EXPECT_EQ(eng.run(), 0);
    std::string produced = slurp(out_base + ".MAP.json");
    EXPECT_FALSE(produced.empty());
    EXPECT_EQ(produced.find("nan"), std::string::npos) << "NaN in output: " << produced;
    size_t p = produced.find("\"value\" : ");
    EXPECT_NE(p, std::string::npos);
    return std::atof(produced.c_str() + p + 10);
}
}  // namespace

TEST(Integration, CancerMAPSlsFindsOptimum) {
    EXPECT_NEAR(run_local_search("cancer.uai", "cancer.evid", MERLIN_ALGO_SLS,
            12345678, 2.0, "cancer_sls"), -2.617844, 1e-4);
}
TEST(Integration, CancerMAPGlsFindsOptimum) {
    EXPECT_NEAR(run_local_search("cancer.uai", "cancer.evid", MERLIN_ALGO_GLS,
            12345678, 2.0, "cancer_gls"), -2.617844, 1e-4);
}
TEST(Integration, Simple5MAPSlsFindsOptimum) {
    EXPECT_NEAR(run_local_search("simple5.uai", NULL, MERLIN_ALGO_SLS,
            12345678, 2.0, "simple5_sls"), 10.982467, 1e-4);
}
TEST(Integration, Simple5MAPGlsFindsOptimum) {
    EXPECT_NEAR(run_local_search("simple5.uai", NULL, MERLIN_ALGO_GLS,
            12345678, 2.0, "simple5_gls"), 10.982467, 1e-4);
}
// Zero-probability robustness: local search must handle log(0) entries (LOG_ZERO
// clamp) and still find the feasible optimum, with no NaN in the output.
TEST(Integration, ZerosMAPLocalSearchRobust) {
    EXPECT_NEAR(run_local_search("zeros.uai", NULL, MERLIN_ALGO_SLS,
            12345678, 2.0, "zeros_sls"), -0.867501, 1e-4);
    EXPECT_NEAR(run_local_search("zeros.uai", NULL, MERLIN_ALGO_GLS,
            12345678, 2.0, "zeros_gls"), -0.867501, 1e-4);
}

// ---- SLS / GLS+ for Marginal MAP (MMAP) -------------------------------------
//
// For MMAP the local search is over the MAP (query) variables only; a complete
// MAP assignment is scored by the probability of evidence over the SUM variables,
// estimated fast by the precompiled weighted mini-bucket heuristic (an upper
// bound). The reported value is therefore a bound, but the argmax MAP assignment
// is what matters: on these small models it reliably equals the exact MMAP
// optimum found by the AND/OR solvers:
//   simple5 MMAP (query 0 1 2) -> config [1,1,0]
//   zeros.uai MMAP (query x0)  -> config [1]  (zero-probability robustness)

namespace {
// Run a local-search MMAP query and return the query-variable configuration from
// the UAI output; assert no crash / empty output.
std::vector<size_t> run_mmap_local_search(const char* model, const char* query,
        int algorithm, size_t seed, double time_limit, size_t ibound,
        const char* tag) {
    const std::string out_base = tmp_path((std::string("mmap_ls_") + tag).c_str());
    Merlin eng;
    eng.set_use_files(true);
    eng.set_model_file(data_path(model));
    eng.set_query_file(data_path(query));
    eng.set_task(MERLIN_TASK_MMAP);
    eng.set_algorithm(algorithm);
    eng.set_seed(seed);
    eng.set_time_limit(time_limit);
    eng.set_ibound(ibound);
    eng.set_output_format(MERLIN_OUTPUT_UAI);
    eng.set_output_file(out_base);
    EXPECT_TRUE(eng.init());
    EXPECT_EQ(eng.run(), 0);
    std::string produced = slurp(out_base + ".MMAP");
    EXPECT_FALSE(produced.empty());
    std::istringstream is(produced);
    std::string tok; size_t count = 0; std::vector<size_t> cfg;
    while (is >> tok)
        if (tok == "MMAP") { is >> count;
            for (size_t i = 0; i < count; ++i) { size_t v; is >> v; cfg.push_back(v); } }
    return cfg;
}
}  // namespace

TEST(Integration, Simple5MMAPSlsFindsOptimum) {
    std::vector<size_t> cfg = run_mmap_local_search("simple5.uai", "simple5.map",
            MERLIN_ALGO_SLS, 12345678, 2.0, 10, "simple5_sls");
    ASSERT_EQ(cfg.size(), 3u);
    const size_t expected[3] = {1, 1, 0};
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(cfg[i], expected[i]) << "query var " << i;
}
TEST(Integration, Simple5MMAPGlsFindsOptimum) {
    std::vector<size_t> cfg = run_mmap_local_search("simple5.uai", "simple5.map",
            MERLIN_ALGO_GLS, 12345678, 2.0, 10, "simple5_gls");
    ASSERT_EQ(cfg.size(), 3u);
    const size_t expected[3] = {1, 1, 0};
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(cfg[i], expected[i]) << "query var " << i;
}
// Zero-probability SUM part: the WMB estimator + LOG_ZERO clamp must not crash or
// produce NaN, and the search still finds the feasible MMAP optimum (x0 = 1).
TEST(Integration, ZerosMMAPLocalSearchRobust) {
    for (int alg : {MERLIN_ALGO_SLS, MERLIN_ALGO_GLS}) {
        std::vector<size_t> cfg = run_mmap_local_search("zeros.uai", "zeros.map",
                alg, 12345678, 2.0, 10, alg == MERLIN_ALGO_SLS ? "zeros_sls" : "zeros_gls");
        ASSERT_EQ(cfg.size(), 1u);
        EXPECT_EQ(cfg[0], 1u);
    }
}
