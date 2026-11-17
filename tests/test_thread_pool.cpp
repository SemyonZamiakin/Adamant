// ThreadPool: every task runs exactly once per job, across many back-to-back
// jobs, after idle periods (workers asleep) and with odd task counts.
// SPDX-License-Identifier: MIT
#include "adamant/thread_pool.hpp"
#include "check.hpp"

#include <chrono>
#include <thread>

using namespace adamant;

int main() {
    for (int nt : {1, 2, 3, 8, 16}) {
        ThreadPool pool(nt);
        CHECK(pool.size() == nt);
        std::vector<std::atomic<int>> hits(1000);
        long long expect = 0;
        for (int job = 0; job < 3000; ++job) {
            const int n = 1 + (job * 37) % 1000;
            pool.run(n, [&](int t) { hits[static_cast<std::size_t>(t)].fetch_add(1, std::memory_order_relaxed); });
            expect += n;
            if (job % 1000 == 999) std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let workers fall asleep
        }
        long long total = 0;
        for (auto& h : hits) total += h.load();
        CHECK_MSG(total == expect, "threads=%d: %lld task runs, expected %lld", nt, total, expect);
        // Results written by tasks are visible to the caller after run() returns.
        std::vector<int> out(4096, 0);
        for (int rep = 0; rep < 200; ++rep) {
            pool.run(64, [&](int t) {
                for (int i = 0; i < 64; ++i) out[static_cast<std::size_t>(t * 64 + i)] = rep;
            });
            bool ok = true;
            for (int v : out) ok &= v == rep;
            CHECK(ok);
        }
    }
    CHECK(default_thread_count() >= 1);
    std::printf("  default threads (physical cores): %d\n", default_thread_count());
    return test_result("test_thread_pool");
}
