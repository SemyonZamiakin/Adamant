// Adamant SIMD-VIC by Euclase: a small fork-join pool for matrix-vector work.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace adamant {

// Runs fn(task) for task in [0, n_tasks) on all threads, the caller included,
// and returns when every task is done. Tasks are handed out through one atomic
// counter, so fast threads take more of them. Workers spin briefly between
// jobs (a token needs ~150 jobs, a few microseconds apart) and fall back to a
// condition variable when idle, so an idle pool costs no CPU.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return static_cast<int>(workers_.size()) + 1; }

    template <class F>
    void run(int n_tasks, F&& fn) {
        if (n_tasks <= 0) return;
        if (workers_.empty() || n_tasks == 1) {
            for (int t = 0; t < n_tasks; ++t) fn(t);
            return;
        }
        Job job{&invoke<F>, &fn, n_tasks};
        dispatch(job);
    }

private:
    struct Job {
        void (*call)(void* ctx, int task);
        void* ctx;
        int n_tasks;
    };
    template <class F>
    static void invoke(void* ctx, int task) {
        (*static_cast<F*>(ctx))(task);
    }

    void dispatch(const Job& job);
    void worker_loop();
    void work(const Job& job);

    std::vector<std::thread> workers_;
    alignas(64) std::atomic<std::uint64_t> epoch_{0};
    alignas(64) std::atomic<int> next_task_{0};
    alignas(64) std::atomic<int> done_{0};  // workers finished with the current job
    alignas(64) std::atomic<int> sleepers_{0};
    Job job_{};
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

int default_thread_count();  // physical cores when known, else hardware_concurrency

}  // namespace adamant
