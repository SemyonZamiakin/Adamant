// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/thread_pool.hpp"

#include <fstream>
#include <set>
#include <string>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define ADAMANT_PAUSE() _mm_pause()
#else
#define ADAMANT_PAUSE() std::this_thread::yield()
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace adamant {

namespace {
constexpr int kSpinIterations = 1 << 16;  // ~0.3-1 ms of spinning before sleeping
}

ThreadPool::ThreadPool(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    workers_.reserve(static_cast<std::size_t>(n_threads - 1));
    for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_.store(true, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::work(const Job& job) {
    for (;;) {
        const int t = next_task_.fetch_add(1, std::memory_order_relaxed);
        if (t >= job.n_tasks) break;
        job.call(job.ctx, t);
    }
}

void ThreadPool::dispatch(const Job& job) {
    job_ = job;
    next_task_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_seq_cst);  // publishes job_; seq_cst pairs with sleepers_
    if (sleepers_.load(std::memory_order_seq_cst) > 0) {
        std::lock_guard<std::mutex> lk(mu_);
        cv_.notify_all();
    }
    work(job);
    // Barrier: every worker checks in once per job, after it has stopped
    // reading job_. Only then may the next dispatch overwrite job_.
    const int n = static_cast<int>(workers_.size());
    while (done_.load(std::memory_order_acquire) < n) ADAMANT_PAUSE();
}

void ThreadPool::worker_loop() {
    std::uint64_t seen = 0;
    for (;;) {
        std::uint64_t e = epoch_.load(std::memory_order_acquire);
        for (int i = 0; e == seen && i < kSpinIterations; ++i) {
            ADAMANT_PAUSE();
            e = epoch_.load(std::memory_order_acquire);
        }
        if (e == seen) {
            std::unique_lock<std::mutex> lk(mu_);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            cv_.wait(lk, [&] { return epoch_.load(std::memory_order_acquire) != seen; });
            sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            e = epoch_.load(std::memory_order_acquire);
        }
        seen = e;
        if (stop_.load(std::memory_order_relaxed)) return;
        work(job_);
        done_.fetch_add(1, std::memory_order_release);
    }
}

int default_thread_count() {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::string buf(len, '\0');
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
            int cores = 0;
            for (DWORD off = 0; off < len;) {
                auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
                if (p->Relationship == RelationProcessorCore) ++cores;
                off += p->Size;
            }
            if (cores > 0) return cores;
        }
    }
#else
    std::ifstream f("/proc/cpuinfo");
    std::set<std::pair<int, int>> cores;
    std::string line;
    int phys = 0;
    while (std::getline(f, line)) {
        if (line.rfind("physical id", 0) == 0) phys = std::stoi(line.substr(line.find(':') + 1));
        if (line.rfind("core id", 0) == 0) cores.emplace(phys, std::stoi(line.substr(line.find(':') + 1)));
    }
    if (!cores.empty()) return static_cast<int>(cores.size());
#endif
    const unsigned n = std::thread::hardware_concurrency();
    return n ? static_cast<int>(n) : 1;
}

}  // namespace adamant
