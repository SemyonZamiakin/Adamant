// adamant-view: live console dashboard for Adamant SIMD-VIC by Euclase.
//
// Reads two Corundum HFT rings, both zero-copy from shared memory:
//   /adamant-tokens   tokens and timing published by `adamant run --stream`
//   /euclase-monitor  EMS1 metric snapshots published by Euclase Monitor
//                     (per-core CPU load, RAM). Without Monitor, the viewer
//                     samples CPU and RAM itself.
// and draws tokens/s, weight bandwidth, per-core load bars, RAM and the text.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/token_wire.h"
#include "corundum.h"
#include "euclase/metrics_wire.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double secs_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// ---------------------------------------------------------------------------
// system metrics: from Euclase Monitor, or sampled locally
// ---------------------------------------------------------------------------

struct SysMetrics {
    std::vector<double> core_load;  // %
    double cpu_load = NAN, ram_used_gb = NAN, ram_total_gb = NAN;
    std::string cpu_name;
    std::string source = "none";
};

class LocalSampler {
public:
    bool sample(SysMetrics& m) {
        std::vector<std::pair<unsigned long long, unsigned long long>> now;  // (busy, total)
        if (!read_cores(now)) return false;
        if (prev_.size() == now.size()) {
            m.core_load.assign(now.size(), 0.0);
            double busy = 0, total = 0;
            for (std::size_t i = 0; i < now.size(); ++i) {
                const double db = static_cast<double>(now[i].first - prev_[i].first);
                const double dt = static_cast<double>(now[i].second - prev_[i].second);
                m.core_load[i] = dt > 0 ? std::clamp(100.0 * db / dt, 0.0, 100.0) : 0.0;
                busy += db;
                total += dt;
            }
            m.cpu_load = total > 0 ? 100.0 * busy / total : 0.0;
        }
        prev_ = now;
        read_ram(m);
        m.source = "local sampler";
        return true;
    }

private:
#ifdef _WIN32
    struct ProcPerf {
        LARGE_INTEGER idle, kernel, user, dpc, interrupt;
        ULONG interrupts;
    };
    bool read_cores(std::vector<std::pair<unsigned long long, unsigned long long>>& out) {
        using Fn = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
        static Fn fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation")));
        if (!fn) return false;
        std::vector<ProcPerf> buf(256);
        ULONG len = 0;
        if (fn(8 /* SystemProcessorPerformanceInformation */, buf.data(), static_cast<ULONG>(buf.size() * sizeof(ProcPerf)), &len) != 0)
            return false;
        const std::size_t n = len / sizeof(ProcPerf);
        for (std::size_t i = 0; i < n; ++i) {
            const auto total = static_cast<unsigned long long>(buf[i].kernel.QuadPart + buf[i].user.QuadPart);  // kernel includes idle
            out.emplace_back(total - static_cast<unsigned long long>(buf[i].idle.QuadPart), total);
        }
        return n > 0;
    }
    void read_ram(SysMetrics& m) {
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms)) {
            m.ram_total_gb = static_cast<double>(ms.ullTotalPhys) / 1e9;
            m.ram_used_gb = static_cast<double>(ms.ullTotalPhys - ms.ullAvailPhys) / 1e9;
        }
    }
#else
    bool read_cores(std::vector<std::pair<unsigned long long, unsigned long long>>& out) {
        std::ifstream f("/proc/stat");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("cpu", 0) != 0 || line.size() < 4 || line[3] < '0' || line[3] > '9') continue;
            unsigned long long v[8] = {};
            std::sscanf(line.c_str() + line.find(' '), "%llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
            unsigned long long total = 0;
            for (auto x : v) total += x;
            out.emplace_back(total - v[3] - v[4], total);
        }
        return !out.empty();
    }
    void read_ram(SysMetrics& m) {
        std::ifstream f("/proc/meminfo");
        std::string k;
        unsigned long long v;
        std::string unit;
        double total = NAN, avail = NAN;
        while (f >> k >> v >> unit) {
            if (k == "MemTotal:") total = static_cast<double>(v) * 1024;
            if (k == "MemAvailable:") avail = static_cast<double>(v) * 1024;
        }
        m.ram_total_gb = total / 1e9;
        m.ram_used_gb = (total - avail) / 1e9;
    }
#endif
    std::vector<std::pair<unsigned long long, unsigned long long>> prev_;
};

class MonitorReader {
public:
    ~MonitorReader() { close(); }
    // Returns true if a fresh snapshot was applied.
    bool poll(SysMetrics& m, bool takeover) {
        if (!open_) {
            if (secs_since(last_try_) < 1.0 && tried_) return false;
            tried_ = true;
            last_try_ = Clock::now();
            const crd_status s = crd_reader_open(&r_, EMS_RING_NAME, takeover ? CRD_OPEN_TAKEOVER : 0);
            if (s != CRD_OK) {
                status_ = crd_strerror(s);
                return false;
            }
            open_ = true;
        }
        bool fresh = false;
        crd_msg msg;
        crd_status s;
        while ((s = crd_read(&r_, &msg)) == CRD_OK) {
            if (msg.type == EMS_RECORD_SNAPSHOT) fresh |= apply(msg.data, msg.size, m);
        }
        crd_release(&r_);
        if (s == CRD_ECORRUPT || (crd_writer_closed(&r_) && crd_reader_pending(&r_) == 0 && secs_since(last_snapshot_) > 3.0)) {
            close();  // Monitor exited or restarted: reattach later
        }
        if (fresh) last_snapshot_ = Clock::now();
        return fresh;
    }
    bool connected() const { return open_ && secs_since(last_snapshot_) < 3.0; }
    const std::string& status() const { return status_; }

private:
    bool apply(const void* data, std::size_t size, SysMetrics& m) {
        // Copy out first: the payload lives in memory another process writes.
        std::vector<unsigned char> buf(static_cast<const unsigned char*>(data), static_cast<const unsigned char*>(data) + size);
        ems_header h;
        ems_iter it;
        if (!ems_begin(buf.data(), buf.size(), &h, &it)) return false;
        std::map<int, double> cores;
        ems_entry e;
        while (ems_next(&it, &e)) {
            const std::string name(e.name, e.name_len);
            if (e.kind == EMS_KIND_NUMBER) {
                if (name == "cpu.load") m.cpu_load = e.number;
                else if (name == "ram.used_gb") m.ram_used_gb = e.number;
                else if (name == "ram.total_gb") m.ram_total_gb = e.number;
                else if (name.rfind("cpu.core", 0) == 0 && name.size() > 13 && name.compare(name.size() - 5, 5, ".load") == 0)
                    cores[std::atoi(name.c_str() + 8)] = e.number;
            } else if (name == "cpu.name") {
                m.cpu_name.assign(e.str, e.str_len);
            }
        }
        if (!cores.empty()) {
            m.core_load.assign(static_cast<std::size_t>(cores.rbegin()->first + 1), 0.0);
            for (auto& [i, v] : cores) m.core_load[static_cast<std::size_t>(i)] = v;
        }
        m.source = "Euclase Monitor (EMS1 on " EMS_RING_NAME ")";
        return true;
    }
    void close() {
        if (open_) crd_reader_close(&r_);
        open_ = false;
    }
    crd_reader r_{};
    bool open_ = false, tried_ = false;
    Clock::time_point last_try_{}, last_snapshot_{};
    std::string status_ = "not connected";
};

// ---------------------------------------------------------------------------
// token stream
// ---------------------------------------------------------------------------

struct Session {
    bool active = false, ended = false;
    std::string model, isa;
    unsigned threads = 0, n_ctx = 0, n_layer = 0;
    std::uint64_t weight_bytes = 0;
    unsigned n_prompt = 0;
    double prefill_tps = 0;
    std::uint64_t n_tokens = 0, gen_ns = 0, dropped = 0;
    std::string text;
    std::deque<double> tps_hist;      // per-token instantaneous tok/s
    std::deque<std::uint64_t> fwd_ns;  // recent forward times
    double final_tps = 0;
};

// ---------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------

const char* kTeal = "\033[38;2;45;212;191m";
const char* kDim = "\033[38;2;94;130;140m";
const char* kText = "\033[38;2;248;250;252m";
const char* kReset = "\033[0m";

std::string bar(double pct, int width) {
    if (!(pct >= 0)) pct = 0;
    pct = std::min(pct, 100.0);
    const double cells = pct / 100.0 * width;
    static const char* parts[] = {"", "\xE2\x96\x8F", "\xE2\x96\x8E", "\xE2\x96\x8D", "\xE2\x96\x8C", "\xE2\x96\x8B", "\xE2\x96\x8A", "\xE2\x96\x89"};
    std::string s;
    const int full = static_cast<int>(cells);
    for (int i = 0; i < full; ++i) s += "\xE2\x96\x88";
    int used = full;
    const int frac = static_cast<int>((cells - full) * 8);
    if (full < width && frac > 0) {
        s += parts[frac];
        ++used;
    }
    std::string out = s;
    out += kDim;
    for (int i = used; i < width; ++i) out += "\xC2\xB7";
    return out;
}

std::string sparkline(const std::deque<double>& v, std::size_t width) {
    static const char* blocks[] = {"\xE2\x96\x81", "\xE2\x96\x82", "\xE2\x96\x83", "\xE2\x96\x84", "\xE2\x96\x85", "\xE2\x96\x86", "\xE2\x96\x87", "\xE2\x96\x88"};
    if (v.empty()) return "";
    const std::size_t n = std::min(width, v.size());
    double mx = 0;
    for (std::size_t i = v.size() - n; i < v.size(); ++i) mx = std::max(mx, v[i]);
    std::string s;
    for (std::size_t i = v.size() - n; i < v.size(); ++i) s += blocks[mx > 0 ? std::min(7, static_cast<int>(v[i] / mx * 7.999)) : 0];
    return s;
}

int term_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) return info.srWindow.Right - info.srWindow.Left + 1;
#else
    winsize w{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
#endif
    return 100;
}

// Wraps UTF-8 text to width columns (counting code points), returns the last `rows` lines.
std::vector<std::string> wrap_tail(const std::string& text, int width, int rows) {
    std::vector<std::string> lines(1);
    int col = 0;
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        const std::size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if (c == '\n') {
            lines.emplace_back();
            col = 0;
            ++i;
            continue;
        }
        if (col >= width) {
            // break at the last space if there is one
            std::string& cur = lines.back();
            const std::size_t sp = cur.rfind(' ');
            std::string carry;
            if (sp != std::string::npos && sp > 0) {
                carry = cur.substr(sp + 1);
                cur.erase(sp);
            }
            lines.push_back(carry);
            col = 0;
            for (std::size_t k = 0; k < carry.size(); ++k)
                if ((static_cast<unsigned char>(carry[k]) & 0xC0) != 0x80) ++col;
        }
        if (c == '\t' || c == '\r') {
            ++i;
            continue;
        }
        lines.back().append(text, i, len);
        ++col;
        i += len;
    }
    if (static_cast<int>(lines.size()) > rows) lines.erase(lines.begin(), lines.end() - rows);
    return lines;
}

void draw(const Session& s, const SysMetrics& m, bool waiting, const std::string& mon_status) {
    const int W = std::max(60, std::min(term_width(), 120));
    std::string o = "\033[H";
    auto line = [&](const std::string& content) { o += content + kReset + "\033[K\n"; };
    auto rule = [&] {
        std::string r = kDim;
        for (int i = 0; i < W - 2; ++i) r += "\xE2\x94\x80";
        line(" " + r);
    };
    char buf[512];
    std::snprintf(buf, sizeof buf, " %s\xE2\x97\x86 ADAMANT SIMD-VIC%s%s \xC2\xB7 live inference%s", kTeal, kReset, kDim, kReset);
    std::string right;
    if (s.active) right = s.model + " \xC2\xB7 " + s.isa + " \xC2\xB7 " + std::to_string(s.threads) + " threads";
    line(std::string(buf) + "   " + kText + right);
    rule();

    double tps = 0;
    if (!s.fwd_ns.empty()) {
        std::uint64_t sum = 0;
        for (auto v : s.fwd_ns) sum += v;
        tps = 1e9 * static_cast<double>(s.fwd_ns.size()) / static_cast<double>(sum);
    }
    if (s.ended && s.final_tps > 0) tps = s.final_tps;
    const double gbs = static_cast<double>(s.weight_bytes) * tps / 1e9;
    std::snprintf(buf, sizeof buf, " %stokens%s %-6llu %sspeed%s %s%6.1f tok/s%s  %s  %sweights%s %5.1f GB/s  %sprefill%s %.1f tok/s",
                  kDim, kText, static_cast<unsigned long long>(s.n_tokens), kDim, kText, kTeal, tps, kText,
                  (std::string(kTeal) + sparkline(s.tps_hist, 24) + kText).c_str(), kDim, kText, gbs, kDim, kText, s.prefill_tps);
    line(buf);

    // per-core bars, 2 columns
    const std::size_t n = m.core_load.size();
    std::snprintf(buf, sizeof buf, " %sCPU%s  %5.1f%%  %s%s", kDim, kText, std::isnan(m.cpu_load) ? 0.0 : m.cpu_load, kDim, m.cpu_name.c_str());
    line(buf);
    const int cols = W >= 100 ? 4 : 2;
    const int bw = std::max(6, (W - 4) / cols - 12);
    for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(cols)) {
        std::string l = " ";
        for (int c = 0; c < cols && i + static_cast<std::size_t>(c) < n; ++c) {
            const double v = m.core_load[i + static_cast<std::size_t>(c)];
            std::snprintf(buf, sizeof buf, " %s%2zu%s %s%s%s %3.0f ", kDim, i + static_cast<std::size_t>(c), kTeal, "", bar(v, bw).c_str(), kText, v);
            l += buf;
        }
        line(l);
    }
    if (!std::isnan(m.ram_used_gb) && m.ram_total_gb > 0) {
        const double pct = 100.0 * m.ram_used_gb / m.ram_total_gb;
        std::snprintf(buf, sizeof buf, " %sRAM%s  %5.1f / %.1f GB  %s%s%s %3.0f%%", kDim, kText, m.ram_used_gb, m.ram_total_gb, kTeal,
                      bar(pct, std::max(10, W - 40)).c_str(), kText, pct);
        line(buf);
    }
    line(std::string(" ") + kDim + "metrics: " + m.source + (m.source.rfind("Euclase", 0) == 0 ? "" : "  (Euclase Monitor: " + mon_status + ")"));
    rule();
    const int text_rows = 14;
    if (waiting && !s.active) {
        line(std::string(" ") + kDim + "waiting for a generation on " ADM_RING_NAME "  (adamant run -m model.gguf -p \"...\" --stream)");
        for (int i = 1; i < text_rows; ++i) line("");
    } else {
        auto lines = wrap_tail(s.text, W - 4, text_rows);
        for (int i = 0; i < text_rows; ++i) line(i < static_cast<int>(lines.size()) ? std::string("  ") + kText + lines[static_cast<std::size_t>(i)] : "");
    }
    rule();
    std::string status;
    if (s.ended)
        std::snprintf(buf, sizeof buf, " %sdone%s: %llu tokens, %.1f tok/s average, %llu records dropped", kTeal, kText,
                      static_cast<unsigned long long>(s.n_tokens), s.final_tps, static_cast<unsigned long long>(s.dropped));
    else if (s.active)
        std::snprintf(buf, sizeof buf, " %sstreaming%s over Corundum HFT ring %s \xC2\xB7 ctx %u \xC2\xB7 %u layers", kTeal, kText, ADM_RING_NAME, s.n_ctx, s.n_layer);
    else
        std::snprintf(buf, sizeof buf, " %sidle%s", kDim, kText);
    line(buf);
    o += "\033[J";
    std::fwrite(o.data(), 1, o.size(), stdout);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE con = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(con, &mode)) SetConsoleMode(con, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
    bool takeover = false, exit_on_end = false, no_monitor = false;
    std::string record_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--takeover") takeover = true;
        else if (a == "--exit-on-end") exit_on_end = true;
        else if (a == "--no-monitor") no_monitor = true;
        else if (a == "--record" && i + 1 < argc) record_path = argv[++i];
        else {
            std::fprintf(stderr,
                         "usage: adamant-view [--exit-on-end] [--record events.jsonl] [--no-monitor] [--takeover]\n"
                         "Live view of `adamant run --stream`: tokens from Corundum ring " ADM_RING_NAME
                         ", CPU/RAM from Euclase Monitor (" EMS_RING_NAME ").\n");
            return 2;
        }
    }
    std::FILE* rec = record_path.empty() ? nullptr : std::fopen(record_path.c_str(), "w");
    const auto t_start = Clock::now();

    std::fputs("\033[?25l\033[2J", stdout);  // hide cursor, clear
    crd_reader tr{};
    bool tok_open = false;
    Session s;
    SysMetrics m;
    LocalSampler local;
    MonitorReader mon;
    auto last_draw = Clock::now() - std::chrono::seconds(1);
    auto last_local = Clock::now() - std::chrono::seconds(1);
    auto last_try = Clock::now() - std::chrono::seconds(1);
    bool quit = false;

    auto json_str = [](const std::string& t) {
        std::string o = "\"";
        for (unsigned char c : t) {
            if (c == '"' || c == '\\') { o += '\\'; o += static_cast<char>(c); }
            else if (c == '\n') o += "\\n";
            else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
            else o += static_cast<char>(c);
        }
        return o + "\"";
    };

    while (!quit) {
        if (!tok_open && secs_since(last_try) > 0.25) {
            last_try = Clock::now();
            if (crd_reader_open(&tr, ADM_RING_NAME, CRD_OPEN_TAKEOVER) == CRD_OK) tok_open = true;
        }
        if (tok_open) {
            crd_msg msg;
            crd_status st;
            while ((st = crd_read(&tr, &msg)) == CRD_OK) {
                // copy the record out of shared memory before parsing it
                std::vector<unsigned char> b(static_cast<const unsigned char*>(msg.data), static_cast<const unsigned char*>(msg.data) + msg.size);
                const double t = secs_since(t_start);
                if (msg.type == ADM_REC_SESSION && b.size() >= sizeof(adm_session)) {
                    adm_session h;
                    std::memcpy(&h, b.data(), sizeof h);
                    s = Session{};
                    s.active = true;
                    s.model.assign(reinterpret_cast<const char*>(b.data()) + sizeof h, std::min<std::size_t>(h.text_len, b.size() - sizeof h));
                    s.isa.assign(h.isa, strnlen(h.isa, sizeof h.isa));
                    s.threads = h.n_threads;
                    s.n_ctx = h.n_ctx;
                    s.n_layer = h.n_layer;
                    s.weight_bytes = h.weight_bytes_per_token;
                    if (rec) std::fprintf(rec, "{\"t\":%.4f,\"ev\":\"session\",\"model\":%s,\"isa\":%s,\"threads\":%u,\"weight_bytes\":%llu}\n", t,
                                          json_str(s.model).c_str(), json_str(s.isa).c_str(), s.threads, static_cast<unsigned long long>(s.weight_bytes));
                } else if (msg.type == ADM_REC_PROMPT && b.size() == sizeof(adm_prompt)) {
                    adm_prompt h;
                    std::memcpy(&h, b.data(), sizeof h);
                    s.n_prompt = h.n_prompt;
                    s.prefill_tps = h.prefill_ns ? h.n_prompt * 1e9 / static_cast<double>(h.prefill_ns) : 0;
                    if (rec) std::fprintf(rec, "{\"t\":%.4f,\"ev\":\"prompt\",\"n\":%u,\"prefill_ns\":%llu}\n", t, h.n_prompt, static_cast<unsigned long long>(h.prefill_ns));
                } else if (msg.type == ADM_REC_TOKEN && b.size() >= sizeof(adm_token)) {
                    adm_token h;
                    std::memcpy(&h, b.data(), sizeof h);
                    const std::string piece(reinterpret_cast<const char*>(b.data()) + sizeof h, std::min<std::size_t>(h.text_len, b.size() - sizeof h));
                    s.text += piece;
                    s.n_tokens = h.seq + 1;
                    if (h.seq > 0 && h.forward_ns > 0) {
                        s.fwd_ns.push_back(h.forward_ns);
                        if (s.fwd_ns.size() > 16) s.fwd_ns.pop_front();
                        s.tps_hist.push_back(1e9 / static_cast<double>(h.forward_ns));
                        if (s.tps_hist.size() > 64) s.tps_hist.pop_front();
                    }
                    if (rec) std::fprintf(rec, "{\"t\":%.4f,\"ev\":\"token\",\"seq\":%llu,\"id\":%d,\"fwd_ns\":%llu,\"text\":%s}\n", t,
                                          static_cast<unsigned long long>(h.seq), h.id, static_cast<unsigned long long>(h.forward_ns), json_str(piece).c_str());
                } else if (msg.type == ADM_REC_END && b.size() == sizeof(adm_end)) {
                    adm_end h;
                    std::memcpy(&h, b.data(), sizeof h);
                    s.ended = true;
                    s.dropped = h.dropped;
                    s.final_tps = h.gen_ns && h.n_generated > 1 ? (h.n_generated - 1) * 1e9 / static_cast<double>(h.gen_ns) : 0;
                    if (rec) std::fprintf(rec, "{\"t\":%.4f,\"ev\":\"end\",\"n\":%llu,\"gen_ns\":%llu,\"dropped\":%llu}\n", t,
                                          static_cast<unsigned long long>(h.n_generated), static_cast<unsigned long long>(h.gen_ns),
                                          static_cast<unsigned long long>(h.dropped));
                    if (exit_on_end) quit = true;
                }
            }
            crd_release(&tr);
            if (st == CRD_ECORRUPT) {
                crd_reader_close(&tr);
                tok_open = false;
            } else if (crd_writer_closed(&tr) && crd_reader_pending(&tr) == 0) {
                // the producer finished; reattach so the next run's ring is picked up
                crd_reader_close(&tr);
                tok_open = false;
            }
        }

        bool fresh = false;
        if (!no_monitor) fresh = mon.poll(m, takeover);
        if (!mon.connected() && secs_since(last_local) >= 0.5) {
            last_local = Clock::now();
            fresh = local.sample(m) || fresh;
        }
        if (fresh && rec) {
            std::fprintf(rec, "{\"t\":%.4f,\"ev\":\"metrics\",\"src\":%s,\"cpu\":%.2f,\"ram_used_gb\":%.3f,\"ram_total_gb\":%.3f,\"cores\":[", secs_since(t_start),
                         json_str(m.source).c_str(), std::isnan(m.cpu_load) ? -1.0 : m.cpu_load, std::isnan(m.ram_used_gb) ? -1.0 : m.ram_used_gb,
                         std::isnan(m.ram_total_gb) ? -1.0 : m.ram_total_gb);
            for (std::size_t i = 0; i < m.core_load.size(); ++i) std::fprintf(rec, "%s%.1f", i ? "," : "", m.core_load[i]);
            std::fprintf(rec, "]}\n");
            std::fflush(rec);
        }
        if (secs_since(last_draw) >= 0.1 || quit) {
            last_draw = Clock::now();
            draw(s, m, !tok_open, mon.status());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    std::fputs("\033[?25h\n", stdout);
    if (tok_open) crd_reader_close(&tr);
    if (rec) std::fclose(rec);
    return 0;
}
