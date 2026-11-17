// Adamant SIMD-VIC by Euclase: read-only zero-copy file mapping.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace adamant {

// Maps a whole file read-only. Weights are used in place from the mapping: no
// copy, no heap allocation proportional to the model size. Pages are loaded by
// the OS on first touch and can be dropped again under memory pressure (they
// are backed by the file, not by the page file / swap), so a model larger than
// free RAM degrades to disk speed instead of exhausting memory.
//   Linux:   open + mmap(PROT_READ, MAP_PRIVATE)
//   Windows: CreateFileW + CreateFileMappingW(PAGE_READONLY) + MapViewOfFile
class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept;
    MappedFile& operator=(MappedFile&& o) noexcept;

    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    // Hints that [offset, offset+len) will be read soon (madvise WILLNEED /
    // PrefetchVirtualMemory). Best effort, never fails.
    void advise_willneed(std::size_t offset, std::size_t len) const noexcept;
    // Reads one byte per page so every page is resident. Returns a checksum so
    // the loop cannot be optimised away. Used by benchmarks (--warm).
    std::uint64_t touch_all() const noexcept;

private:
    void close() noexcept;
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#endif
};

}  // namespace adamant
