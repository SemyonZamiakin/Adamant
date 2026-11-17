// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Euclase
#include "adamant/mmap_file.hpp"

#include "adamant/common.hpp"

#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace adamant {

#ifdef _WIN32
namespace {
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}
std::string last_error() { return "error " + std::to_string(static_cast<unsigned long>(GetLastError())); }
}  // namespace

MappedFile::MappedFile(const std::string& path) {
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw Error("cannot open " + path + ": " + last_error());
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) {
        CloseHandle(f);
        throw Error("cannot map " + path + ": empty or unreadable file");
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) {
        const std::string e = last_error();
        CloseHandle(f);
        throw Error("CreateFileMapping failed for " + path + ": " + e);
    }
    void* v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!v) {
        const std::string e = last_error();
        CloseHandle(m);
        CloseHandle(f);
        throw Error("MapViewOfFile failed for " + path + ": " + e);
    }
    file_ = f;
    mapping_ = m;
    data_ = static_cast<const std::uint8_t*>(v);
    size_ = static_cast<std::size_t>(sz.QuadPart);
}

void MappedFile::close() noexcept {
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle(mapping_);
    if (file_) CloseHandle(file_);
    data_ = nullptr;
    mapping_ = file_ = nullptr;
    size_ = 0;
}

void MappedFile::advise_willneed(std::size_t offset, std::size_t len) const noexcept {
    if (!data_ || offset >= size_) return;
    if (len > size_ - offset) len = size_ - offset;
    // PrefetchVirtualMemory exists since Windows 8; declared here and looked up at
    // run time so it builds with SDK headers that target older Windows versions.
    struct RangeEntry {
        void* VirtualAddress;
        SIZE_T NumberOfBytes;
    } r{const_cast<std::uint8_t*>(data_ + offset), len};
    using Fn = BOOL(WINAPI*)(HANDLE, ULONG_PTR, RangeEntry*, ULONG);
    static Fn fn = reinterpret_cast<Fn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory")));
    if (fn) fn(GetCurrentProcess(), 1, &r, 0);
}
#else
MappedFile::MappedFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw Error("cannot open " + path + ": " + std::strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        throw Error("cannot map " + path + ": empty or unreadable file");
    }
    void* v = mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    const int err = errno;
    ::close(fd);  // the mapping keeps its own reference to the file
    if (v == MAP_FAILED) throw Error("mmap failed for " + path + ": " + std::strerror(err));
    data_ = static_cast<const std::uint8_t*>(v);
    size_ = static_cast<std::size_t>(st.st_size);
}

void MappedFile::close() noexcept {
    if (data_) munmap(const_cast<std::uint8_t*>(data_), size_);
    data_ = nullptr;
    size_ = 0;
}

void MappedFile::advise_willneed(std::size_t offset, std::size_t len) const noexcept {
    if (!data_ || offset >= size_) return;
    if (len > size_ - offset) len = size_ - offset;
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const std::size_t start = offset / page * page;
    madvise(const_cast<std::uint8_t*>(data_ + start), len + (offset - start), MADV_WILLNEED);
}
#endif

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& o) noexcept { *this = std::move(o); }

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
    if (this != &o) {
        close();
        data_ = std::exchange(o.data_, nullptr);
        size_ = std::exchange(o.size_, 0);
#ifdef _WIN32
        file_ = std::exchange(o.file_, nullptr);
        mapping_ = std::exchange(o.mapping_, nullptr);
#endif
    }
    return *this;
}

std::uint64_t MappedFile::touch_all() const noexcept {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < size_; i += 4096) sum += data_[i];
    return sum;
}

}  // namespace adamant
