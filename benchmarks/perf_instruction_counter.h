#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_PERF_INSTRUCTION_COUNTER_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_PERF_INSTRUCTION_COUNTER_H

// Userspace hardware-instruction counter (RAII) for the timed solve region.
//
// Wraps a single perf_event_open counter configured for the calling thread
// with kernel counting excluded (exclude_kernel=1), so it reports only the
// process's own userspace retired-instruction count. Userspace-only counting
// is permitted at perf_event_paranoid <= 2; on a host that forbids it the
// syscall fails and the counter reports a negative unavailable sentinel
// rather than a silent zero -- a zero would read to the downstream gate as a
// free pass, so the sentinel is deliberately out-of-band and negative.
//
// The counter is deterministic and machine-portable in a way wall time is
// not: instructions/iter is derived downstream as the counted instructions
// divided by the solver iteration count. Linux-only; on other platforms the
// counter never arms and every read returns the unavailable sentinel.

#include <cstdint>

#if defined(__linux__)
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#endif

namespace argmin::bench
{

// Written into the instructions column when the counter could not arm or a
// read failed. Negative and out-of-band so a downstream ratio/ceiling gate
// never mistakes an unavailable counter for a small (free-pass) instruction
// count; a genuine measurement is always strictly positive.
inline constexpr std::int64_t instructions_unavailable = -1;

class perf_instruction_counter
{
public:
    perf_instruction_counter() noexcept
    {
#if defined(__linux__)
        perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.type           = PERF_TYPE_HARDWARE;
        attr.size           = sizeof(attr);
        attr.config         = PERF_COUNT_HW_INSTRUCTIONS;
        attr.disabled       = 1;
        attr.exclude_kernel = 1;  // userspace only: armed at paranoid <= 2
        attr.exclude_hv     = 1;

        // pid = 0 (calling thread), cpu = -1 (follow the thread across cpus),
        // group_fd = -1 (own group), flags = 0.
        const long fd = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0UL);
        if(fd == -1)
            return;

        fd_ = static_cast<int>(fd);
        ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
#endif
    }

    ~perf_instruction_counter()
    {
#if defined(__linux__)
        if(fd_ != -1)
        {
            ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
            close(fd_);
        }
#endif
    }

    perf_instruction_counter(const perf_instruction_counter&)            = delete;
    perf_instruction_counter& operator=(const perf_instruction_counter&) = delete;
    perf_instruction_counter(perf_instruction_counter&&)                 = delete;
    perf_instruction_counter& operator=(perf_instruction_counter&&)      = delete;

    // True when the counter armed and a read yields a real measurement.
    [[nodiscard]] auto armed() const noexcept -> bool { return fd_ != -1; }

    // Accumulated userspace retired-instruction count since construction, or
    // instructions_unavailable if the counter never armed or the read failed.
    [[nodiscard]] auto read() const noexcept -> std::int64_t
    {
#if defined(__linux__)
        if(fd_ == -1)
            return instructions_unavailable;
        std::uint64_t value = 0;
        const ssize_t n = ::read(fd_, &value, sizeof(value));
        if(n != static_cast<ssize_t>(sizeof(value)))
            return instructions_unavailable;
        return static_cast<std::int64_t>(value);
#else
        return instructions_unavailable;
#endif
    }

private:
    int fd_{-1};
};

}

#endif
