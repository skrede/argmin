#ifndef HPP_GUARD_ARGMIN_TESTS_GOLDEN_GOLDEN_IDENTITY_H
#define HPP_GUARD_ARGMIN_TESTS_GOLDEN_GOLDEN_IDENTITY_H

#include <string>

namespace argmin::golden
{

inline std::string compiler_tag()
{
#if defined(__clang__)
    return "clang-" + std::to_string(__clang_major__);
#elif defined(__GNUC__)
    return "gcc-" + std::to_string(__GNUC__);
#elif defined(_MSC_VER)
    return "msvc-" + std::to_string(_MSC_VER);
#else
    return "unknown-compiler";
#endif
}

inline std::string arch_tag()
{
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm32";
#else
    return "unknown-arch";
#endif
}

// The vector width and FMA availability Eigen compiles for change the low bits, so a build for a wider
// instruction set never claims the baseline identity.
inline std::string isa_tag()
{
#if defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__) && defined(__FMA__)
    return "avx2-fma";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__AVX__)
    return "avx";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86-64-baseline";
#else
    return arch_tag();
#endif
}

inline std::string flavor_tag()
{
#if defined(NDEBUG)
    return "NDEBUG";
#else
    return "DEBUG";
#endif
}

inline std::string running_identity()
{
    return compiler_tag() + ";" + arch_tag() + ";" + flavor_tag() + ";" + isa_tag();
}

// Glibc picks FMA variants of exp, log and pow at run time, so the CPU is reported beside a mismatch.
inline std::string runtime_fma()
{
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("fma") ? "yes" : "no";
#else
    return "unknown";
#endif
}

}

#endif
