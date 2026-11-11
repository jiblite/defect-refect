#pragma once
// Minimal radix-2 complex FFT (size must be a power of two). No dependencies.
#include <cmath>
#include <utility>
#include <vector>

namespace dr
{

inline void fft (std::vector<double>& re, std::vector<double>& im, bool inverse = false)
{
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; ++i)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap (re[i], re[j]); std::swap (im[i], im[j]); }
    }
    for (size_t len = 2; len <= n; len <<= 1)
    {
        const double ang = (inverse ? 2.0 : -2.0) * 3.14159265358979323846 / (double) len;
        const double wr = std::cos (ang), wi = std::sin (ang);
        for (size_t i = 0; i < n; i += len)
        {
            double cr = 1.0, ci = 0.0;
            for (size_t k = 0; k < len / 2; ++k)
            {
                const size_t a = i + k, b = i + k + len / 2;
                const double tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
                const double nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr; cr = nr;
            }
        }
    }
    if (inverse)
        for (size_t i = 0; i < n; ++i) { re[i] /= (double) n; im[i] /= (double) n; }
}

} // namespace dr
