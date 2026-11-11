#pragma once
// Small numeric helpers that mirror the numpy / scipy calls the analysis was developed with,
// so the app's numbers match the Python prototype. Plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

#include "Fft.h"

namespace dr
{
using Vec = std::vector<double>;
constexpr double SR = 48000.0;
constexpr double PI = 3.14159265358979323846;

inline size_t nextPow2 (size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }

// scipy.ndimage.uniform_filter1d (origin 0). mode 0 = 'reflect' (scipy default), 1 = 'nearest'.
inline Vec uniformFilter (const Vec& x, int size, int mode = 0)
{
    const long n = (long) x.size(); Vec out ((size_t) n);
    if (n == 0) return out;
    size = std::max (1, size);
    const long lo = size / 2, hi = size - 1 - size / 2;
    auto at = [&] (long i) -> double
    {
        if (mode == 1) return x[(size_t) std::clamp (i, 0L, n - 1)];
        // reflect: (d c b a | a b c d | d c b a)
        const long p = 2 * n;
        long j = ((i % p) + p) % p;
        if (j >= n) j = p - 1 - j;
        return x[(size_t) j];
    };
    double s = 0.0;
    for (long i = -lo; i <= hi; ++i) s += at (i);
    for (long i = 0; i < n; ++i)
    {
        out[(size_t) i] = s / size;
        s += at (i + hi + 1) - at (i - lo);
    }
    return out;
}

// scipy.ndimage.median_filter (1-D, mode 'reflect').
inline Vec medianFilter (const Vec& x, int size)
{
    const long n = (long) x.size(); Vec out ((size_t) n), w ((size_t) size);
    const long lo = size / 2;
    for (long i = 0; i < n; ++i)
    {
        for (long k = 0; k < size; ++k)
        {
            long j = i - lo + k; const long p = 2 * n;
            j = ((j % p) + p) % p; if (j >= n) j = p - 1 - j;
            w[(size_t) k] = x[(size_t) j];
        }
        std::nth_element (w.begin(), w.begin() + size / 2, w.end());
        out[(size_t) i] = w[(size_t) size / 2];
    }
    return out;
}

inline double mean (const Vec& v) { return v.empty() ? 0.0 : std::accumulate (v.begin(), v.end(), 0.0) / (double) v.size(); }
inline double variance (const Vec& v) { const double m = mean (v); double s = 0; for (double a : v) s += (a - m) * (a - m); return v.empty() ? 0.0 : s / (double) v.size(); }
inline double stdev (const Vec& v) { return std::sqrt (variance (v)); }

// numpy.percentile, linear interpolation.
inline double percentile (Vec v, double q)
{
    if (v.empty()) return 0.0;
    std::sort (v.begin(), v.end());
    const double pos = q / 100.0 * (double) (v.size() - 1);
    const size_t i = (size_t) std::floor (pos); const size_t j = std::min (i + 1, v.size() - 1);
    return v[i] + (v[j] - v[i]) * (pos - (double) i);
}
inline double median (const Vec& v) { return percentile (v, 50.0); }

inline Vec hanningSym (size_t n)    // numpy.hanning
{
    Vec w (n);
    if (n == 1) { w[0] = 1.0; return w; }
    for (size_t i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos (2.0 * PI * (double) i / (double) (n - 1));
    return w;
}
inline Vec hannPeriodic (size_t n)  // scipy.signal.get_window('hann', n)
{
    Vec w (n);
    for (size_t i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos (2.0 * PI * (double) i / (double) n);
    return w;
}

// Real FFT of x zero-padded to n (power of two). Returns n/2+1 bins.
inline std::vector<std::complex<double>> rfft (const Vec& x, size_t n)
{
    Vec re (n, 0.0), im (n, 0.0);
    for (size_t i = 0; i < std::min (n, x.size()); ++i) re[i] = x[i];
    fft (re, im);
    std::vector<std::complex<double>> out (n / 2 + 1);
    for (size_t k = 0; k <= n / 2; ++k) out[k] = { re[k], im[k] };
    return out;
}
// Inverse of rfft (real output of length n).
inline Vec irfft (const std::vector<std::complex<double>>& X, size_t n)
{
    Vec re (n, 0.0), im (n, 0.0);
    for (size_t k = 0; k <= n / 2 && k < X.size(); ++k)
    {
        re[k] = X[k].real(); im[k] = X[k].imag();
        if (k > 0 && k < n / 2) { re[n - k] = X[k].real(); im[n - k] = -X[k].imag(); }
    }
    fft (re, im, true);
    return re;
}

// Least squares fit of t to a*sin + b*cos + c at angular frequency w (samples at rate fs).
struct SineFit { double a = 0, b = 0, c = 0, r2 = 0; };
inline SineFit fitSine (const Vec& t, double rate, double fs)
{
    const size_t n = t.size(); SineFit f;
    if (n < 3) return f;
    double A[3][3] {}, B[3] {};
    for (size_t i = 0; i < n; ++i)
    {
        const double ph = 2.0 * PI * rate * (double) i / fs, r[3] = { std::sin (ph), std::cos (ph), 1.0 };
        for (int p = 0; p < 3; ++p) { B[p] += r[p] * t[i]; for (int q = 0; q < 3; ++q) A[p][q] += r[p] * r[q]; }
    }
    // Solve 3x3 by Gaussian elimination with partial pivoting.
    double M[3][4];
    for (int p = 0; p < 3; ++p) { for (int q = 0; q < 3; ++q) M[p][q] = A[p][q]; M[p][3] = B[p]; }
    for (int c = 0; c < 3; ++c)
    {
        int piv = c; for (int r = c + 1; r < 3; ++r) if (std::abs (M[r][c]) > std::abs (M[piv][c])) piv = r;
        for (int q = 0; q < 4; ++q) std::swap (M[c][q], M[piv][q]);
        if (std::abs (M[c][c]) < 1e-15) return f;
        for (int r = 0; r < 3; ++r) if (r != c) { const double k = M[r][c] / M[c][c]; for (int q = c; q < 4; ++q) M[r][q] -= k * M[c][q]; }
    }
    f.a = M[0][3] / M[0][0]; f.b = M[1][3] / M[1][1]; f.c = M[2][3] / M[2][2];
    double res = 0.0; const double v = variance (t);
    Vec e (n);
    for (size_t i = 0; i < n; ++i) { const double ph = 2.0 * PI * rate * (double) i / fs; e[i] = t[i] - (f.a * std::sin (ph) + f.b * std::cos (ph) + f.c); }
    res = variance (e);
    f.r2 = v > 0 ? 1.0 - res / v : 0.0;
    return f;
}

// Stereo buffer: two channels of equal length.
struct Stereo
{
    Vec l, r;
    size_t size() const { return l.size(); }
    Vec mid() const { Vec m (l.size()); for (size_t i = 0; i < l.size(); ++i) m[i] = 0.5 * (l[i] + r[i]); return m; }
    Stereo slice (size_t a, size_t n) const
    {
        Stereo s; s.l.assign (n, 0.0); s.r.assign (n, 0.0);
        for (size_t i = 0; i < n && a + i < l.size(); ++i) { s.l[i] = l[a + i]; s.r[i] = r[a + i]; }
        return s;
    }
    double rms() const { double e = 0; for (size_t i = 0; i < l.size(); ++i) e += 0.5 * (l[i] * l[i] + r[i] * r[i]); return l.empty() ? 0.0 : std::sqrt (e / (double) l.size()); }
    double peak() const { double p = 0; for (size_t i = 0; i < l.size(); ++i) p = std::max ({ p, std::abs (l[i]), std::abs (r[i]) }); return p; }
};

} // namespace dr

namespace dr
{
// Mixed-radix complex FFT for any size whose prime factors are small (falls back to a plain DFT
// for a large prime factor). Used where the prototype's FFT size was not a power of two.
inline void fftAny (std::vector<std::complex<double>>& a)
{
    const size_t n = a.size(); if (n <= 1) return;
    size_t p = 2; while (p * p <= n && n % p) ++p; if (n % p) p = n;   // smallest factor
    const size_t m = n / p;
    if (p == n)
    {
        std::vector<std::complex<double>> o (n);
        for (size_t k = 0; k < n; ++k) { std::complex<double> s = 0; for (size_t j = 0; j < n; ++j) s += a[j] * std::polar (1.0, -2.0 * PI * (double) ((j * k) % n) / (double) n); o[k] = s; }
        a.swap (o); return;
    }
    std::vector<std::vector<std::complex<double>>> sub (p, std::vector<std::complex<double>> (m));
    for (size_t r = 0; r < p; ++r) for (size_t j = 0; j < m; ++j) sub[r][j] = a[j * p + r];
    for (auto& s : sub) fftAny (s);
    for (size_t k = 0; k < n; ++k)
    {
        std::complex<double> s = 0;
        for (size_t r = 0; r < p; ++r) s += sub[r][k % m] * std::polar (1.0, -2.0 * PI * (double) ((r * k) % n) / (double) n);
        a[k] = s;
    }
}
} // namespace dr
