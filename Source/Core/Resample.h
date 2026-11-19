#pragma once
// A change of sample rate without delay, for a file that is not at the rate the app works at. The app's loader and the
// command-line tests both use it, so a file at 44.1 or 96 kHz gives the same samples in both.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "Dsp.h"

namespace dr {

// x at another sample rate: sample i of the result is the sound at time i / to, so the result lines up with x to the
// sample. A windowed sinc (24 crossings each side, Hann), low-passed at the lower of the two rates: coming down from
// 96 kHz, nothing above 24 kHz folds into the hearing range. With whole-number rates the filter is worked out once per
// phase (160 phases for 44.1 to 48 kHz); any other pair of rates works it out per sample: the same numbers, slower.
inline Vec resampled (const Vec& x, double from, double to)
{
    const long n = (long) x.size(), m = (long) std::floor ((double) n * to / from); Vec o ((size_t) std::max (m, 0L), 0.0);
    if (m <= 0) return o;
    const double step = from / to, fc = std::min (1.0, to / from); const int half = (int) std::ceil (24.0 / fc); const long taps = 2L * half;
    auto tap = [fc, half] (double d)      // the filter, d input samples away from the output's place
    {
        const double u = d * fc;
        return (0.5 + 0.5 * std::cos (PI * d / (double) half)) * fc * (std::abs (u) < 1e-12 ? 1.0 : std::sin (PI * u) / (PI * u));
    };
    long long p = 0, q = 0;               // step = p / q in whole numbers, when both rates are whole numbers
    if (from >= 1.0 && to >= 1.0 && from < 1e9 && to < 1e9 && std::abs (from - std::round (from)) < 1e-9 && std::abs (to - std::round (to)) < 1e-9)
    {
        const long long a = (long long) std::llround (from), b = (long long) std::llround (to), g = std::gcd (a, b); p = a / g; q = b / g;
    }
    if (q > 0 && (double) q * (double) taps <= 4.0e6)
    {
        std::vector<double> table ((size_t) q * (size_t) taps);              // phase r: an output r / q past an input sample
        for (long long r = 0; r < q; ++r)
            for (long j = 0; j < taps; ++j) table[(size_t) r * (size_t) taps + (size_t) j] = tap ((double) r / (double) q - (double) (j - half + 1));
        for (long i = 0; i < m; ++i)
        {
            const long long at = (long long) i * p; const long c = (long) (at / q); const double* t = &table[(size_t) (at % q) * (size_t) taps];      // i * step = c + r / q, exactly
            const long k0 = c - half + 1, j0 = std::max (0L, -k0), j1 = std::min (taps, n - k0); double acc = 0.0;
            for (long j = j0; j < j1; ++j) acc += x[(size_t) (k0 + j)] * t[j];
            o[(size_t) i] = acc;
        }
        return o;
    }
    for (long i = 0; i < m; ++i)
    {
        const double t = (double) i * step; const long c = (long) std::floor (t); double acc = 0.0;
        for (long k = std::max (c - half + 1, 0L); k <= std::min (c + (long) half, n - 1); ++k) acc += x[(size_t) k] * tap (t - (double) k);
        o[(size_t) i] = acc;
    }
    return o;
}
inline Stereo resampled (const Stereo& y, double from, double to) { Stereo o; o.l = resampled (y.l, from, to); o.r = resampled (y.r, from, to); return o; }

} // namespace dr
