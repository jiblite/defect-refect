// The change of sample rate (Source/Core/Resample.h) against the plain formula it is built from, and what it promises:
// no delay, a tone below the cut keeps its level, and coming down from a higher rate nothing folds into the hearing range.
//   g++ -O2 -std=c++17 -o drresample Tests/resample.cpp && ./drresample        (prints one line per check; exit 1 on a failure)
#include <chrono>
#include <cstdio>
#include <random>
#include "../Source/Core/Resample.h"

// the plain formula: every output sample from the filter worked out at its own place
static dr::Vec plain (const dr::Vec& x, double from, double to)
{
    const long n = (long) x.size(), m = (long) std::floor ((double) n * to / from); dr::Vec o ((size_t) std::max (m, 0L), 0.0);
    const double step = from / to, fc = std::min (1.0, to / from); const int half = (int) std::ceil (24.0 / fc);
    for (long i = 0; i < m; ++i)
    {
        const double t = (double) i * step; const long c = (long) std::floor (t); double acc = 0.0;
        for (long k = c - half + 1; k <= c + half; ++k)
        {
            if (k < 0 || k >= n) continue;
            const double d = (t - (double) k) * fc, w = 0.5 + 0.5 * std::cos (dr::PI * (t - (double) k) / (double) half);
            acc += x[(size_t) k] * w * fc * (std::abs (d) < 1e-12 ? 1.0 : std::sin (dr::PI * d) / (dr::PI * d));
        }
        o[(size_t) i] = acc;
    }
    return o;
}
static double rms (const dr::Vec& v, size_t a, size_t b) { double s = 0; for (size_t i = a; i < b; ++i) s += v[i] * v[i]; return std::sqrt (s / (double) std::max<size_t> (b - a, 1)); }

int main()
{
    int bad = 0; std::mt19937 rng (5); std::uniform_real_distribution<double> u (-1.0, 1.0);
    // 1. equal to the plain formula: noise at many pairs of rates and lengths
    const double pairs[][2] = { { 44100, 48000 }, { 96000, 48000 }, { 88200, 48000 }, { 192000, 48000 }, { 176400, 48000 }, { 32000, 48000 }, { 22050, 48000 }, { 11025, 48000 },
                                { 8000, 48000 }, { 47952, 48000 }, { 44056, 48000 }, { 44101, 48000 }, { 48000, 44100 }, { 44100.5, 48000 }, { 192001, 48000 } };
    for (auto& pr : pairs)
        for (long n : { 0L, 1L, 2L, 47L, 1000L, 20000L })
        {
            dr::Vec x ((size_t) n); for (auto& v : x) v = u (rng);
            const dr::Vec a = dr::resampled (x, pr[0], pr[1]), b = plain (x, pr[0], pr[1]); double d = 0;
            if (a.size() != b.size()) { std::printf ("LENGTH %g to %g, %ld samples: %zu against %zu\n", pr[0], pr[1], n, a.size(), b.size()); ++bad; continue; }
            for (size_t i = 0; i < a.size(); ++i) d = std::max (d, std::abs (a[i] - b[i]));
            if (d > 1e-9) { std::printf ("DIFFERENT %g to %g, %ld samples: %.3g\n", pr[0], pr[1], n, d); ++bad; }
        }
    std::printf ("equal to the plain formula: %s\n", bad ? "NO" : "yes (15 pairs of rates, 6 lengths each)");
    // 2. no delay and level kept: a 1 kHz tone at 44.1 and at 96 kHz against the same tone made at 48 kHz
    for (double from : { 44100.0, 96000.0 })
    {
        const long n = (long) from; dr::Vec x ((size_t) n); for (long i = 0; i < n; ++i) x[(size_t) i] = std::sin (2 * dr::PI * 1000.0 * (double) i / from);
        const dr::Vec y = dr::resampled (x, from, 48000.0); double d = 0;
        for (size_t i = 2000; i + 2000 < y.size(); ++i) d = std::max (d, std::abs (y[i] - std::sin (2 * dr::PI * 1000.0 * (double) i / 48000.0)));
        const bool ok = y.size() == 48000 && d < 1e-4; if (! ok) ++bad;
        std::printf ("1 kHz tone from %g Hz: %zu samples, furthest from the same tone at 48 kHz %.2g: %s\n", from, y.size(), d, ok ? "in place" : "WRONG");
    }
    // 3. coming down from 96 kHz: a 30 kHz tone must not come out as an 18 kHz one
    {
        const long n = 96000; dr::Vec x ((size_t) n); for (long i = 0; i < n; ++i) x[(size_t) i] = std::sin (2 * dr::PI * 30000.0 * (double) i / 96000.0);
        const dr::Vec y = dr::resampled (x, 96000.0, 48000.0); const double left = 20 * std::log10 (rms (y, 2000, y.size() - 2000) / std::sqrt (0.5) + 1e-30);
        const bool ok = left < -60; if (! ok) ++bad;       // the old loader of the app let it through at full level, as an 18 kHz tone
        std::printf ("30 kHz tone from 96 kHz: %.0f dB of it is left: %s\n", left, ok ? "does not fold down" : "FOLDS DOWN");
    }
    // 4. time: ten minutes of one side at 44.1 kHz
    {
        dr::Vec x ((size_t) (44100 * 600)); for (auto& v : x) v = u (rng);
        const auto t0 = std::chrono::steady_clock::now(); const dr::Vec y = dr::resampled (x, 44100.0, 48000.0);
        std::printf ("ten minutes at 44.1 kHz, one side: %.1f s (%zu samples)\n", std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count(), y.size());
    }
    std::printf (bad ? "FAILED: %d\n" : "all checks passed\n", bad);
    return bad ? 1 : 0;
}
