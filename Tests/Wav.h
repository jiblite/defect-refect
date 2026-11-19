#pragma once
// Minimal WAV reader for the command-line tests (PCM 16/24/32, float 32), and a plain change of sample rate for the
// tests that take a file at another rate than the app works at (the app has its own).
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include "../Source/Core/Dsp.h"
namespace dr {
inline bool readWav (const std::string& path, Stereo& out, double& sr)
{
    FILE* f = std::fopen (path.c_str(), "rb"); if (! f) return false;
    std::vector<unsigned char> d; unsigned char buf[65536]; size_t n;
    while ((n = std::fread (buf, 1, sizeof buf, f)) > 0) d.insert (d.end(), buf, buf + n);
    std::fclose (f);
    if (d.size() < 12 || std::memcmp (d.data(), "RIFF", 4) || std::memcmp (d.data() + 8, "WAVE", 4)) return false;
    int fmt = 0, ch = 0, bits = 0; size_t p = 12;
    while (p + 8 <= d.size())
    {
        uint32_t sz; std::memcpy (&sz, &d[p + 4], 4);
        if (! std::memcmp (&d[p], "fmt ", 4)) { uint16_t a, c, b; uint32_t s; std::memcpy (&a, &d[p + 8], 2); std::memcpy (&c, &d[p + 10], 2); std::memcpy (&s, &d[p + 12], 4); std::memcpy (&b, &d[p + 22], 2); fmt = a; ch = c; sr = s; bits = b; if (fmt == 0xFFFE) { uint16_t sub; std::memcpy (&sub, &d[p + 32], 2); fmt = sub; } }
        if (! std::memcmp (&d[p], "data", 4))
        {
            const size_t bps = (size_t) bits / 8, frames = std::min<size_t> (sz, d.size() - p - 8) / (bps * (size_t) ch);
            out.l.assign (frames, 0); out.r.assign (frames, 0);
            for (size_t i = 0; i < frames; ++i)
                for (int c = 0; c < std::min (ch, 2); ++c)
                {
                    const unsigned char* s = &d[p + 8 + (i * (size_t) ch + (size_t) c) * bps]; double v = 0;
                    if (fmt == 3 && bits == 32) { float x; std::memcpy (&x, s, 4); v = x; }
                    else if (bits == 16) { int16_t x; std::memcpy (&x, s, 2); v = x / 32768.0; }
                    else if (bits == 24) { int32_t x = (s[0] << 8) | (s[1] << 16) | (s[2] << 24); v = (x >> 8) / 8388608.0; }
                    else if (bits == 32) { int32_t x; std::memcpy (&x, s, 4); v = x / 2147483648.0; }
                    (c == 0 ? out.l : out.r)[i] = std::isfinite (v) ? v : 0.0;       // as the app's loader: a value that is not a number counts as silence
                }
            if (ch == 1) out.r = out.l;
            return true;
        }
        p += 8 + sz + (sz & 1);
    }
    return false;
}
// x at another sample rate: a windowed sinc (24 crossings each side, Hann), low-passed at the lower of the two rates.
inline Vec resampled (const Vec& x, double from, double to)
{
    const long n = (long) x.size(), m = (long) std::floor ((double) n * to / from); Vec o ((size_t) std::max (m, 0L), 0.0);
    const double step = from / to, fc = std::min (1.0, to / from); const int half = (int) std::ceil (24.0 / fc);
    for (long i = 0; i < m; ++i)
    {
        const double t = (double) i * step; const long c = (long) std::floor (t); double acc = 0.0;
        for (long k = c - half + 1; k <= c + half; ++k)
        {
            if (k < 0 || k >= n) continue;
            const double d = (t - (double) k) * fc, w = 0.5 + 0.5 * std::cos (PI * (t - (double) k) / (double) half);
            acc += x[(size_t) k] * w * fc * (std::abs (d) < 1e-12 ? 1.0 : std::sin (PI * d) / (PI * d));
        }
        o[(size_t) i] = acc;
    }
    return o;
}
inline Stereo resampled (const Stereo& y, double from, double to) { Stereo o; o.l = resampled (y.l, from, to); o.r = resampled (y.r, from, to); return o; }
}
