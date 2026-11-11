#pragma once
// Minimal WAV reader for the command-line tests (PCM 16/24/32, float 32).
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
                    (c == 0 ? out.l : out.r)[i] = v;
                }
            if (ch == 1) out.r = out.l;
            return true;
        }
        p += 8 + sz + (sz & 1);
    }
    return false;
}
}
