#pragma once
// The 12 effects the detector knows, as processors with the same settings as the prototype's
// fxlib.py, so a setting read from one sound can be put on another. Each runs stereo in,
// stereo out (left output from the left input, right from the right). LFO effects take a
// start phase in cycles, so their movement can line up with the sound it was read from.

#include <map>
#include <random>
#include <string>

#include "Dsp.h"

namespace dr
{
using Params = std::map<std::string, double>;
inline double get (const Params& p, const char* k, double d) { auto it = p.find (k); return it == p.end() ? d : it->second; }

namespace fx
{
// ---------------------------------------------------------------- helpers
inline Vec fracDelay (const Vec& x, const Vec& d)      // x(t - d(t)), linear interpolation
{
    const long n = (long) x.size(); Vec y ((size_t) n);
    for (long t = 0; t < n; ++t)
    {
        const double i = (double) t - d[(size_t) t]; const double f0 = std::floor (i); const long i0 = (long) f0; const double f = i - f0;
        const double a = (i0 >= 0 && i0 < n) ? x[(size_t) i0] : 0.0, b = (i0 + 1 >= 0 && i0 + 1 < n) ? x[(size_t) (i0 + 1)] : 0.0;
        y[(size_t) t] = a * (1 - f) + b * f;
    }
    return y;
}
// LFO value at sample t: sine or triangle, phase in radians, t0 = start offset in seconds.
inline double lfoAt (double t, double rate, double ph, bool tri)
{
    if (! tri) return std::sin (2 * PI * rate * t + ph);
    double c = rate * t + ph / (2 * PI); c -= std::floor (c);
    return 2 * std::abs (2 * c - 1) - 1;
}
inline double startT (const Params& p) { const double r = get (p, "rate", 0); const double ph = get (p, "phase", 0); return r > 1e-3 ? (ph - std::floor (ph)) / r : 0.0; }

struct Biquad
{
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
    double tick (double x) { const double y = b0 * x + z1; z1 = b1 * x - a1 * y + z2; z2 = b2 * x - a2 * y; return y; }
    void setLow (double fc)                                   // scipy butter(2, low)
    {
        const double K = std::tan (PI * std::min (fc, SR * 0.49) / SR), s2 = std::sqrt (2.0), nrm = 1.0 / (1 + s2 * K + K * K);
        b0 = K * K * nrm; b1 = 2 * b0; b2 = b0; a1 = 2 * (K * K - 1) * nrm; a2 = (1 - s2 * K + K * K) * nrm;
    }
    void setPeak (double fc, double q)                        // scipy iirpeak
    {
        const double w0 = 2 * std::min (fc, SR * 0.49) / SR * PI, bw = w0 / q;
        const double beta = std::tan (bw / 2), g = 1 / (1 + beta);
        b0 = 1 - g; b1 = 0; b2 = -(1 - g); a1 = -2 * g * std::cos (w0); a2 = 2 * g - 1;
    }
};

// FFT convolution, output trimmed to x's length.
inline Vec convolve (const Vec& x, const Vec& h)
{
    const size_t n = x.size(), m = h.size(); if (n == 0 || m == 0) return Vec (n, 0.0);
    const size_t N = nextPow2 (2 * m), L = N - m + 1;
    const auto H = rfft (h, N); Vec y (n, 0.0);
    for (size_t s = 0; s < n; s += L)
    {
        Vec blk (std::min (L, n - s)); for (size_t i = 0; i < blk.size(); ++i) blk[i] = x[s + i];
        auto X = rfft (blk, N); for (size_t k = 0; k < X.size(); ++k) X[k] *= H[k];
        const Vec o = irfft (X, N);
        for (size_t i = 0; i < N && s + i < n; ++i) y[s + i] += o[i];
    }
    return y;
}
inline double rmsOf (const Vec& v) { double e = 0; for (double a : v) e += a * a; return v.empty() ? 0.0 : std::sqrt (e / (double) v.size()); }
inline double peakOf (const Vec& v) { double p = 0; for (double a : v) p = std::max (p, std::abs (a)); return p; }

// ---------------------------------------------------------------- the effects (mono in -> one channel out, `side` = 0 left, 1 right)
inline Vec flanger (const Vec& x, const Params& p, int side)
{
    const double rate = get (p, "rate", 0.3), dmin = get (p, "dmin", 0.5e-3), dmax = get (p, "dmax", 5e-3), depth = get (p, "depth", 0.7), fb = get (p, "fb", 0.0);
    const double ph = side ? 2 * PI * get (p, "stereo_phase", 0.0) : 0.0, t0 = startT (p);
    Vec d (x.size()); for (size_t i = 0; i < x.size(); ++i) d[i] = (dmin + (dmax - dmin) * 0.5 * (1 + lfoAt (t0 + (double) i / SR, rate, ph, true))) * SR;
    Vec y (x.size());
    if (fb <= 0) { const Vec w = fracDelay (x, d); for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] + depth * w[i]; }
    else
    {
        y = x; for (int it = 0; it < 6; ++it) { const Vec w = fracDelay (y, d); for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] + fb * w[i]; }
        for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] + depth * (y[i] - x[i]);
    }
    for (auto& v : y) v /= (1 + depth);
    return y;
}
inline Vec chorus (const Vec& x, const Params& p, int side)
{
    const double rate = get (p, "rate", 0.8), base = get (p, "base", 15e-3), width = get (p, "width", 4e-3), mix = get (p, "mix", 0.5);
    const int voices = std::max (1, (int) std::lround (get (p, "voices", 2))); const double t0 = startT (p);
    Vec y = x; const double ph0 = side ? PI / 2 : 0.0;
    for (int v = 0; v < voices; ++v)
    {
        Vec d (x.size()); for (size_t i = 0; i < x.size(); ++i) d[i] = std::max (0.0, (base + width * lfoAt (t0 + (double) i / SR, rate * (1 + 0.13 * v), ph0 + v * 2.1, false)) * SR);
        const Vec w = fracDelay (x, d); for (size_t i = 0; i < x.size(); ++i) y[i] += mix * w[i] / voices;
    }
    for (auto& v : y) v /= (1 + mix);
    return y;
}
inline Vec phaser (const Vec& x, const Params& p, int)
{
    const double rate = get (p, "rate", 0.4), fmin = get (p, "fmin", 200), fmax = get (p, "fmax", 3000), fb = get (p, "fb", 0.3), mix = get (p, "mix", 0.5);
    const int stages = std::max (1, (int) std::lround (get (p, "stages", 6))); const double t0 = startT (p);
    const size_t n = x.size(), blk = 64; Vec y (n), z ((size_t) stages, 0.0); double last = 0;
    for (size_t s0 = 0; s0 < n; s0 += blk)
    {
        const double f = fmin * std::pow (fmax / fmin, 0.5 * (1 + std::sin (2 * PI * rate * (t0 + (double) s0 / SR))));
        const double t = std::tan (PI * std::min (f, SR * 0.49) / SR), a = (t - 1) / (t + 1);
        const size_t e = std::min (n, s0 + blk); double fbIn = fb * last;
        for (size_t i = s0; i < e; ++i)
        {
            double v = x[i] + fbIn;
            for (int k = 0; k < stages; ++k) { const double o = a * v + z[(size_t) k]; z[(size_t) k] = v - a * o; v = o; }
            y[i] = v;
        }
        last = y[e - 1];
    }
    for (size_t i = 0; i < n; ++i) y[i] = (1 - mix) * x[i] + mix * y[i];
    return y;
}
inline Vec distortion (const Vec& x, const Params& p, int)
{
    const double drive = get (p, "drive", 8.0); const bool tanhKind = get (p, "kind", 1) > 0.5; const double pk = peakOf (x) + 1e-12;
    Vec y (x.size()); for (size_t i = 0; i < x.size(); ++i) { const double g = x[i] / pk * drive; y[i] = 0.7 * (tanhKind ? std::tanh (g) : std::clamp (g, -1.0, 1.0)); }
    return y;
}
inline Vec bitcrush (const Vec& x, const Params& p, int)
{
    const double bits = std::max (1.0, get (p, "bits", 6)); const int down = std::max (1, (int) std::lround (get (p, "down", 4)));
    const double pk = peakOf (x) + 1e-12, q = std::pow (2.0, bits - 1); Vec y (x.size());
    for (size_t i = 0; i < x.size(); ++i) { const size_t j = i - i % (size_t) down; y[i] = 0.7 * std::nearbyint (x[j] / pk * q) / q; }
    return y;
}
inline Vec filterSweep (const Vec& x, const Params& p, int)
{
    const double rate = get (p, "rate", 0.25), fmin = get (p, "fmin", 300), fmax = get (p, "fmax", 5000), q = std::max (0.3, get (p, "q", 4.0)); const double t0 = startT (p);
    Biquad lp, pk; const size_t n = x.size(), blk = 256; Vec y (n);
    for (size_t s0 = 0; s0 < n; s0 += blk)
    {
        const double fc = fmin * std::pow (fmax / fmin, 0.5 * (1 + std::sin (2 * PI * rate * (t0 + (double) s0 / SR))));
        lp.setLow (fc); pk.setPeak (fc, q);
        for (size_t i = s0; i < std::min (n, s0 + blk); ++i) { const double y1 = lp.tick (x[i]); y[i] = y1 + 0.8 * pk.tick (y1); }
    }
    return y;
}
// reverb: decaying noise tail per channel, level matched to the dry sound
inline Vec reverb (const Vec& x, const Params& p, int side)
{
    const double rt60 = std::clamp (get (p, "rt60", 1.8), 0.05, 12.0), mix = get (p, "mix", 0.35);
    const size_t n = (size_t) (rt60 * 1.2 * SR); std::mt19937 rng (side ? 7u : 1u); std::normal_distribution<double> nd;
    Vec ir (n); double e = 0;
    for (size_t i = 0; i < n; ++i) { ir[i] = i < (size_t) (0.01 * SR) ? 0.0 : nd (rng) * std::pow (10.0, -3.0 * ((double) i / SR) / rt60); e += ir[i] * ir[i]; }
    for (auto& v : ir) v /= std::sqrt (e) + 1e-20;
    const Vec w = convolve (x, ir); const double s = rmsOf (x) / (rmsOf (w) + 1e-12);
    Vec y (x.size()); for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * (1 - mix) + mix * s * w[i];
    return y;
}
// fxlib.delay on one input; returns (L, R) because ping-pong splits the repeats
inline Stereo delayMono (const Vec& x, const Params& p)
{
    const double ms = get (p, "ms", 300), fb = get (p, "fb", 0.4), mix = get (p, "mix", 0.4); const bool pp = get (p, "pingpong", 0) > 0.5;
    const size_t d = (size_t) std::max (1.0, ms * 1e-3 * SR), n = x.size(); Stereo o { x, x }; double g = mix;
    for (int k = 1; k < 8; ++k)
    {
        const size_t off = (size_t) k * d; if (off >= n) break;
        for (size_t i = off; i < n; ++i)
        {
            const double v = g * x[i - off];
            if (pp && (k % 2)) o.r[i] += v; else if (pp) o.l[i] += v; else { o.l[i] += v; o.r[i] += v; }
        }
        g *= fb;
    }
    return o;
}
inline Vec shift (const Vec& x, long d)          // uneffect.shift: d > 0 moves earlier
{
    Vec y (x.size(), 0.0); const long n = (long) x.size();
    for (long i = 0; i < n; ++i) { const long j = i + d; if (j >= 0 && j < n) y[(size_t) i] = x[(size_t) j]; }
    return y;
}
inline Stereo widenerMs (const Vec& x, double amount)
{
    // high-pass 300 Hz (butter 1), crude allpass-ish decorrelator, 7 ms delay
    const double K = std::tan (PI * 300.0 / SR), b0 = 1 / (1 + K), a1 = (K - 1) / (K + 1);
    Vec h (x.size()); double xp = 0, yp = 0;
    for (size_t i = 0; i < x.size(); ++i) { const double v = b0 * (x[i] - xp) - a1 * yp; xp = x[i]; yp = v; h[i] = v; }
    Vec s (x.size());
    for (size_t i = 0; i < x.size(); ++i) s[i] = 0.6 * h[i] + (i >= 3 ? h[i - 3] - 0.6 * s[i - 3] : 0.0);
    s = fracDelay (s, Vec (x.size(), 7e-3 * SR));
    Stereo o; o.l.resize (x.size()); o.r.resize (x.size());
    for (size_t i = 0; i < x.size(); ++i) { o.l[i] = (x[i] + amount * s[i]) / (1 + amount); o.r[i] = (x[i] - amount * s[i]) / (1 + amount); }
    return o;
}
} // namespace fx

// The order the chain is rebuilt in (estimate.ORDER).
inline const std::vector<std::string>& chainOrder()
{
    static const std::vector<std::string> o { "distortion", "bitcrush", "filtersweep", "phaser", "flanger", "chorus", "tremolo", "width", "haas", "delay", "reverb", "autopan" };
    return o;
}
inline bool isLfoFx (const std::string& f) { return f == "flanger" || f == "chorus" || f == "phaser" || f == "filtersweep"; }

// Distortion: pick the drive that squashes the target as much as the source was squashed.
inline double distortionDriveFor (const Vec& m, double top, bool tanhKind)
{
    double best = 4, bd = 1e9; const double pk = fx::peakOf (m) + 1e-12;
    for (int i = 0; i < 25; ++i)
    {
        const double d = 1.5 * std::pow (20.0, i / 24.0); size_t c = 0;
        for (double v : m) { const double g = v / pk * d, y = 0.7 * (tanhKind ? std::tanh (g) : std::clamp (g, -1.0, 1.0)); c += std::abs (y) > 0.9 * 0.7; }
        const double e = std::abs ((double) c / (double) std::max<size_t> (1, m.size()) - top);
        if (e < bd) { bd = e; best = d; }
    }
    return best;
}

// One effect on a stereo sound, fully on (amount handled by the caller).
inline Stereo applyEffect (const std::string& f, const Stereo& y, const Params& e)
{
    Stereo o;
    auto perCh = [&] (Vec (*fn) (const Vec&, const Params&, int)) { o.l = fn (y.l, e, 0); o.r = fn (y.r, e, 1); };
    if (f == "flanger") perCh (fx::flanger);
    else if (f == "chorus") perCh (fx::chorus);
    else if (f == "phaser") perCh (fx::phaser);
    else if (f == "filtersweep") perCh (fx::filterSweep);
    else if (f == "reverb") perCh (fx::reverb);
    else if (f == "bitcrush") perCh (fx::bitcrush);
    else if (f == "distortion")
    {
        Params q = e; q["drive"] = get (e, "top", -1) >= 0 ? distortionDriveFor (y.mid(), get (e, "top", 0), get (e, "kind", 1) > 0.5) * get (e, "drive_scale", 1.0) : get (e, "drive", 8);
        o.l = fx::distortion (y.l, q, 0); o.r = fx::distortion (y.r, q, 1);
    }
    else if (f == "delay")
    {
        if (y.l == y.r) o = fx::delayMono (y.l, e);
        else { o.l = fx::delayMono (y.l, e).l; o.r = fx::delayMono (y.r, e).r; }
    }
    else if (f == "haas")
    {
        const long d = std::lround (get (e, "lag", 0)); o = y;
        if (d < 0) o.r = fx::shift (y.r, d); else o.l = fx::shift (y.l, -d);
    }
    else if (f == "width")
    {
        const Vec M = y.mid(); const Stereo w = fx::widenerMs (M, get (e, "amount", 0.5)); o = w;
        for (size_t i = 0; i < y.size(); ++i) { const double S0 = 0.5 * (y.l[i] - y.r[i]); o.l[i] += S0; o.r[i] -= S0; }
    }
    else if (f == "autopan" || f == "tremolo")
    {
        const double r = get (e, "rate", 1), a = get (e, "amp_db", 0), ph = get (e, "phase", 0); o = y;
        for (size_t i = 0; i < y.size(); ++i)
        {
            const double g = a * std::sin (2 * PI * (r * (double) i / SR + ph));
            if (f == "autopan") { o.l[i] *= std::pow (10.0, g / 40); o.r[i] *= std::pow (10.0, -g / 40); }
            else { const double k = std::pow (10.0, g / 20); o.l[i] *= k; o.r[i] *= k; }
        }
    }
    else o = y;
    return o;
}

// estimate.rebuild step: run the effect, keep the level, then blend by s (0 = off, 1 = as found, 2 = double).
inline Stereo applyStep (const std::string& f, const Stereo& prev, const Params& e, double s)
{
    if (std::abs (s) < 1e-9) return prev;
    Stereo y = applyEffect (f, prev, e);
    const double e0 = prev.rms(), e1 = y.rms(), k = e1 > 1e-12 ? e0 / e1 : 1.0;
    for (size_t i = 0; i < y.size(); ++i) { y.l[i] = prev.l[i] + s * (k * y.l[i] - prev.l[i]); y.r[i] = prev.r[i] + s * (k * y.r[i] - prev.r[i]); }
    return y;
}

} // namespace dr
