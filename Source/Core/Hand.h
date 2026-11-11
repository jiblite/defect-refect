#pragma once
// Hand-made checks that sit next to the neural detector (port of the prototype's detect.py,
// echo.py and stackdata.hand): echoes, stereo (Haas lag, pan movement), clipping and bit depth,
// level wobble. Their 15 numbers feed the second-stage deciders.

#include "Dsp.h"

namespace dr
{
namespace hand
{
constexpr int HOP = 480;

inline Vec envDb (const Vec& x, double ms, int hop = HOP)   // detect.env_db (mode 'nearest')
{
    Vec sq (x.size()); for (size_t i = 0; i < x.size(); ++i) sq[i] = x[i] * x[i];
    const Vec e = uniformFilter (sq, (int) (ms * 1e-3 * SR), 1);
    Vec out; out.reserve (x.size() / (size_t) hop + 1);
    for (size_t i = 0; i < e.size(); i += (size_t) hop) out.push_back (10.0 * std::log10 (e[i] + 1e-12));
    return out;
}
inline std::vector<char> active (const Vec& x)
{
    const Vec L = envDb (x, 20); const double mx = *std::max_element (L.begin(), L.end());
    std::vector<char> a (L.size()); for (size_t i = 0; i < L.size(); ++i) a[i] = L[i] > mx - 35.0; return a;
}

struct Periodic { double rate = 0, r2 = 0, amp = 0; };
inline Periodic periodicity (Vec t, double lo, double hi, double fs = 100.0)
{
    Periodic p; const size_t n = t.size();
    const double m = mean (t); for (auto& v : t) v -= m;
    if (n < (size_t) (3 * fs) || stdev (t) < 1e-9) return p;
    const Vec w = hanningSym (n); const size_t N = nextPow2 (n * 8);
    Vec tw (n); for (size_t i = 0; i < n; ++i) tw[i] = t[i] * w[i];
    const auto F = rfft (tw, N);
    const double flo = std::max (lo, 1.5 * fs / (double) n);
    double best = -1; size_t bk = 0; bool any = false;
    for (size_t k = 0; k < F.size(); ++k)
    {
        const double f = (double) k * fs / (double) N;
        if (f >= flo && f <= hi) { any = true; if (std::abs (F[k]) > best) { best = std::abs (F[k]); bk = k; } }
    }
    if (! any) return p;
    p.rate = (double) bk * fs / (double) N;
    const SineFit s = fitSine (t, p.rate, fs);
    p.r2 = std::max (0.0, s.r2); p.amp = std::hypot (s.a, s.b);
    return p;
}

// Real cepstrum of a whole channel (echo.ceps).
inline Vec ceps (const Vec& x)
{
    const size_t n = nextPow2 (x.size()); const Vec w = hanningSym (x.size());
    Vec xw (x.size()); for (size_t i = 0; i < x.size(); ++i) xw[i] = x[i] * w[i];
    auto X = rfft (xw, n);
    double mx = 0; for (auto& c : X) mx = std::max (mx, std::abs (c));
    for (auto& c : X) c = { std::log (std::abs (c) + 1e-6 * mx), 0.0 };
    return irfft (X, n);
}

// Sample-exact echoes 60-750 ms present in one channel (echo.echo_detect): (score, lag ms).
inline std::pair<double, double> echoDetect (const Stereo& y, double zmin = 40.0, double smin = 8.0)
{
    double bestS = 0.0, bestMs = 0.0;
    for (const Vec* ch : { &y.l, &y.r })
    {
        const Vec c = ceps (*ch); const long lo = (long) (0.06 * SR), hi = (long) (0.75 * SR);
        if ((long) c.size() <= lo) continue;
        Vec seg; for (long i = lo; i < std::min (hi, (long) c.size()); ++i) seg.push_back (std::abs (c[(size_t) i]));
        const double med = median (seg); Vec dev (seg.size()); for (size_t i = 0; i < seg.size(); ++i) dev[i] = std::abs (seg[i] - med);
        const double mad = median (dev) + 1e-12;
        std::vector<size_t> idx (seg.size()); std::iota (idx.begin(), idx.end(), 0);
        std::partial_sort (idx.begin(), idx.begin() + std::min<size_t> (12, idx.size()), idx.end(), [&] (size_t a, size_t b) { return seg[a] > seg[b]; });
        for (size_t q = 0; q < std::min<size_t> (12, idx.size()); ++q)
        {
            const long k = (long) idx[q]; const double z = (seg[(size_t) k] - med) / mad;
            if (z < zmin) break;
            double nb = -1e300; const long S = (long) seg.size();
            for (long j = std::max (0L, k - 8); j < std::max (0L, k - 2); ++j) nb = std::max (nb, seg[(size_t) j]);
            for (long j = k + 3; j < std::min (S, k + 9); ++j) nb = std::max (nb, seg[(size_t) j]);
            if (nb < -1e299) nb = 0.0;
            const double sh = seg[(size_t) k] / (nb + 1e-12);
            if (sh >= smin)
            {
                const double sc = z * std::min (sh, 40.0) / 40.0;
                if (sc > bestS) { bestS = sc; bestMs = (double) (k + lo) / SR * 1000.0; }
                break;
            }
        }
    }
    return { bestS, bestMs };
}

// GCC-PHAT lag between left and right within 0.8-40 ms (samples; > 0 when L is late).
struct Gcc { int lag = 0; double z = 0; };
inline Gcc gccPhat (const Stereo& y, double maxSeconds = 6.0)
{
    Gcc g; const size_t n = std::min (y.size(), (size_t) (SR * maxSeconds)); if (n < 64) return g;
    const size_t N = 2 * n;                       // same size as the prototype (not a power of two)
    std::vector<std::complex<double>> A (N), B (N);
    for (size_t i = 0; i < n; ++i) { A[i] = y.l[i]; B[i] = y.r[i]; }
    fftAny (A); fftAny (B);
    for (size_t k = 0; k < N; ++k) { auto G = A[k] * std::conj (B[k]); A[k] = std::conj (G / (std::abs (G) + 1e-12)); }
    fftAny (A);                                    // inverse via conj(FFT(conj(x))) / N
    Vec cc (N); for (size_t k = 0; k < N; ++k) cc[k] = A[k].real() / (double) N;
    const double sd = stdev (cc);
    double best = -1e300;
    const long lo = (long) (0.0008 * SR), hi = (long) (0.04 * SR);
    for (long lag = -hi; lag <= hi; ++lag)
    {
        if (std::abs (lag) < lo) continue;
        const double v = cc[(size_t) (lag >= 0 ? lag : (long) N + lag)];
        if (v > best) { best = v; g.lag = (int) lag; }
    }
    g.z = best / (sd + 1e-12);
    return g;
}

struct StereoInfo { double sideRatio = 0, corr = 1, haasZ = 0, haasMs = 0, panR2 = 0, panSpan = 0; bool hasSide = false; };
inline StereoInfo stereo (const Stereo& y)
{
    StereoInfo s; const size_t n = y.size();
    double em = 0, es = 0; Vec M (n);
    for (size_t i = 0; i < n; ++i) { const double m = 0.5 * (y.l[i] + y.r[i]), d = 0.5 * (y.l[i] - y.r[i]); em += m * m; es += d * d; M[i] = m; }
    s.sideRatio = std::sqrt (es / (double) n) / (std::sqrt (em / (double) n) + 1e-12);
    const double sl = stdev (y.l), sr = stdev (y.r);
    if (sl > 0 && sr > 0)
    {
        const double ml = mean (y.l), mr = mean (y.r); double c = 0; for (size_t i = 0; i < n; ++i) c += (y.l[i] - ml) * (y.r[i] - mr);
        s.corr = c / (double) n / (sl * sr);
    }
    if (s.sideRatio < 0.02) return s;
    s.hasSide = true;
    const Gcc g = gccPhat (y); s.haasMs = (double) g.lag / SR * 1000.0; s.haasZ = g.z;
    const Vec el = envDb (y.l, 50), er = envDb (y.r, 50); Vec p (el.size()); for (size_t i = 0; i < p.size(); ++i) p[i] = el[i] - er[i];
    const auto act = active (M);
    size_t na = 0; for (char a : act) na += a;
    if (na > 100) { Vec q; for (size_t i = 0; i < p.size() && i < act.size(); ++i) if (act[i]) q.push_back (p[i]); p = q; }
    const Periodic per = periodicity (medianFilter (p, 5), 0.05, 5.0);
    s.panR2 = per.r2;
    s.panSpan = percentile (p, 95) - percentile (p, 5);
    return s;
}

struct NonLinear { double clipFrac = 0, uniqueFrac = 1, hold = 1, crest = 0, kurtosis = 0, bits = 0; };
inline NonLinear nonlinear (const Stereo& y)
{
    NonLinear o; const Vec& m = y.l; const size_t n = m.size(); if (n == 0) return o;
    double pk = 0; for (double v : m) pk = std::max (pk, std::abs (v)); pk += 1e-12;
    size_t near = 0; for (double v : m) near += std::abs (v) > 0.98 * pk;
    o.clipFrac = (double) near / (double) n;
    std::vector<double> q; for (double v : m) if (std::abs (v) > 1e-6) q.push_back (std::round (v / pk * 1e7) / 1e7);
    std::vector<double> u = q; std::sort (u.begin(), u.end()); u.erase (std::unique (u.begin(), u.end()), u.end());
    o.uniqueFrac = (double) u.size() / (double) std::max<size_t> (1, q.size());
    std::vector<long> ch; for (size_t i = 0; i + 1 < n; ++i) if (m[i + 1] - m[i] != 0.0) ch.push_back ((long) i);
    if (ch.size() >= 2) { Vec runs; for (size_t i = 0; i + 1 < ch.size(); ++i) runs.push_back ((double) (ch[i + 1] - ch[i])); o.hold = median (runs); }
    Vec cfs;
    for (size_t s = 0; s + 2400 < n; s += 2400)
    {
        double mx = 0, e = 0; for (size_t i = s; i < s + 2400; ++i) { mx = std::max (mx, std::abs (m[i])); e += m[i] * m[i]; }
        const double rms = std::sqrt (e / 2400.0);
        if (rms > (pk - 1e-12) * 0.05) cfs.push_back (mx / (rms + 1e-12));
    }
    o.crest = cfs.empty() ? 0.0 : median (cfs);
    const double mu = mean (m); double m2 = 0, m4 = 0; for (double v : m) { const double d = v - mu; m2 += d * d; m4 += d * d * d * d; }
    m2 /= (double) n; m4 /= (double) n; o.kurtosis = m2 > 0 ? m4 / (m2 * m2) - 3.0 : -3.0;
    if (o.uniqueFrac < 0.01 && u.size() > 1)
    {
        Vec d; for (size_t i = 0; i + 1 < u.size(); ++i) d.push_back (std::abs (u[i + 1] - u[i]));
        const double step = median (d); if (step > 0) o.bits = std::log2 (2.0 / step);
    }
    return o;
}

struct Tremolo { double rate = 0, r2 = 0, depthDb = 0; };
inline Tremolo tremolo (const Stereo& y)
{
    const Vec m = y.mid(); const Vec L = envDb (m, 10); const auto act = active (m);
    const Vec slow = uniformFilter (L, 100, 1);
    Vec l (L.size()); for (size_t i = 0; i < L.size(); ++i) l[i] = L[i] - slow[i];
    size_t na = 0; for (char a : act) na += a;
    Vec use = l; if (na > 300) { use.clear(); for (size_t i = 0; i < l.size() && i < act.size(); ++i) if (act[i]) use.push_back (l[i]); }
    const Periodic p = periodicity (use, 1.0, 15.0);
    return { p.rate, p.r2, 2.0 * p.amp };
}
} // namespace hand

// The 15 hand numbers in the order the deciders were trained with (stackdata.hand).
inline std::vector<float> handFeatures (const Stereo& y)
{
    using namespace hand;
    const auto [sc, ms] = echoDetect (y, 15.0, 3.0);
    const StereoInfo s = stereo (y); const NonLinear nl = nonlinear (y); const Tremolo tr = tremolo (y);
    return { (float) sc, (float) ms, (float) s.sideRatio, (float) s.corr, (float) s.haasZ, (float) std::abs (s.haasMs), (float) s.panR2, (float) s.panSpan,
             (float) nl.clipFrac, (float) std::log10 (nl.uniqueFrac + 1e-6), (float) nl.hold, (float) nl.crest, (float) nl.kurtosis, (float) tr.r2, (float) tr.depthDb };
}

} // namespace dr
