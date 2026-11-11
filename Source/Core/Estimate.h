#pragma once
// Reading each found effect's settings from the source sound (port of estimate.py, delayfit.py,
// revest.py), the controls that take an effect back out of the source (uneffect.py, "do its best"),
// and the fader layout: per effect an Amount fader plus its most useful settings.

#include "Effects.h"
#include "Hand.h"
#include "Inspector.h"
#include <functional>
#include <future>
#include <set>

namespace dr
{
namespace est
{
// ---------------------------------------------------------------- delay (uneffect.delay_dry + delayfit)
inline void combInv (Vec& x, long D, double g)                // x = y - g x(t-D)
{
    for (long s = D; s < (long) x.size(); ++s) x[(size_t) s] -= g * x[(size_t) (s - D)];
}
inline long bestLag (const Vec& x)
{
    const Vec c = hand::ceps (x); const long a = (long) (0.06 * SR), b = std::min ((long) (0.75 * SR), (long) c.size());
    if (b <= a + 20) return -1;
    Vec seg ((size_t) (b - a)); for (long i = a; i < b; ++i) seg[(size_t) (i - a)] = std::abs (c[(size_t) i]);
    const double med = median (seg); Vec dv (seg.size()); for (size_t i = 0; i < seg.size(); ++i) dv[i] = std::abs (seg[i] - med);
    const double mad = median (dv) + 1e-12;
    std::vector<size_t> idx (seg.size()); std::iota (idx.begin(), idx.end(), 0);
    std::partial_sort (idx.begin(), idx.begin() + 12, idx.end(), [&] (size_t p, size_t q) { return seg[p] > seg[q]; });
    for (int r = 0; r < 12; ++r)
    {
        const long k = (long) idx[(size_t) r]; const double z = (seg[(size_t) k] - med) / mad;
        double nb = 0;
        for (long j = std::max (0L, k - 8); j < std::max (0L, k - 2); ++j) nb = std::max (nb, seg[(size_t) j]);
        for (long j = k + 3; j < std::min ((long) seg.size(), k + 9); ++j) nb = std::max (nb, seg[(size_t) j]);
        if (z > 25 && seg[(size_t) k] / (nb + 1e-12) > 6) return k + a;
    }
    return -1;
}
struct Tap { int ch; double ms, g; };
inline Stereo delayDry (const Stereo& y, std::vector<Tap>& taps, int rounds = 4)
{
    Stereo out = y;
    for (int c = 0; c < 2; ++c)
    {
        Vec& x = c ? out.r : out.l;
        for (int r = 0; r < rounds; ++r)
        {
            const long D = bestLag (x); if (D < 0) break;
            double bestG = 0, be = 1e300;
            for (int i = 0; i < 39; ++i)
            {
                const double g = -0.95 + 1.9 * i / 38.0; Vec t = x; combInv (t, D, g);
                double e = 0; for (size_t j = (size_t) D; j < t.size(); ++j) e += t[j] * t[j];
                if (e < be) { be = e; bestG = g; }
            }
            if (std::abs (bestG) < 0.03) break;
            combInv (x, D, bestG); taps.push_back ({ c, (double) D / SR * 1e3, bestG });
        }
    }
    return out;
}
// IIR filter with taps at multiples of d samples (b, a up to 3 coefficients each).
inline double kEnergy (const double* b, const double* a, const Vec& x, size_t d)
{
    Vec y (x.size()); double e = 0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        double v = b[0] * x[i];
        if (i >= d) v += b[1] * x[i - d] - a[1] * y[i - d];
        if (i >= 2 * d) v += b[2] * x[i - 2 * d] - a[2] * y[i - 2 * d];
        y[i] = v; e += v * v;
    }
    return e / (double) std::max<size_t> (1, x.size());
}
inline Params fitDelay (const Stereo& y, size_t d, double& energy)
{
    double best = 1e300; Params P;
    for (int pp = 0; pp < 2; ++pp)
        for (int fi = 0; fi <= 18; ++fi)
            for (int mi = 0; mi <= 16; ++mi)
            {
                const double fb = 0.05 * fi, mix = 0.05 + 0.05 * mi; double e = 0;
                for (int c = 0; c < 2; ++c)
                {
                    double B[3], A[3];   // model (B, A); inverse filter = A / B
                    if (! pp) { B[0] = 1; B[1] = mix - fb; B[2] = 0; A[0] = 1; A[1] = -fb; A[2] = 0; }
                    else if (c == 1) { B[0] = 1; B[1] = mix; B[2] = -fb * fb; A[0] = 1; A[1] = 0; A[2] = -fb * fb; }
                    else { B[0] = 1; B[1] = 0; B[2] = mix * fb - fb * fb; A[0] = 1; A[1] = 0; A[2] = -fb * fb; }
                    e += kEnergy (A, B, c ? y.r : y.l, d);
                }
                if (e < best) { best = e; P = { { "ms", (double) d * 1000 / SR }, { "fb", fb }, { "mix", mix }, { "pingpong", (double) pp } }; }
            }
    double e0 = 0; for (size_t i = 0; i < y.size(); ++i) e0 += y.l[i] * y.l[i] + y.r[i] * y.r[i];
    energy = best / (e0 / (double) std::max<size_t> (1, y.size()) / 2 + 1e-20);
    return P;
}
inline bool fitDelayAny (const Stereo& y, const std::vector<Tap>& taps, Params& out)
{
    std::set<size_t> cands;
    for (auto& t : taps) for (double f : { 1.0, 0.5, 1.0 / 3.0 }) { const size_t d = (size_t) std::lround (t.ms * f * SR / 1000); if (d >= (size_t) (0.04 * SR)) cands.insert (d); }
    if (cands.empty()) return false;
    // fit on at most 8 s to keep it quick
    const Stereo s = y.slice (0, std::min (y.size(), (size_t) (8 * SR)));
    double bestSane = 1e300, bestAny = 1e300; Params pS, pA; bool sane = false;
    for (size_t d : cands)
    {
        double e; const Params p = fitDelay (s, d, e);
        if (e < bestAny) { bestAny = e; pA = p; }
        if (p.at ("fb") < 0.85 && p.at ("mix") < 0.8 && e < bestSane) { bestSane = e; pS = p; sane = true; }
    }
    out = sane ? pS : pA; return true;
}

// ---------------------------------------------------------------- reverb time from free decays (revest.py)
inline double rt60FromDecays (const Stereo& y, int& count)
{
    const size_t h = (size_t) (0.01 * SR); Vec sq (y.size());
    for (size_t i = 0; i < y.size(); ++i) sq[i] = 0.5 * (y.l[i] * y.l[i] + y.r[i] * y.r[i]);
    const Vec e = uniformFilter (sq, (int) h); Vec L; for (size_t i = 0; i < e.size(); i += h) L.push_back (10 * std::log10 (e[i] + 1e-12));
    const double fs = 100.0; count = 0; if (L.size() < 40) return 0;
    const double top = *std::max_element (L.begin(), L.end()); Vec slopes;
    size_t i = 0; const size_t n = L.size();
    while (i + 30 < n)
    {
        double lm = -1e300; for (size_t j = (i >= 5 ? i - 5 : 0); j < std::min (n, i + 6); ++j) lm = std::max (lm, L[j]);
        if (L[i] > top - 35 && L[i] >= lm)
        {
            const size_t len = std::min (n - i, (size_t) (1.5 * fs));
            auto firstBelow = [&] (double db) -> long { for (size_t k = 0; k < len; ++k) if (L[i + k] < L[i] - db) return (long) k; return -1; };
            const long k15 = firstBelow (15);
            if (k15 > 0)
            {
                const long a = std::max (0L, firstBelow (5)); long b = firstBelow (25); if (b < 0) b = (long) len - 1;
                if (b - a >= 4)
                {
                    double st = 0, sv = 0, stt = 0, stv = 0; const double m = (double) (b - a);
                    for (long k = a; k < b; ++k) { const double t = (double) k / fs, v = L[i + (size_t) k]; st += t; sv += v; stt += t * t; stv += t * v; }
                    slopes.push_back ((m * stv - st * sv) / (m * stt - st * st)); ++count; i += (size_t) b; continue;
                }
            }
        }
        ++i;
    }
    Vec s; for (double v : slopes) if (v < -5) s.push_back (v);
    if (count < 3 || s.size() < 3) return 0;
    return std::clamp (-60.0 / percentile (s, 75), 0.2, 8.0);
}

// ---------------------------------------------------------------- gain lanes (autopan, tremolo) and sine fits
inline Vec envDbFull (const Vec& x, double ms, int mode)
{
    Vec sq (x.size()); for (size_t i = 0; i < x.size(); ++i) sq[i] = x[i] * x[i];
    Vec e = uniformFilter (sq, std::max (1, (int) (ms * 1e-3 * SR)), mode); for (auto& v : e) v = 10 * std::log10 (v + 1e-12); return e;
}
inline Vec panLane (const Stereo& y)        // dB L minus R, decimated by 240 (200 per second)
{
    const Vec a = envDbFull (y.l, 40, 0), b = envDbFull (y.r, 40, 0), m = envDbFull (y.mid(), 40, 0);
    const double mx = *std::max_element (m.begin(), m.end());
    Vec p (a.size()); std::vector<char> ok (a.size());
    for (size_t i = 0; i < a.size(); ++i) { p[i] = a[i] - b[i]; ok[i] = m[i] > mx - 45; }
    // fill quiet parts by holding the nearest active value
    long last = -1; for (size_t i = 0; i < p.size(); ++i) if (ok[i]) { if (last < 0) for (size_t j = 0; j < i; ++j) p[j] = p[i]; else for (size_t j = (size_t) last + 1; j < i; ++j) p[j] = p[(size_t) last] + (p[i] - p[(size_t) last]) * (double) (j - (size_t) last) / (double) (i - (size_t) last); last = (long) i; }
    if (last < 0) std::fill (p.begin(), p.end(), 0.0); else for (size_t j = (size_t) last + 1; j < p.size(); ++j) p[j] = p[(size_t) last];
    Vec dec; for (size_t i = 0; i < p.size(); i += 240) dec.push_back (p[i]);
    const double med = median (dec); for (auto& v : p) v -= med;
    const Vec s = uniformFilter (p, (int) (0.02 * SR));
    Vec out; for (size_t i = 0; i < s.size(); i += 240) out.push_back (std::clamp (s[i], -30.0, 30.0));
    return out;
}
inline Vec tremLane (const Stereo& y)
{
    const Vec e = envDbFull (y.mid(), 8, 0); Vec l; for (size_t i = 0; i < e.size(); i += 240) l.push_back (e[i]);
    const Vec slow = uniformFilter (l, 160, 1); for (size_t i = 0; i < l.size(); ++i) l[i] -= slow[i];
    return l;
}
struct Sine { double rate = 1, amp = 0, phase = 0, r2 = 0; };
inline Sine fitSineGrid (const Vec& lane, double fs, double lo, double hi)
{
    Sine best; best.r2 = -1;
    for (int i = 0; i < 200; ++i)
    {
        const double r = lo * std::pow (hi / lo, i / 199.0); const SineFit f = fitSine (lane, r, fs);
        if (f.r2 > best.r2) { best.r2 = f.r2; best.rate = r; best.amp = std::hypot (f.a, f.b); double ph = std::atan2 (f.b, f.a) / (2 * PI); best.phase = ph - std::floor (ph); }
    }
    return best;
}

// ---------------------------------------------------------------- short-time spectrum (for width and moving-filter removal)
struct Spec { size_t n = 2048, hop = 512, frames = 0, len = 0; std::vector<std::vector<std::complex<double>>> L, R; };
inline Spec stft (const Stereo& y, size_t n = 2048, size_t hop = 512)
{
    Spec S; S.n = n; S.hop = hop; S.len = y.size(); const Vec w = hannPeriodic (n);
    const size_t pad = n / 2, total = y.size() + 2 * pad; S.frames = total >= n ? 1 + (total - n + hop - 1) / hop : 1;
    S.L.resize (S.frames); S.R.resize (S.frames);
    for (size_t f = 0; f < S.frames; ++f)
    {
        Vec a (n, 0.0), b (n, 0.0);
        for (size_t i = 0; i < n; ++i) { const long j = (long) (f * hop + i) - (long) pad; if (j >= 0 && j < (long) y.size()) { a[i] = y.l[(size_t) j] * w[i]; b[i] = y.r[(size_t) j] * w[i]; } }
        S.L[f] = rfft (a, n); S.R[f] = rfft (b, n);
    }
    return S;
}
inline Stereo istft (const Spec& S)
{
    const Vec w = hannPeriodic (S.n); const size_t pad = S.n / 2;
    Stereo o; o.l.assign (S.len, 0.0); o.r.assign (S.len, 0.0); Vec ws (S.len, 0.0);
    for (size_t f = 0; f < S.frames; ++f)
    {
        const Vec a = irfft (S.L[f], S.n), b = irfft (S.R[f], S.n);
        for (size_t i = 0; i < S.n; ++i) { const long j = (long) (f * S.hop + i) - (long) pad; if (j >= 0 && j < (long) S.len) { o.l[(size_t) j] += a[i] * w[i]; o.r[(size_t) j] += b[i] * w[i]; ws[(size_t) j] += w[i] * w[i]; } }
    }
    for (size_t i = 0; i < S.len; ++i) if (ws[i] > 1e-9) { o.l[i] /= ws[i]; o.r[i] /= ws[i]; }
    return o;
}
inline Stereo widthControl (const Stereo& y, double s)
{
    Spec S = stft (y); Vec a (S.frames);
    for (size_t f = 0; f < S.frames; ++f)
    {
        double num = 0, den = 0;
        for (size_t k = 0; k < S.L[f].size(); ++k) { const auto M = 0.5 * (S.L[f][k] + S.R[f][k]), Sd = 0.5 * (S.L[f][k] - S.R[f][k]); num += (Sd * std::conj (M)).real(); den += std::norm (M); }
        a[f] = num / (den + 1e-12);
    }
    a = uniformFilter (a, 5);
    for (size_t f = 0; f < S.frames; ++f)
        for (size_t k = 0; k < S.L[f].size(); ++k)
        {
            const auto M = 0.5 * (S.L[f][k] + S.R[f][k]), Sd = 0.5 * (S.L[f][k] - S.R[f][k]);
            const auto Sc = a[f] * M, S2 = Sc + s * (Sd - Sc);
            S.L[f][k] = M + S2; S.R[f][k] = M - S2;
        }
    return istft (S);
}
// Moving spectral shape (uneffect.motion_control): s = 0 flattens the movement, 2 doubles it.
inline Stereo motionControl (const Stereo& y, double s, double rate)
{
    Spec S = stft (y); const size_t nb = S.n / 2 + 1, T = S.frames; const int perOct = 12;
    std::vector<double> edges; for (int i = 0; ; ++i) { const double e = 60 * std::pow (2.0, (double) i / perOct); if (e > 18000) break; edges.push_back (e); }
    const size_t B = edges.size() - 1; std::vector<std::vector<size_t>> sel (B);
    for (size_t b = 0; b < B; ++b)
    {
        for (size_t k = 0; k < nb; ++k) { const double f = (double) k * SR / (double) S.n; if (f >= edges[b] && f < edges[b + 1]) sel[b].push_back (k); }
        if (sel[b].empty()) sel[b].push_back ((size_t) std::lround (std::sqrt (edges[b] * edges[b + 1]) * (double) S.n / SR));
    }
    std::vector<Vec> P (T, Vec (nb)); Vec tot (T, 0.0);
    for (size_t t = 0; t < T; ++t) for (size_t k = 0; k < nb; ++k) { P[t][k] = 0.5 * (std::norm (S.L[t][k]) + std::norm (S.R[t][k])); tot[t] += P[t][k]; }
    double mx = -1e300; for (auto& v : tot) { v = 10 * std::log10 (v + 1e-12); mx = std::max (mx, v); }
    std::vector<Vec> dev (B, Vec (T));
    for (size_t b = 0; b < B; ++b)
    {
        Vec act;
        for (size_t t = 0; t < T; ++t) { double e = 0; for (size_t k : sel[b]) e += P[t][k]; dev[b][t] = 10 * std::log10 (e / (double) sel[b].size() + 1e-12) - tot[t]; if (tot[t] > mx - 45) act.push_back (dev[b][t]); }
        const double med = act.empty() ? 0 : median (act); for (auto& v : dev[b]) v -= med;
        if (rate > 0)   // zero-phase low-pass at 3.5x the effect's speed
        {
            fx::Biquad q;   // the lane runs at SR/hop frames per second: design the low-pass for that rate
            const double fsL = SR / (double) S.hop, fc = std::min (0.95 * fsL / 2, 3.5 * rate), K = std::tan (PI * fc / fsL), s2 = std::sqrt (2.0), nrm = 1 / (1 + s2 * K + K * K);
            q.b0 = K * K * nrm; q.b1 = 2 * q.b0; q.b2 = q.b0; q.a1 = 2 * (K * K - 1) * nrm; q.a2 = (1 - s2 * K + K * K) * nrm;
            fx::Biquad q2 = q; for (auto& v : dev[b]) v = q.tick (v);
            for (size_t t = T; t-- > 0;) dev[b][t] = q2.tick (dev[b][t]);
        }
    }
    for (size_t t = 0; t < T; ++t)
    {
        Vec g (B); for (size_t b = 0; b < B; ++b) { const double sm = (dev[b > 0 ? b - 1 : b][t] + dev[b][t] + dev[b + 1 < B ? b + 1 : b][t]) / 3.0; g[b] = tot[t] > mx - 45 ? std::clamp ((s - 1) * sm, -24.0, 15.0) : 0.0; }
        Vec G (nb, 0.0); double e0 = 0, e1 = 0;
        for (size_t k = 0; k < nb; ++k)
        {
            const double f = std::max (1.0, (double) k * SR / (double) S.n), lf = std::log (f);
            // interpolate over band centres
            size_t b = 0; while (b + 1 < B && std::log (std::sqrt (edges[b + 1] * edges[b + 2 < edges.size() ? b + 2 : b + 1])) < lf) ++b;
            const double c0 = std::log (std::sqrt (edges[b] * edges[b + 1]));
            if (b + 1 < B) { const double c1 = std::log (std::sqrt (edges[b + 1] * edges[b + 2])); const double u = std::clamp ((lf - c0) / (c1 - c0), 0.0, 1.0); G[k] = g[b] + u * (g[b + 1] - g[b]); }
            else G[k] = g[b];
            if (lf < c0 && b == 0) G[k] = g[0];
            e0 += P[t][k]; e1 += P[t][k] * std::pow (10.0, G[k] / 10);
        }
        const double fix = 10 * std::log10 (std::clamp (e0 / (e1 + 1e-20), 1.0 / 16, 16.0));
        for (size_t k = 0; k < nb; ++k) { const double m = std::pow (10.0, (G[k] + fix) / 20); S.L[t][k] *= m; S.R[t][k] *= m; }
    }
    return istft (S);
}

// ---------------------------------------------------------------- sweep fit (estimate.fit_lfo): speed, start phase and shape
// The source's tone-over-time picture is matched against pink noise run through the effect.
struct ToneLane { std::vector<Vec> L; std::vector<char> act; };
inline ToneLane toneLane (const Vec& m, int perOct)
{
    const size_t n = 4096, hop = 1024, pad = n / 2, total = m.size() + 2 * pad, T = total >= n ? 1 + (total - n + hop - 1) / hop : 1;
    static const Vec w = hannPeriodic (4096);
    std::vector<double> e; for (int i = 0; ; ++i) { const double v = 60 * std::pow (2.0, (double) i / perOct); if (i > 0 && v > 16000 * 1.0000001) break; e.push_back (v); }
    // numpy: arange(0, int(log2(16000/60)*perOct)+1)
    e.resize ((size_t) (std::log2 (16000.0 / 60.0) * perOct) + 1);
    for (size_t i = 0; i < e.size(); ++i) e[i] = 60 * std::pow (2.0, (double) i / perOct);
    const size_t B = e.size() - 1; std::vector<std::vector<size_t>> sel (B);
    for (size_t b = 0; b < B; ++b)
    {
        for (size_t k = 0; k <= n / 2; ++k) { const double f = (double) k * SR / (double) n; if (f >= e[b] && f < e[b + 1]) sel[b].push_back (k); }
        if (sel[b].empty()) sel[b].push_back ((size_t) std::lround (e[b] * (double) n / SR));
    }
    ToneLane o; o.L.assign (B, Vec (T)); o.act.resize (T); Vec tot (T);
    Vec re (n), im (n);
    for (size_t t = 0; t < T; ++t)
    {
        for (size_t i = 0; i < n; ++i) { const long j = (long) (t * hop + i) - (long) pad; re[i] = (j >= 0 && j < (long) m.size()) ? m[(size_t) j] * w[i] : 0.0; im[i] = 0; }
        fft (re, im); double sum = 0; Vec P (n / 2 + 1);
        for (size_t k = 0; k <= n / 2; ++k) { P[k] = re[k] * re[k] + im[k] * im[k]; sum += P[k]; }
        tot[t] = 10 * std::log10 (sum + 1e-12);
        for (size_t b = 0; b < B; ++b) { double a = 0; for (size_t k : sel[b]) a += P[k]; o.L[b][t] = 10 * std::log10 (a / (double) sel[b].size() + 1e-12) - tot[t]; }
    }
    const double mx = *std::max_element (tot.begin(), tot.end());
    for (size_t t = 0; t < T; ++t) o.act[t] = tot[t] > mx - 40;
    for (auto& row : o.L)
    {
        Vec a; for (size_t t = 0; t < T; ++t) if (o.act[t]) a.push_back (row[t]);
        const double med = a.empty() ? 0 : median (a);
        for (size_t t = 0; t < T; ++t) row[t] = o.act[t] ? row[t] - med : 0.0;
    }
    return o;
}
inline double laneCorr (const ToneLane& A, const ToneLane& P)
{
    const size_t T = std::min (A.act.size(), P.act.size()); double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 0;
    for (size_t b = 0; b < A.L.size(); ++b) for (size_t t = 0; t < T; ++t) if (A.act[t])
    { const double x = A.L[b][t], y = P.L[b][t]; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; n += 1; }
    if (n < 2) return -1;
    const double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
    return va > 0 && vb > 0 ? cov / std::sqrt (va * vb) : -1;
}
inline const Vec& pinkProbe (size_t n)
{
    static Vec p;
    if (p.size() < n)
    {
        const size_t N = nextPow2 (n); std::mt19937 rng (5); std::normal_distribution<double> nd; Vec x (N); for (auto& v : x) v = nd (rng);
        auto X = rfft (x, N); for (size_t k = 0; k < X.size(); ++k) X[k] /= std::sqrt (std::max ((double) k * SR / (double) N, 20.0));
        p = irfft (X, N); double pk = 0; for (double v : p) pk = std::max (pk, std::abs (v)); for (auto& v : p) v /= pk * 2;
    }
    return p;
}
inline Vec runMono (const std::string& f, const Vec& x, const Params& p)
{
    if (f == "flanger") return fx::flanger (x, p, 0);
    if (f == "phaser") return fx::phaser (x, p, 0);
    if (f == "filtersweep") return fx::filterSweep (x, p, 0);
    const Vec a = fx::chorus (x, p, 0), b = fx::chorus (x, p, 1); Vec m (a.size()); for (size_t i = 0; i < m.size(); ++i) m[i] = 0.5 * (a[i] + b[i]); return m;
}
inline std::vector<Params> lfoGrid (const std::string& f)
{
    std::vector<Params> g;
    if (f == "flanger") for (double a : { 0.3e-3, 1e-3, 2e-3 }) for (double b : { 3e-3, 5e-3, 8e-3 }) g.push_back ({ { "dmin", a }, { "dmax", b }, { "depth", 0.7 } });
    if (f == "phaser") for (double s : { 4.0, 8.0 }) for (double a : { 150.0, 400.0 }) for (double b : { 2000.0, 5000.0 }) g.push_back ({ { "stages", s }, { "fmin", a }, { "fmax", b }, { "fb", 0.4 }, { "mix", 0.5 } });
    if (f == "filtersweep") for (double a : { 200.0, 500.0 }) for (double b : { 2000.0, 6000.0 }) for (double q : { 2.0, 6.0 }) g.push_back ({ { "fmin", a }, { "fmax", b }, { "q", q } });
    if (f == "chorus") for (double b : { 10e-3, 20e-3 }) for (double w : { 2e-3, 6e-3 }) g.push_back ({ { "base", b }, { "width", w }, { "voices", 2 }, { "mix", 0.5 } });
    return g;
}
inline int lfoPerOct (const std::string& f) { return f == "phaser" ? 12 : f == "filtersweep" ? 6 : 24; }
inline Params fitLfo (const Stereo& y, const std::string& f, double rate0, const Params* shape, double& match, double secs = 6.0)
{
    const size_t n = std::min (y.size(), (size_t) (secs * SR)); const int po = lfoPerOct (f);
    Vec m (n); for (size_t i = 0; i < n; ++i) m[i] = 0.5 * (y.l[i] + y.r[i]);
    const ToneLane LD = toneLane (m, po); const Vec pr (pinkProbe (n).begin(), pinkProbe (n).begin() + (long) n);
    auto score = [&] (const Params& q) { return laneCorr (LD, toneLane (runMono (f, pr, q), po)); };
    auto with = [] (Params g, double r, double ph) { g["rate"] = r; g["phase"] = ph; return g; };
    const auto grid = lfoGrid (f); std::vector<Params> shapes; if (shape) shapes.push_back (*shape); else shapes.assign (grid.begin(), grid.begin() + 4);
    double best = -9; Params cur;
    for (double k : { 0.6, 0.75, 0.87, 1.0, 1.15, 1.33, 1.6 })
        for (auto& g : shapes)
            for (int i = 0; i < 8; ++i) { const Params q = with (g, rate0 * k, i / 8.0); const double c = score (q); if (c > best) { best = c; cur = q; } }
    if (! shape) for (auto& g : grid) { const Params q = with (g, cur["rate"], cur["phase"]); const double c = score (q); if (c > best) { best = c; cur = q; } }
    for (auto [dr_, dp] : { std::pair<double, double> { 0.04, 0.06 }, { 0.02, 0.03 }, { 0.01, 0.015 }, { 0.005, 0.008 } })
    {
        const Params base = cur;
        for (int v = 0; v < 4; ++v)
        {
            Params q = base;
            if (v == 0) q["rate"] *= 1 + dr_; else if (v == 1) q["rate"] *= 1 - dr_;
            else { double ph = q["phase"] + (v == 2 ? dp : -dp); q["phase"] = ph - std::floor (ph); }
            const double c = score (q); if (c > best) { best = c; cur = q; }
        }
    }
    if (f == "filtersweep" && ! shape) for (double qq : { 1.5, 3.0, 5.0, 8.0 }) { Params q = cur; q["q"] = qq; const double c = score (q); if (c > best) { best = c; cur = q; } }
    match = best; return cur;
}
} // namespace est

// ---------------------------------------------------------------- what was found, and its faders
struct Knob
{
    std::string id, label, tip, key;   // key: the setting this fader scales ("" = the effect's Amount)
    double neutral = 0, lo = 0, hi = 1e9;
};
struct FoundFx
{
    std::string name, title; double prob = 0;
    Params est;                        // settings as read from the source
    std::vector<Knob> knobs;
    double lfoRate = 0;                // for removal of moving filters
};

inline std::string titleFor (const std::string& f)
{
    static const std::map<std::string, std::string> t { { "flanger", "FLANGER" }, { "chorus", "CHORUS" }, { "phaser", "PHASER" }, { "haas", "HAAS WIDTH" },
        { "width", "STEREO WIDTH" }, { "delay", "DELAY" }, { "reverb", "REVERB" }, { "distortion", "DISTORTION" }, { "bitcrush", "BITCRUSH" },
        { "filtersweep", "FILTER SWEEP" }, { "tremolo", "TREMOLO" }, { "autopan", "AUTO PAN" } };
    auto it = t.find (f); return it == t.end() ? f : it->second;
}

inline std::vector<Knob> knobsFor (const std::string& f)
{
    auto K = [] (std::string key, std::string label, std::string tip, double neutral = 0, double lo = 0, double hi = 1e9) { return Knob { "", std::move (label), std::move (tip), std::move (key), neutral, lo, hi }; };
    std::vector<Knob> k { K ("", "Amount", "How much of the effect. Middle = as found. Left = none. Right = double.") };
    if (f == "flanger") { k.push_back (K ("rate", "Speed", "Sweep speed, scaled from the speed found.", 0, 0.01, 10)); k.push_back (K ("dmax", "Sweep", "How far the sweep travels.", 0, 0.2e-3, 20e-3)); k.push_back (K ("depth", "Intensity", "Strength of the comb.", 0, 0, 1)); }
    else if (f == "chorus") { k.push_back (K ("rate", "Speed", "Wobble speed.", 0, 0.01, 10)); k.push_back (K ("base", "Delay", "Distance of the copies.", 0, 1e-3, 60e-3)); k.push_back (K ("width", "Wobble", "How far the copies drift.", 0, 0, 30e-3)); k.push_back (K ("mix", "Mix", "Level of the copies.", 0, 0, 1)); }
    else if (f == "phaser") { k.push_back (K ("rate", "Speed", "Sweep speed.", 0, 0.01, 10)); k.push_back (K ("fmin", "Low", "Bottom of the sweep.", 0, 30, 8000)); k.push_back (K ("fmax", "High", "Top of the sweep.", 0, 200, 20000)); k.push_back (K ("fb", "Feedback", "Sharpness of the notches.", 0, 0, 0.95)); k.push_back (K ("mix", "Mix", "Level of the swept part.", 0, 0, 1)); }
    else if (f == "filtersweep") { k.push_back (K ("rate", "Speed", "Sweep speed.", 0, 0.01, 10)); k.push_back (K ("fmin", "Low", "Bottom of the sweep.", 0, 30, 8000)); k.push_back (K ("fmax", "High", "Top of the sweep.", 0, 200, 20000)); k.push_back (K ("q", "Resonance", "Peak at the cutoff.", 0, 0.5, 20)); }
    else if (f == "width") k.push_back (K ("amount", "Width", "Added stereo spread.", 0, 0, 2));
    else if (f == "delay") { k.push_back (K ("ms", "Time", "Echo time.", 0, 5, 2000)); k.push_back (K ("fb", "Feedback", "How long the echoes go on.", 0, 0, 0.9)); k.push_back (K ("mix", "Mix", "Level of the echoes.", 0, 0, 1.2)); }
    else if (f == "reverb") { k.push_back (K ("rt60", "Length", "Decay time.", 0, 0.1, 10)); k.push_back (K ("mix", "Mix", "Level of the room.", 0, 0, 1)); }
    else if (f == "distortion") k.push_back (K ("drive_scale", "Drive", "How hard it is pushed.", 0, 0.05, 4));
    else if (f == "bitcrush") { k.push_back (K ("bits", "Bit Depth", "Fewer levels to the right.", 16, 1, 16)); k.push_back (K ("down", "Sample Rate", "Coarser steps to the right.", 1, 1, 32)); }
    else if (f == "tremolo" || f == "autopan") { k.push_back (K ("rate", "Speed", "Wobble speed.", 0, 0.02, 20)); k.push_back (K ("amp_db", "Depth", "How deep the movement goes.", 0, 0, 40)); }
    for (auto& x : k) x.id = f + "." + (x.key.empty() ? std::string ("amount") : x.key);
    return k;
}

// Settings of each found effect (estimate.estimate, without the slower sweep fit for now).
inline std::vector<FoundFx> estimate (const Inspection& I, const Stereo& y, double cut = 0.65)
{
    std::vector<FoundFx> out;
    (void) est::pinkProbe ((size_t) (6 * SR));      // built once before the fits run in parallel
    auto S = [&] (const std::string& f, const char* k, double d) { auto a = I.settings.find (f); if (a == I.settings.end()) return d; auto b = a->second.find (k); return b == a->second.end() ? d : b->second; };
    auto L = [&] (const std::string& f, const char* k, const std::string& d) { auto a = I.labels.find (f); if (a == I.labels.end()) return d; auto b = a->second.find (k); return b == a->second.end() ? d : b->second; };
    auto R = [&] (const std::string& f, double d) { auto a = I.reg.find (f); return a == I.reg.end() ? d : a->second; };
    for (const auto& f : chainOrder())
    {
        auto pit = I.prob.find (f); if (pit == I.prob.end() || pit->second <= cut) continue;
        FoundFx F; F.name = f; F.title = titleFor (f); F.prob = pit->second; F.knobs = knobsFor (f); Params& e = F.est;
        if (f == "delay")
        {
            std::vector<est::Tap> taps; est::delayDry (y.slice (0, std::min (y.size(), (size_t) (12 * SR))), taps);
            std::vector<double> ms; for (auto& t : taps) ms.push_back (t.ms);
            if (! est::fitDelayAny (y, taps, e))
                e = { { "ms", S (f, "ms", 300) }, { "fb", std::clamp (S (f, "fb", 0.4), 0.0, 0.85) }, { "mix", std::clamp (S (f, "mix", 0.4), 0.05, 0.8) }, { "pingpong", L (f, "pingpong", "False") == "True" ? 1.0 : 0.0 } };
        }
        else if (f == "haas") e["lag"] = (double) hand::gccPhat (y, 30.0).lag;
        else if (f == "width") e["amount"] = std::clamp (S (f, "amount", 0.5), 0.05, 1.2);
        else if (f == "autopan") { const auto s = est::fitSineGrid (est::panLane (y), 200, 0.1, 4); e = { { "rate", s.rate }, { "amp_db", s.amp }, { "phase", s.phase } }; }
        else if (f == "tremolo") { const double r0 = std::clamp (R (f, 5), 0.3, 20.0); const auto s = est::fitSineGrid (est::tremLane (y), 200, r0 / 1.5, r0 * 1.5); e = { { "rate", s.rate }, { "amp_db", s.amp }, { "phase", s.phase } }; }
        else if (f == "bitcrush") { const auto nl = hand::nonlinear (y); e = { { "bits", nl.bits > 0 ? std::round (nl.bits) : 8.0 }, { "down", std::max (1.0, std::round (nl.hold)) } }; }
        else if (f == "distortion")
        {
            const Vec m = y.mid(); const double pk = fx::peakOf (m); size_t c = 0; for (double v : m) c += std::abs (v) > 0.9 * pk;
            e = { { "top", (double) c / (double) std::max<size_t> (1, m.size()) }, { "kind", L (f, "kind", "tanh") == "tanh" ? 1.0 : 0.0 }, { "drive", S (f, "drive", 8) }, { "drive_scale", 1.0 } };
        }
        else if (f == "reverb") { int n = 0; const double t = est::rt60FromDecays (y, n); e = { { "rt60", std::clamp (t > 0 ? t : S (f, "rt60", 1.5), 0.3, 5.0) }, { "mix", std::clamp (S (f, "mix", 0.35), 0.05, 0.8) } }; }
        else   // flanger, chorus, phaser, filtersweep: shape and speed from the settings reader
        {
            const double rate = std::clamp (S (f, "rate", R (f, 0.5)), 0.05, 4.0);
            e["rate"] = rate; e["phase"] = 0;
            if (f == "flanger") { e["dmin"] = S (f, "dmin", 0.5e-3); e["dmax"] = S (f, "dmax", 5e-3); e["depth"] = std::clamp (S (f, "depth", 0.7), 0.1, 1.0); if (e["dmin"] >= e["dmax"]) e["dmin"] = e["dmax"] * 0.3; }
            if (f == "chorus") { e["base"] = S (f, "base", 15e-3); e["width"] = S (f, "width", 4e-3); e["voices"] = std::stod (L (f, "voices", "2")); e["mix"] = std::clamp (S (f, "mix", 0.5), 0.1, 1.0); }
            if (f == "phaser") { e["stages"] = std::stod (L (f, "stages", "6")); e["fmin"] = S (f, "fmin", 200); e["fmax"] = S (f, "fmax", 3000); e["fb"] = std::clamp (S (f, "fb", 0.3), 0.0, 0.9); e["mix"] = std::clamp (S (f, "mix", 0.5), 0.1, 1.0); if (e["fmin"] >= e["fmax"]) e["fmax"] = e["fmin"] * 4; }
            if (f == "filtersweep") { e["fmin"] = S (f, "fmin", 300); e["fmax"] = S (f, "fmax", 5000); e["q"] = S (f, "q", 4); if (e["fmin"] >= e["fmax"]) e["fmax"] = e["fmin"] * 4; }
            // speed and start phase: match the source's movement (two speed guesses, reader's shape or a grid; keep the best)
            Params shape = e; shape.erase ("rate"); shape.erase ("phase");
            double bestM = -9; Params bestP = e;
            std::set<double> r0s { std::round (std::clamp (R (f, rate), 0.08, 3.0) * 1000) / 1000, std::round (std::clamp (rate, 0.08, 3.0) * 1000) / 1000 };
            std::vector<std::future<std::pair<double, Params>>> jobs;     // the fits run side by side
            for (double r0 : r0s) for (int useShape = 1; useShape >= 0; --useShape)
                jobs.push_back (std::async (std::launch::async, [&y, &f, &shape, r0, useShape] { double mt; Params q = est::fitLfo (y, f, r0, useShape ? &shape : nullptr, mt); return std::make_pair (mt, q); }));
            for (auto& j : jobs) { auto [mt, q] = j.get(); if (mt > bestM) { bestM = mt; bestP = q; } }
            e = bestP; e["match"] = bestM;
            F.lfoRate = e["rate"];
        }
        out.push_back (std::move (F));
    }
    return out;
}

// The settings with each setting fader applied (fader value v in 0..2, 1 = as found).
inline Params scaled (const FoundFx& F, const std::map<std::string, double>& faders)
{
    Params p = F.est;
    for (auto& k : F.knobs)
    {
        if (k.key.empty()) continue;
        auto it = faders.find (k.id); const double v = it == faders.end() ? 1.0 : it->second;
        if (std::abs (v - 1.0) < 1e-9) continue;
        const double base = get (p, k.key.c_str(), k.neutral);
        double nv = k.neutral + v * (base - k.neutral);
        p[k.key] = std::clamp (nv, k.lo, k.hi);
    }
    if (F.name == "flanger" && p["dmin"] >= p["dmax"]) p["dmin"] = p["dmax"] * 0.5;
    if ((F.name == "phaser" || F.name == "filtersweep") && p["fmin"] >= p["fmax"] * 0.9) p["fmin"] = p["fmax"] * 0.9;
    return p;
}
inline double amountOf (const FoundFx& F, const std::map<std::string, double>& faders)
{
    auto it = faders.find (F.name + ".amount"); return it == faders.end() ? 1.0 : it->second;
}
inline bool settingsMoved (const FoundFx& F, const std::map<std::string, double>& faders)
{
    for (auto& k : F.knobs) if (! k.key.empty()) { auto it = faders.find (k.id); if (it != faders.end() && std::abs (it->second - 1.0) > 1e-9) return true; }
    return false;
}

// Two sounds: put the source's effects on the target. Left = not applied, right = double.
inline Stereo applyTo (const Stereo& target, const std::vector<FoundFx>& found, const std::map<std::string, double>& faders, const std::function<bool()>& cancelled = {})
{
    Stereo y = target;
    for (auto& F : found)
    {
        if (cancelled && cancelled()) break;
        y = applyStep (F.name, y, scaled (F, faders), amountOf (F, faders));
    }
    return y;
}

// One sound: take the found effects out (left), leave as is (middle) or double (right).
// Settings faders move the effect's settings by adding the difference they make.
inline bool canRemove (const std::string& f) { return f != "reverb" && f != "distortion" && f != "bitcrush"; }
inline Stereo reshape (const Stereo& src, const std::vector<FoundFx>& found, const std::map<std::string, double>& faders, const std::function<bool()>& cancelled = {})
{
    Stereo y = src;
    for (auto it = found.rbegin(); it != found.rend(); ++it)      // undo in reverse chain order
    {
        if (cancelled && cancelled()) break;
        const FoundFx& F = *it; const double s = amountOf (F, faders); const Stereo before = y;
        if (settingsMoved (F, faders))
        {
            const Stereo a = applyStep (F.name, before, F.est, 1.0), b = applyStep (F.name, before, scaled (F, faders), 1.0);
            for (size_t i = 0; i < y.size(); ++i) { y.l[i] += b.l[i] - a.l[i]; y.r[i] += b.r[i] - a.r[i]; }
        }
        if (std::abs (s - 1.0) < 1e-9) continue;
        if (s > 1.0) { y = applyStep (F.name, y, scaled (F, faders), s - 1.0); continue; }
        // s < 1: take it out as far as this effect allows
        const std::string& f = F.name;
        if (f == "delay")
        {
            std::vector<est::Tap> taps; const Stereo dry = est::delayDry (y, taps);
            for (size_t i = 0; i < y.size(); ++i) { y.l[i] = dry.l[i] + s * (y.l[i] - dry.l[i]); y.r[i] = dry.r[i] + s * (y.r[i] - dry.r[i]); }
        }
        else if (f == "haas")
        {
            const long d = std::lround (get (F.est, "lag", 0)); const long D = std::labs (d), sh = std::lround ((double) D * (1 - s));
            if (d < 0) y.r = fx::shift (y.r, sh); else y.l = fx::shift (y.l, sh);
        }
        else if (f == "width") y = est::widthControl (y, s);
        else if (f == "autopan" || f == "tremolo")
        {
            Params inv = F.est; inv["amp_db"] = -(1 - s) * get (F.est, "amp_db", 0);
            y = applyEffect (f, y, inv);
        }
        else if (isLfoFx (f)) y = est::motionControl (y, s, F.lfoRate);
        // reverb, distortion, bitcrush: not removable yet (left half leaves them as found)
    }
    return y;
}

} // namespace dr
