#pragma once
// Reading each found effect's settings from the source sound (port of estimate.py, revest.py), the controls
// that take an effect back out of the source (uneffect.py, "do its best"), and the fader layout: per effect an
// Amount fader plus its most useful settings. Reverb, comb filter, tone match and delay are the work of their
// own engines (RoomEngine.h, Tone.h, Echo.h); their groups are built here, and so are those of the engines that came
// after them (one header each, from Level.h to Bend.h).

#include "Echo.h"
#include "Level.h"
#include "Sweep.h"
#include "Moving.h"
#include "Chorus.h"
#include "Sides.h"
#include "Ring.h"
#include "Crush.h"
#include "Stutter.h"
#include "Movement.h"
#include "Filter.h"
#include "Bend.h"
#include "Effects.h"
#include "Hand.h"
#include "Inspector.h"
#include "RoomEngine.h"
#include "Tone.h"
#include "Verify.h"
#include <functional>
#include <future>
#include <memory>
#include <set>

namespace dr
{
namespace est
{
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
    bool ownTip = false;               // the tip says what left and right do (otherwise: left = none, right = double)
    std::string tip2;                  // the tip while there is a second sound, where it differs ("" = the same words)
};
struct FoundFx
{
    std::string name, title; double prob = 0;
    Params est;                        // settings as read from the source
    std::vector<Knob> knobs;
    double lfoRate = 0;                // for removal of moving filters
    std::shared_ptr<const room::Found> room;   // a reverb the room engine found: its faders work that engine (see roomFx)
    std::shared_ptr<const tone::Found> tone;   // a group of the tone engine (COMB FILTER, TONE MATCH): what it found on the first sound
    std::shared_ptr<const echo::Found> echo;   // the echoes the echo engine found: its faders work that engine (see echoFx)
    std::shared_ptr<const level::Movement> level;   // the level movement the level engine found (TREMOLO, AUTO PAN, GATED RHYTHM): its faders work that engine (see levelFx)
    std::shared_ptr<const sweep::Result> sweep;     // the moving copy the flanger engine found: its faders work that engine (see flangerFx)
    std::shared_ptr<const moving::Found> moving;    // the moving tone curve the moving-tone engine found (PHASER, FILTER SWEEP): its faders work that engine (see movingFx)
    std::shared_ptr<const chorus::Found> chorus;    // the voices the chorus engine found (CHORUS): its faders work that engine (see chorusFx)
    std::shared_ptr<const sides::Found> sides;      // what the sides engine found between the two sides (HAAS WIDTH, STEREO WIDTH): its fader works that engine (see sidesFx)
    std::shared_ptr<const ring::Found> ring;        // the tone the ring engine found the sound multiplied by (RING MOD): its faders work that engine (see ringFx)
    std::shared_ptr<const crush::Found> crush;      // the grid of values and the hold the crush engine found (BITCRUSH): its faders work that engine (see crushFx)
    std::shared_ptr<const stutter::Found> stutter;  // the slices the stutter engine found played again (STUTTER): its faders work that engine (see stutterFx)
    std::shared_ptr<const fixed::Found> filter;     // the curve the filter engine found (FILTER): its faders work that engine (see filterFx)
    std::shared_ptr<const bend::Found> bend;        // the curve the bend engine found the sound's values pushed through (DISTORTION, SATURATION): its fader works that engine (see bendFx)
    std::shared_ptr<const move::Found> moveOwn, moveOther;   // the MOVEMENT group: what moves in the sound worked on, and in the other sound (two sounds); see movementFx
    Vec curveL, curveR;                        // TONE MATCH, for the picture and Export IR: the curves of the last render (left, right; Amount in the middle)
    Vec foundL, foundR;                        // the same with Detail in the middle too
};

inline std::string titleFor (const std::string& f)
{
    static const std::map<std::string, std::string> t { { "flanger", "FLANGER" }, { "chorus", "CHORUS" }, { "phaser", "PHASER" }, { "haas", "HAAS WIDTH" },
        { "width", "STEREO WIDTH" }, { "delay", "DELAY" }, { "reverb", "REVERB" }, { "distortion", "DISTORTION" }, { "bitcrush", "BITCRUSH" },
        { "filtersweep", "FILTER SWEEP" }, { "tremolo", "TREMOLO" }, { "autopan", "AUTO PAN" },
        { "saturation", "SATURATION" }, { "exciter", "EXCITER" }, { "ringmod", "RING MOD" }, { "filter", "FILTER" }, { "stutter", "STUTTER" }, { "gatedrhythm", "GATED RHYTHM" },
        { "comb", "COMB FILTER" }, { "tonematch", "TONE MATCH" }, { "movement", "MOVEMENT" } };
    auto it = t.find (f); return it == t.end() ? f : it->second;
}

inline std::vector<Knob> knobsFor (const std::string& f)
{
    auto K = [] (std::string key, std::string label, std::string tip, double neutral = 0, double lo = 0, double hi = 1e9) { Knob k; k.label = std::move (label); k.tip = std::move (tip); k.key = std::move (key); k.neutral = neutral; k.lo = lo; k.hi = hi; return k; };
    std::vector<Knob> k { K ("", "Amount", "How much of the effect. Middle = as found. Left = none. Right = double.") };
    if (f == "flanger") { k.push_back (K ("rate", "Speed", "Sweep speed, scaled from the speed found.", 0, 0.01, 10)); k.push_back (K ("dmax", "Sweep", "How far the sweep travels.", 0, 0.2e-3, 20e-3)); k.push_back (K ("depth", "Intensity", "Strength of the comb.", 0, 0, 1)); }
    else if (f == "chorus") { k.push_back (K ("rate", "Speed", "Wobble speed.", 0, 0.01, 10)); k.push_back (K ("base", "Delay", "Distance of the copies.", 0, 1e-3, 60e-3)); k.push_back (K ("width", "Wobble", "How far the copies drift.", 0, 0, 30e-3)); k.push_back (K ("mix", "Mix", "Level of the copies.", 0, 0, 1)); }
    else if (f == "phaser") { k.push_back (K ("rate", "Speed", "Sweep speed.", 0, 0.01, 10)); k.push_back (K ("fmin", "Low", "Bottom of the sweep.", 0, 30, 8000)); k.push_back (K ("fmax", "High", "Top of the sweep.", 0, 200, 20000)); k.push_back (K ("fb", "Feedback", "Sharpness of the notches.", 0, 0, 0.95)); k.push_back (K ("mix", "Mix", "Level of the swept part.", 0, 0, 1)); }
    else if (f == "filtersweep") { k.push_back (K ("rate", "Speed", "Sweep speed.", 0, 0.01, 10)); k.push_back (K ("fmin", "Low", "Bottom of the sweep.", 0, 30, 8000)); k.push_back (K ("fmax", "High", "Top of the sweep.", 0, 200, 20000)); k.push_back (K ("q", "Resonance", "Peak at the cutoff.", 0, 0.5, 20)); }
    else if (f == "width") k.push_back (K ("amount", "Width", "Added stereo spread.", 0, 0, 2));
    else if (f == "reverb") { k.push_back (K ("rt60", "Length", "Decay time.", 0, 0.1, 10)); k.push_back (K ("mix", "Mix", "Level of the room.", 0, 0, 1)); }
    else if (f == "distortion") k.push_back (K ("drive_scale", "Drive", "How hard it is pushed.", 0, 0.05, 4));
    else if (f == "bitcrush") { k.push_back (K ("bits", "Bit Depth", "Fewer levels to the right.", 16, 1, 16)); k.push_back (K ("down", "Sample Rate", "Coarser steps to the right.", 1, 1, 32)); }
    else if (f == "saturation") k.push_back (K ("drive", "Drive", "How hard it is pushed (dB).", 0, 0, 36));
    else if (f == "exciter") { k.push_back (K ("amount", "Intensity", "Level of the added sparkle.", 0, 0, 1.5)); k.push_back (K ("freq", "Frequency", "Where the sparkle starts.", 0, 500, 14000)); }
    else if (f == "ringmod") { k.push_back (K ("carrier", "Carrier", "Ring frequency.", 0, 5, 5000)); k.push_back (K ("mix", "Mix", "Level of the ringing part.", 0, 0, 1)); }
    else if (f == "filter") { k.push_back (K ("cutoff", "Cutoff", "Filter frequency.", 0, 20, 20000)); k.push_back (K ("resonance", "Resonance", "Peak at the cutoff.", 0, 0.3, 12)); }
    else if (f == "stutter") { k.push_back (K ("slice_ms", "Slice", "Length of each repeated slice.", 0, 10, 1000)); k.push_back (K ("repeats", "Repeats", "How many times a slice repeats.", 1, 1, 16)); k.push_back (K ("prob", "Chance", "How often it stutters.", 0, 0, 1)); }
    else if (f == "gatedrhythm") { k.push_back (K ("rate", "Speed", "Steps per second.", 0, 0.2, 40)); k.push_back (K ("duty", "Open", "How long each step stays open.", 0, 0.05, 1)); k.push_back (K ("depth", "Depth", "How far it cuts down.", 0, 0, 1)); }
    else if (f == "tremolo" || f == "autopan") { k.push_back (K ("rate", "Speed", "Wobble speed.", 0, 0.02, 20)); k.push_back (K ("amp_db", "Depth", "How deep the movement goes.", 0, 0, 40)); }
    // Every fader needs its own id: a setting that is itself called "amount" (STEREO WIDTH's Width, EXCITER's Intensity)
    // must not share the id of the group's Amount fader, or the two faders move one value.
    for (auto& x : k) x.id = f + "." + (x.key.empty() ? std::string ("amount") : x.key == "amount" ? std::string ("depth") : x.key);
    return k;
}

// ---------------------------------------------------------------- the reverb the room engine found (RoomEngine.h)
// What the room engine's answer says about whether there is a room on the sound at all.
struct RoomEvidence
{
    double share = -99;     // the part of the sound the engine takes for room against the part it takes for sound, dB (cells within 50 dB of the loudest)
    double deep = 0;        // share of those cells that are clearly room (answer under -12 dB)
    int bands = 0;          // bands with moments that were all room
    double moments = 0;     // such moments, all bands together
    double decay = 0, strength = 0;     // mid bands: decay time (s), room against direct sound (dB)
    double own = -99;       // the part of the sound that left and right do not share, dB under the whole (a mono file: -40)
};
inline RoomEvidence roomEvidence (const room::Found& f)
{
    RoomEvidence e; const size_t n = f.gains.size(); if (n == 0 || f.picture.LE.size() != n) return e;
    const double top = f.picture.top(); double asRoom = 0, asSound = 0, unshared = 0; size_t live = 0, clear = 0; const bool wide = f.picture.U256.size() == n;
    for (size_t i = 0; i < n; ++i)
    {
        if (! (f.picture.LE[i] > top - 50.0)) continue;
        const double E = std::pow (10.0, (f.picture.LE[i] - top) / 10.0), G = std::pow (10.0, (double) f.gains[i] / 10.0);
        asRoom += E * (1.0 - G); asSound += E * G; ++live; clear += (double) f.gains[i] < -12.0 ? 1u : 0u;
        if (wide) unshared += E * std::pow (10.0, std::clamp ((double) f.picture.U256[i], -40.0, 0.0) / 10.0);
    }
    e.share = 10.0 * std::log10 (std::max (asRoom, 1e-30) / std::max (asSound, 1e-30)); e.deep = live > 0 ? (double) clear / (double) live : 0.0;
    if (wide && asRoom + asSound > 0) e.own = 10.0 * std::log10 (std::max (unshared, 1e-30) / (asRoom + asSound));
    if (f.found)
    {
        const room::Bands B; Vec t, k;
        for (size_t b = 0; b < (size_t) room::NB; ++b) { e.bands += f.ev[b] > 0; e.moments += f.ev[b]; if (room::midBand (B.fc[b])) { t.push_back (f.t60[b]); k.push_back (f.kdb[b]); } }
        e.decay = room::middle (t); e.strength = room::middle (k);
    }
    return e;
}
// How sure it is that there is a room on the sound, 0 .. 1: what the Sureness slider compares. A probability for a
// sound that is somewhat more likely to carry reverb than not (62 in 100). Most of the answer is whether the engine
// finds a room at all; among the found ones, more cells that are clearly room and more bands with all-room moments
// make it surer.
// Fitted for the model of 0.4.6 (first look unet_y1, second look unet_y1p2, with the guard of RoomNumbers.h) on 2470
// sounds that are not its training cases (Research/room/bench/prep_sb.py, prep_nw.py, fit_sure.py): every reverb
// tool on a clean clip and after another effect, ringing out and cut off at 6 s, against clean clips, the other 17
// effect families alone and tone effects alone. The rule is fitted on the half made with the tools and clips that
// training used and checked on the other half, which the model never saw. There the engine finds a room on 95 % of
// the reverbs that ring out, on 76 % of those that are cut off and on 13 % of the sounds without one (the model of
// 0.4.5: 96, 77 and 18 %); at the slider's 65 % the reverb shows on 94 %, on 68 % and on 11 % (clean clips 11 %,
// another effect alone 12 %, a tone effect alone 2 %; with the look without the other effects at the end of this
// file: another effect alone 10 %, all sounds without reverb 9 %), and what the number says holds: of the sounds it
// calls 65 to 80 % sure 77 % have a reverb, 80 to 90: 85 %, over 90: 95 %. (The rule of 0.4.5, for its model: 1.902,
// 0.847, 2.649 and 0.22.) The sounds without reverb it still shows on are mostly recorded loops and recordings,
// which may carry a room of their own, and sampled instruments that ring by themselves or carry the hall they were
// recorded in. More evidence (the room part's share, strength, decay time, the levels at the file's two ends, a look
// with silence after the sound, a look at left and right summed to mono) was tried and did not help on the unseen
// half.
// roomNone: what is left when the engine finds no room (then about 3 in 4 sounds have none).
constexpr double roomSure0 = 1.892, roomSureDeep = 0.758, roomSureBands = 2.875, roomNone = 0.25;
inline double roomSureness (const room::Found& f)
{
    if (! f.found) return roomNone;
    const RoomEvidence e = roomEvidence (f);
    return 1.0 / (1.0 + std::exp (-(roomSure0 + roomSureDeep * std::log10 (e.deep + 0.003) + roomSureBands * (double) e.bands / (double) room::NB)));
}
// ---------------------------------------------------------------- a long sound
// The model has learned from sounds of 4.5 to 9 s, and the rule above is fitted on sounds of 6 and 9 s. A look at the
// whole of a longer sound reads a room into it far too easily: on 121 sounds of 18 to 48 s without reverb (clean clips
// played three and six times over, and five to eight different clips in a row, clean or with a sweep, a phaser or a
// chorus) the rule showed REVERB on 40 % (on 10 of the 12 clean rows of clips; build 0.4.4 on 9 of them); the clips
// of the loops show it on 10 % when each is looked at alone. Told about a room it believes to be there somewhere,
// the model finds some of it everywhere.
// So a sound of more than 9 s is judged in pieces of 6 s, each looked at as a sound of its own, as the model has
// learned it: a new piece every 3 s at most (at most roomPiecesMost in all, spread evenly, the last one ending where
// the sound ends; silence of more than 60 dB under the loudest sample before and after the sound is left out). The
// verdict is the larger of two: the average of the pieces' sureness (each piece counting by how much of it sounds:
// its cells within 50 dB of the sound's loudest), which asks whether the room shows along the sound, and the
// sureness of the last piece, which asks whether the sound rings out at its end: a room on a dense sound may show
// nowhere else. (The last piece is only asked when it holds at least a fifth of what the fullest piece holds.)
// The room that is shown and worked with is still the one found on the whole sound; if none is found there, there
// is nothing to show.
// Measured on those 121 sounds and on 144 looks at sounds of 18 to 48 s with a reverb (Research/room/bench/
// prep_long.py, long_eval3.py; shown at 65 %): without reverb 40 % -> 17 % (clean clips looped 24 -> 10 %, which is
// what those clips show alone; rows of clips 73 -> 32 %, 11 of the 13 by a last clip that reads as a room by
// itself); with reverb 76 -> 74 % (ringing out 85 -> 81 %, cut off 71 -> 70 %). The average alone would show 5 % and
// 69 %, and miss the reverbs that only their tail gives away (ringing out: 68 %). Taking the whole sound's evidence
// stretch by stretch instead (long_eval3.py) leaves 22 % and 74 %. The pieces take about twice as long as the look at
// the whole sound, which adds about a fifth to the analysis time of a long sound.
// A piece's own verdict can depend a good deal on where exactly it starts (on one looped clip with a light reverb,
// pieces that start within 50 ms of each other read between 22 and 97 %): the average evens that out, the last
// piece does not.
constexpr int roomWholeFrames = 852;            // a sound of up to this many frames (9.02 s) is judged whole, as above
constexpr double roomPieceSec = 6.0, roomPieceStepSec = 3.0; constexpr int roomPiecesMost = 24;
struct RoomPiece { size_t from = 0; double sure = 0, weight = 0; };     // where it starts (samples), its sureness, how much of it sounds
// The verdict on a long sound from its pieces (see above); -1 when stopped on the way. each: the pieces, if asked for.
inline double roomInPieces (const Stereo& y, const room::Net& net, const room::Net* second, const room::Stop& stop = {}, std::vector<RoomPiece>* each = nullptr)
{
    if (each != nullptr) each->clear();
    const size_t n = y.size(); const double quiet = y.peak() * 1e-3; if (n == 0 || ! (quiet > 0.0)) return roomNone;
    size_t a = 0, b = n;        // the sound without the silence around it
    while (a < n && std::abs (y.l[a]) <= quiet && std::abs (y.r[a]) <= quiet) ++a;
    while (b > a && std::abs (y.l[b - 1]) <= quiet && std::abs (y.r[b - 1]) <= quiet) --b;
    const size_t len = std::min ((size_t) (roomPieceSec * SR), n), step = (size_t) (roomPieceStepSec * SR);
    if (b - a < len) { a = std::min (a, n - len); b = a + len; }
    const size_t way = b - a - len;         // from the first piece's start to the last one's
    const int count = (int) std::min ((way + step - 1) / step + 1, (size_t) roomPiecesMost);
    std::vector<RoomPiece> P; std::vector<Vec> LE; double top = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < count; ++k)
    {
        RoomPiece p; p.from = a + (count > 1 ? (size_t) std::llround ((double) way * (double) k / (double) (count - 1)) : 0); p.sure = roomNone;
        const Stereo z = y.slice (p.from, len);
        if (z.peak() > quiet)       // a piece of silence holds nothing to look at
        {
            room::Found f = room::find (z, net, second, stop); if (f.stopped) return -1.0;
            p.sure = roomSureness (f); top = std::max (top, f.picture.top()); LE.push_back (std::move (f.picture.LE));
        }
        else LE.emplace_back();
        P.push_back (p);
    }
    double sum = 0.0, all = 0.0, most = 0.0;
    for (size_t k = 0; k < P.size(); ++k)
    {
        size_t live = 0; for (double v : LE[k]) live += v > top - 50.0 ? 1u : 0u;
        P[k].weight = (double) live; sum += P[k].sure * P[k].weight; all += P[k].weight; most = std::max (most, P[k].weight);
    }
    if (each != nullptr) *each = P;
    if (! (all > 0.0)) return roomNone;
    return std::max (sum / all, P.back().weight >= 0.2 * most ? P.back().sure : 0.0);
}
// How sure it is that there is a room on the sound y, of which f is what the room engine found: a sound of up to 9 s
// by the rule on the whole sound, a longer one in pieces.
inline double roomVerdict (const Stereo& y, const room::Found& f, const room::Net& net, const room::Net* second, const room::Stop& stop = {}, std::vector<RoomPiece>* each = nullptr)
{
    if (each != nullptr) each->clear();
    if (! f.found || f.picture.frames <= roomWholeFrames) return roomSureness (f);
    const double s = roomInPieces (y, net, second, stop, each); return s < 0.0 ? roomSureness (f) : s;
}
// The REVERB group of a found room: Amount, Length, Brightness and Width, each 0 .. 2 with 1 = as found (the
// settings are the fader values themselves, so scaled() hands them through).
inline FoundFx roomFx (std::shared_ptr<const room::Found> found, double prob)
{
    FoundFx F; F.name = "reverb"; F.title = "REVERB"; F.prob = prob; F.room = std::move (found);
    F.est = { { "length", 1.0 }, { "bright", 1.0 }, { "width", 1.0 } };
    auto K = [] (const char* key, const char* label, const char* tip)
    {
        Knob k; k.key = key; k.id = "reverb." + (k.key.empty() ? std::string ("amount") : k.key); k.label = label; k.tip = tip; k.neutral = 0; k.lo = 0; k.hi = 2; k.ownTip = ! k.key.empty();
        return k;
    };
    F.knobs = { K ("", "Amount", ""),
                K ("length", "Length", "How long the room rings. Middle = as found. Left = shorter, right = up to twice as long."),
                K ("bright", "Brightness", "The tone of the room. Middle = as found. Left = darker (the highs die away sooner), right = brighter."),
                K ("width", "Width", "How wide the room is. Middle = as found. Left = narrower, down to mono. Right = wider.") };
    return F;
}
// The Amount fader of a found room on the sound itself (s 0 .. 2, 1 = untouched). Left of the middle the engine's
// own answer comes in along a straight line (in dB), and at far left it is in full. Right of the middle: more reverb.
// (The prototype's engine.fader. Up to 0.4.5 the answer was in full at half way left and was made bolder from there
// to far left: the doubtful cells were turned down further. Measured on the scoreboard's 120 reverbs that was the
// worst of 28 ways to shape the answer: 37 % of the cases came out worse than untouched against 14 % for the plain
// answer, and on the sound itself, without its ring-out, 56 % against 28 %: Research/room/train/depth2_test.py.)
// What the fader does to the room's level, for the display (dB): how far the room comes down where it is on its own
// is the typical value over the scoreboard's 120 reverb cases with the model of 0.4.5, 7.5 dB with the engine's own
// answer (Research/room/train/bold_test.py; it differs from sound to sound, and under the sound itself less can be
// taken: there the sound loses 1.6 dB). Right of the middle: plain level.
constexpr double roomDownOwn = 7.5;
inline double roomAmountDb (double s) { return s >= 1.0 ? 20.0 * std::log10 (s) : -roomDownOwn * (1.0 - s); }
inline Stereo roomFader (const Stereo& y, const room::Found& f, double s, double length, double bright, double width, const room::Stop& stop = {})
{
    return room::oneSound (y, f, s, length, bright, width, stop);
}
// What Export IR writes: the found room alone (no direct sound; the tail starts 10.7 ms after where the direct sound
// would be), shaped by the Length, Brightness and Width faders, brought to unity level (a sound run through it
// comes out about as loud as it went in), which is what convolution reverbs expect of a file without the direct
// sound. levelDb is the wet level against the dry signal (dB) at which a convolution reverb with this file gives
// the room as the app puts it on a second sound with Amount in the middle.
struct RoomIr { Stereo h; double levelDb = 0; };
inline RoomIr exportIr (const room::Found& f, double length = 1.0, double bright = 1.0, double width = 1.0)
{
    RoomIr out; out.h = room::roomIr (f, length, bright, width); const size_t n = out.h.size(); if (n == 0) return out;
    double e = 0; for (size_t i = 0; i < n; ++i) e += 0.5 * (out.h.l[i] * out.h.l[i] + out.h.r[i] * out.h.r[i]);
    if (! (e > 0)) return out;
    out.levelDb = 10.0 * std::log10 (e) + room::leanDb (f); double g = 1.0 / std::sqrt (e); const double pk = out.h.peak() * g; if (pk > 0.999) g *= 0.999 / pk;
    for (size_t i = 0; i < n; ++i) { out.h.l[i] *= g; out.h.r[i] *= g; }
    return out;
}
// The room engine's verdict goes into the inspection, and the detector's own guess about reverb goes out: on 76
// scoreboard sounds the detector showed the reverb on 67 % of those that have one, on 7 % of the dry ones and on
// 25 % of those with a tone effect only, and where the two disagree it is no help either way.
// That holds for narrow and mono recordings too since the model of 0.4.1, which was also trained on them: on ten
// scoreboard sounds narrowed step by step down to mono it still finds the room on 9 and shows it (rule above 65 %) on
// 8, where the first model showed it on about 4 in 10 (Research/room/train/narrow_test.py). The first model's
// exception (no room found on a nearly mono file: the detector's guess stays) is gone with it; the function still
// answers whether the engine's verdict was taken, which is now always.
// sure: roomVerdict() of the sound.
inline bool mergeRoom (Inspection& I, double sure)
{
    I.prob["reverb"] = sure; return true;
}

// A sound that is cut off while it is still playing has no ending. The models up to 0.4.4 had learned from sounds
// that end inside the file only: they read a room into many dry sounds that simply stop, and on a real reverb that is
// cut off they did as much harm as good. From 0.4.2 a found room was therefore looked at a second time with silence
// after the sound, and the less sure verdict counted (recheckRoomEnd). The model of 0.4.5 has learned from files that
// stop while the sound plays and from excerpts cut at both ends (Research/room/train/mkcases_x.py); next to its
// verdict on the sound as it is, the look with silence adds nothing (fit_sure.py), so it is gone, and with it one
// run of the model per sound.

// ---------------------------------------------------------------- the tone engine's groups (Tone.h)
// COMB FILTER: short reflections found on the sound (a copy of it a few milliseconds later, which colours the tone
// with evenly spaced notches). The finder is strict: on the scoreboard it reports all 8 sounds that have one and
// none of the 184 others, and on 1057 clean clips it reports 2 that are not there; with one sound in twenty
// carrying a reflection that makes a report right 19 times in 20, which is the sureness shown.
// Amount: on the sound itself 0 = taken out (the exact inverse filter), 1 = untouched, 2 = once more; on a second
// sound 0 = none, 1 = as found, 2 = twice. Delay stretches the reflections' times (0.25 .. 2).
constexpr double combSure = 0.95;
inline FoundFx combFx (std::shared_ptr<const tone::Found> found)
{
    FoundFx F; F.name = "comb"; F.title = titleFor ("comb"); F.prob = combSure; F.tone = std::move (found); F.est = { { "delay", 1.0 } };
    Knob a; a.id = "comb.amount"; a.label = "Amount"; a.hi = 2;
    Knob d; d.key = "delay"; d.id = "comb.delay"; d.label = "Delay"; d.neutral = 0; d.lo = 0.25; d.hi = 2; d.ownTip = true;
    d.tip = "The time of the reflection. Middle = as found. Left = shorter (the notches move up and apart), right = longer, up to twice (they move down and closer).";
    F.knobs = { a, d }; return F;
}
inline void mergeTone (Inspection& I, const tone::Found& f) { if (! f.echoes.empty() && ! f.spec.silent) I.prob["comb"] = combSure; }
// The gain curve (dB per bin) of the COMB FILTER group. onSecond: the found reflections go onto another sound
// (s times the curve). Otherwise the sound's own: the found curve comes out and the one the faders give goes in.
inline Vec combGain (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    Vec g ((size_t) tone::NB, 0.0); if (F.tone == nullptr || F.tone->echoes.empty() || F.tone->spec.silent) return g;
    const double stretch = std::clamp (get (p, "delay", 1.0), 0.25, 2.0);
    const Vec now = tone::combCurve (F.tone->echoes, F.tone->spec.both, stretch);
    if (onSecond) { for (size_t k = 0; k < g.size(); ++k) g[k] = s * now[k]; return g; }
    if (std::abs (stretch - 1.0) < 1e-9) { for (size_t k = 0; k < g.size(); ++k) g[k] = (s - 1.0) * now[k]; return g; }
    const Vec was = tone::combCurve (F.tone->echoes, F.tone->spec.both, 1.0);
    for (size_t k = 0; k < g.size(); ++k) g[k] = s * now[k] - was[k];
    return g;
}
// TONE MATCH (two sounds, switched on by its button): the second sound takes the long-term tone of the first. It
// is not a found effect but a transfer, so it shows no sureness and is off until asked for: on two unrelated
// sounds it changes the second one a lot (most such pairs reach a cap somewhere).
// Amount: 0 = none, 1 = matched, 2 = the curve twice. Detail: how fine the curve is; in the middle a Gaussian of
// 1/12 octave (Detail 0.6), which brings two different sounds' spectra within 1.6 dB (median, scoreboard)
// without copying the first sound's single partials. When the two sounds are versions of the same recording
// (`same`) the difference of their spectra is the effect's own curve: then the middle is full detail and left
// and right each get their own curve (on the scoreboard that takes out 97.6 % of an EQ, filter, resonance or
// comb against 70.9 % at Detail 0.6).
constexpr double matchDetail = 0.6;
inline FoundFx matchFx (std::shared_ptr<const tone::Found> found, bool same)
{
    FoundFx F; F.name = "tonematch"; F.title = titleFor ("tonematch"); F.prob = -1.0; F.tone = std::move (found);
    F.est = { { "detail", same ? 1.0 : matchDetail }, { "same", same ? 1.0 : 0.0 } };
    Knob a; a.id = "tonematch.amount"; a.label = "Amount"; a.hi = 2;
    Knob d; d.key = "detail"; d.id = "tonematch.detail"; d.label = "Detail"; d.neutral = 0; d.lo = 0; d.hi = 1; d.ownTip = true;
    d.tip = same ? "How fine the tone curve is. The two sounds are versions of the same recording, so the middle already takes every detail: the curve is the effect's own. Left = broad strokes only."
                 : "How fine the tone curve is. Left = broad strokes only (an octave wide), middle = a twelfth of an octave, right = every detail (that also copies the first sound's single notes: right for two versions of the same recording).";
    F.knobs = { a, d }; return F;
}
// The Detail fader (v 0 .. 2) as Detail 0 .. 1: the middle is the group's own value, far left 0, far right 1.
inline double matchDetailAt (const FoundFx& F, double v)
{
    const double base = std::clamp (get (F.est, "detail", matchDetail), 0.0, 1.0);
    return std::clamp (v <= 1.0 ? base * v : base + (1.0 - base) * (v - 1.0), 0.0, 1.0);
}
// The curves (left, right; dB per bin, Amount in the middle) that give a sound with the long-term spectra `sb` the
// tone of the group's first sound at this Detail. Empty when one of the two is silent.
inline std::pair<Vec, Vec> matchCurves (const FoundFx& F, const tone::Spectra& sb, double detail)
{
    if (F.tone == nullptr || F.tone->spec.silent || sb.silent) return {};
    if (get (F.est, "same", 0.0) > 0.5) return { tone::matchCurve (F.tone->spec.left, sb.left, detail), tone::matchCurve (F.tone->spec.right, sb.right, detail) };
    Vec c = tone::matchCurve (F.tone->spec.both, sb.both, detail); return { c, c };
}

// What Export IR writes for a group of the tone engine: its curve (caps included) as a minimum-phase impulse
// response, 16384 taps, left and right. A file cannot hold more than full scale, so a response that would go over
// is turned down; downDb says by how much (add it back after the convolution).
struct ToneIr { Stereo h; double downDb = 0; };
inline ToneIr toneIr (const Vec& curveL, const Vec& curveR)
{
    ToneIr out; if (curveL.size() != (size_t) tone::NB || curveR.size() != (size_t) tone::NB) return out;
    out.h.l = tone::toIr (tone::softCap (curveL)); out.h.r = curveL == curveR ? out.h.l : tone::toIr (tone::softCap (curveR));
    const double pk = out.h.peak(); if (pk > 0.999) { const double g = 0.999 / pk; out.downDb = -20.0 * std::log10 (g); for (auto& v : out.h.l) v *= g; for (auto& v : out.h.r) v *= g; }
    return out;
}

// ---------------------------------------------------------------- the echo engine's group (Echo.h)
// DELAY: the echoes the echo engine found on the sound (30 ms to 2.5 s; one echo time per channel, with its
// repeats). On the scoreboard's 8 cases with a delay from outside effects the group shows on 7 at the default
// Sureness (none with a wrong echo time), takes 89 of 100 out (median; mean 77) and puts 95 of 100 onto the case's
// other sound (mean 83), where the detector and the fitted delay of 0.4.1 showed it on 6 and reached 54 and 50
// (Research/room/bench/famboard.py; sound by sound: Research/echo/evcpp.py).
// Amount: on the sound itself 0 = taken out (the exact inverse of the found echo filter), 1 = untouched, 2 = the
// echoes twice as loud; on a second sound 0 = none, 1 = as found, 2 = twice. Time: the echo time, half to double.
// Feedback: how long the repeats go on. Brightness: the tone of the echoes. The setting faders hand their values
// (0 .. 2, 1 = as found) through; echoShape turns them into what the engine takes.
inline FoundFx echoFx (std::shared_ptr<const echo::Found> found, double prob)
{
    FoundFx F; F.name = "delay"; F.title = titleFor ("delay"); F.prob = prob; F.echo = std::move (found);
    F.est = { { "time", 1.0 }, { "feedback", 1.0 }, { "bright", 1.0 } };
    auto K = [] (const char* key, const char* label, const char* tip)
    {
        Knob k; k.key = key; k.id = "delay." + (k.key.empty() ? std::string ("amount") : k.key); k.label = label; k.tip = tip; k.neutral = 0; k.lo = 0; k.hi = 2; k.ownTip = ! k.key.empty();
        return k;
    };
    F.knobs = { K ("", "Amount", ""),
                K ("time", "Time", "The echo time. Middle = as found. Left = shorter, down to half. Right = longer, up to twice."),
                K ("feedback", "Feedback", "How long the echoes go on. Middle = as found. Left = fewer repeats, down to the first one alone. Right = more repeats (an echo without repeats gets some)."),
                K ("bright", "Brightness", "The tone of the echoes. Middle = as found. Left = darker, right = brighter.") };
    return F;
}
inline echo::Shape echoShape (const Params& p, double s)
{
    echo::Shape sh; sh.amount = std::max (s, 0.0); sh.time = std::pow (2.0, std::clamp (get (p, "time", 1.0), 0.0, 2.0) - 1.0);
    sh.feedback = std::clamp (get (p, "feedback", 1.0), 0.0, 2.0); sh.bright = std::clamp (get (p, "bright", 1.0), 0.0, 2.0); return sh;
}
// The echo engine's verdict goes into the inspection and the detector's own guess about delay goes out (it showed
// a delay on 6 of the scoreboard's 8 delay cases and on about one clean sound in ten).
inline void mergeEcho (Inspection& I, const echo::Found& f) { I.prob["delay"] = f.any() ? f.sure : echo::sureNone; }

// What Export IR writes for the DELAY group: the echoes alone, without the sound itself (the file starts with the
// silence before the first echo), left and right, as the Time, Feedback and Brightness faders have them and with
// Amount in the middle. In a convolution plug-in, wet and dry at the same level then give the echoes as found. A
// channel without an echo stays silent. A file cannot hold more than full scale, so a response that would go over
// is turned down; downDb says by how much (add it back after the convolution).
struct EchoIr { Stereo h; double downDb = 0; };
inline EchoIr echoIr (const echo::Found& f, const Params& p)
{
    EchoIr out; if (! f.any()) return out;
    echo::Fft F; const echo::Shape sh = echoShape (p, 1.0); Vec h[2];
    for (int c = 0; c < 2; ++c)
    {
        if (! f.ch[c].found) continue;
        if (c == 1 && f.twin && f.ch[0].found) { h[1] = h[0]; continue; }
        h[c] = echo::response (F, f.ch[c], sh, false); if (! h[c].empty()) h[c][0] -= 1.0;      // the sound itself out
    }
    const size_t n = std::max (h[0].size(), h[1].size()); if (n < 2) return out;
    h[0].resize (n, 0.0); h[1].resize (n, 0.0); out.h.l = std::move (h[0]); out.h.r = std::move (h[1]);
    const double pk = out.h.peak(); if (pk > 0.999) { const double g = 0.999 / pk; out.downDb = -20.0 * std::log10 (g); for (auto& v : out.h.l) v *= g; for (auto& v : out.h.r) v *= g; }
    return out;
}

// ---------------------------------------------------------------- the level engine's groups (Level.h)
// TREMOLO, AUTO PAN and GATED RHYTHM: a gain that moves and repeats, as the level engine measured it on the sound
// (the period, the shape over one cycle for each side, a gate's steps to a fraction of a millisecond). Left and
// right moving against each other is AUTO PAN; both together is GATED RHYTHM when the shape has steep steps and
// TREMOLO otherwise.
// Amount: on the sound itself 0 = taken out (the sound lifted by what the gain took), 1 = untouched, 2 = twice as
// deep; on a second sound 0 = none, 1 = as found, 2 = twice as deep (all in dB). Speed: half to double; on the sound
// itself a changed speed means the found movement out and the new one in.
inline const char* levelName (const level::Movement& m) { return m.against ? "autopan" : (m.steps ? "gatedrhythm" : "tremolo"); }
inline FoundFx levelFx (std::shared_ptr<const level::Movement> found, double prob)
{
    FoundFx F; F.name = levelName (*found); F.title = titleFor (F.name); F.prob = prob; F.level = std::move (found); F.est = { { "speed", 1.0 } };
    Knob a; a.id = F.name + ".amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2;
    Knob s; s.key = "speed"; s.id = F.name + ".speed"; s.label = "Speed"; s.neutral = 0; s.lo = 0; s.hi = 2; s.ownTip = true;
    s.tip = "How fast the movement goes. Middle = as found. Left = slower, down to half. Right = faster, up to twice.";
    F.knobs = { a, s }; return F;
}
// The Speed fader (0 .. 2, 1 = as found) as a factor: half to double.
inline double levelSpeed (const Params& p) { return std::pow (2.0, std::clamp (get (p, "speed", 1.0), 0.0, 2.0) - 1.0); }
// The level engine's verdict goes into the inspection: of the three groups only the one it found can show.
inline void mergeLevel (Inspection& I, const level::Movement& m)
{
    for (const char* f : { "tremolo", "autopan", "gatedrhythm" }) I.prob[f] = level::sureNone;
    if (m.found) I.prob[levelName (m)] = m.sure;
}

// ---------------------------------------------------------------- the flanger engine's group (Sweep.h)
// FLANGER: the copy of the sound whose delay moves, as the flanger engine measured it. Amount: on the sound itself 0 =
// the copy taken out, 1 = untouched, 2 = the copy twice as loud; on a second sound 0 = none, 1 = as found, 2 = twice
// as loud. Speed: the sweep half to twice as fast. Sweep: how far the delay moves (0 = it stands still at its middle,
// 2 = twice as far). Feedback: less or more ringing than found. On the sound itself a moved setting means the found
// copy out and the new one in.
constexpr double sweepSure = 0.9;                                       // the flanger engine's rule says yes or no: a yes stands here on the Sureness scale
inline FoundFx flangerFx (std::shared_ptr<const sweep::Result> found, double prob)
{
    FoundFx F; F.name = "flanger"; F.title = titleFor (F.name); F.prob = prob; F.sweep = std::move (found); F.est = { { "speed", 1.0 }, { "sweep", 1.0 }, { "feedback", 1.0 } };
    auto K = [] (const char* key, const char* label, const char* tip) { Knob k; k.key = key; k.id = std::string ("flanger.") + key; k.label = label; k.tip = tip; k.neutral = 0; k.lo = 0; k.hi = 2; k.ownTip = true; return k; };
    Knob a; a.id = "flanger.amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2;
    F.knobs = { a, K ("speed", "Speed", "How fast the comb sweeps. Middle = as found. Left = slower, down to half. Right = faster, up to twice."),
                K ("sweep", "Sweep", "How far the comb moves. Middle = as found. Left = less, down to a comb that stands still. Right = further, up to twice as far."),
                K ("feedback", "Feedback", "How much the comb rings. Middle = as found. Left = less. Right = more.") };
    return F;
}
// The faders (0 .. 2, 1 = as found) as the engine takes them: the speed as a factor (half to double), the sweep's
// reach as a factor (0 .. 2), the feedback as what is added to the found one (-0.5 .. +0.5).
inline double sweepSpeed (const Params& p) { return std::pow (2.0, std::clamp (get (p, "speed", 1.0), 0.0, 2.0) - 1.0); }
inline double sweepReach (const Params& p) { return std::clamp (get (p, "sweep", 1.0), 0.0, 2.0); }
inline double sweepFeedback (const Params& p) { return 0.5 * (std::clamp (get (p, "feedback", 1.0), 0.0, 2.0) - 1.0); }
// The flanger engine's verdict goes into the inspection: the FLANGER group shows when the engine found a moving copy.
// A long sound is measured in pieces: when only one of several shows the copy the report stays under the default Sureness.
constexpr double sweepAlone = 0.5;
inline void mergeSweep (Inspection& I, const sweep::Result& r) { I.prob["flanger"] = ! r.found ? level::sureNone : r.sounding >= 2 && r.shown < 2 ? sweepAlone : sweepSure; }

// ---------------------------------------------------------------- the moving-tone engine's groups (Moving.h)
// PHASER and FILTER SWEEP: a filter whose curve moves, as the moving-tone engine read it off the sound (the change in
// level per band and moment). Amount: on the sound itself 0 = the curve taken off, 1 = untouched, 2 = twice as strong
// (in dB); on a second sound 0 = none, 1 = as found, 2 = twice as strong. Speed: the movement half to twice as fast; on
// the sound itself a changed speed means the found curve off and the new one on.
inline FoundFx movingFx (std::shared_ptr<const moving::Found> found, double prob)
{
    FoundFx F; F.name = found->kind; F.title = titleFor (F.name); F.prob = prob; F.moving = std::move (found); F.est = { { "speed", 1.0 } };
    Knob a; a.id = F.name + ".amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2;
    Knob s; s.key = "speed"; s.id = F.name + ".speed"; s.label = "Speed"; s.neutral = 0; s.lo = 0; s.hi = 2; s.ownTip = true;
    s.tip = "How fast the filter moves. Middle = as found. Left = slower, down to half. Right = faster, up to twice.";
    F.knobs = { a, s }; return F;
}
// The Speed fader (0 .. 2, 1 = as found) as a factor: half to double.
inline double movingSpeed (const Params& p) { return std::pow (2.0, std::clamp (get (p, "speed", 1.0), 0.0, 2.0) - 1.0); }
// The moving-tone engine's verdict goes into the inspection: of its two groups only the one it found can show, and
// neither when the flanger engine found its moving copy (a comb that sweeps also draws moving notches in the picture
// the moving-tone engine reads; the flanger engine sees the copy itself and takes it out exactly).
inline void mergeMoving (Inspection& I, const moving::Found& f, const sweep::Result* flanger = nullptr)
{
    for (const char* k : { "phaser", "filtersweep" }) I.prob[k] = level::sureNone;
    if (f.found && ! (flanger != nullptr && flanger->found)) I.prob[f.kind] = f.sure;
}

// ---------------------------------------------------------------- the chorus engine's group (Chorus.h)
// CHORUS: copies of the sound whose delays move round and round, as the chorus engine read them off the sound (each
// voice's delay over time and its loudness per frequency range). Amount: on the sound itself 0 = the voices taken out,
// 1 = untouched, 2 = twice as loud; on a second sound 0 = none, 1 = as found, 2 = twice as loud. Speed: the voices move
// half to twice as fast. Depth: how far they swing (0 = they stand still at their middle, 2 = twice as far). On the
// sound itself a moved Speed or Depth means the found voices out and the changed ones in.
inline FoundFx chorusFx (std::shared_ptr<const chorus::Found> found, double prob)
{
    FoundFx F; F.name = "chorus"; F.title = titleFor (F.name); F.prob = prob; F.chorus = std::move (found); F.est = { { "speed", 1.0 }, { "depth", 1.0 } };
    auto K = [] (const char* key, const char* label, const char* tip) { Knob k; k.key = key; k.id = std::string ("chorus.") + key; k.label = label; k.tip = tip; k.neutral = 0; k.lo = 0; k.hi = 2; k.ownTip = true; return k; };
    Knob a; a.id = "chorus.amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2;
    F.knobs = { a, K ("speed", "Speed", "How fast the copies drift. Middle = as found. Left = slower, down to half. Right = faster, up to twice."),
                K ("depth", "Depth", "How far the copies drift. Middle = as found. Left = less, down to copies that stand still. Right = further, up to twice as far.") };
    return F;
}
// The faders (0 .. 2, 1 = as found) as the engine takes them: the speed as a factor (half to double), the depth as a factor (0 .. 2).
inline double chorusSpeed (const Params& p) { return std::pow (2.0, std::clamp (get (p, "speed", 1.0), 0.0, 2.0) - 1.0); }
inline double chorusDepth (const Params& p) { return std::clamp (get (p, "depth", 1.0), 0.0, 2.0); }
// The chorus engine's verdict goes into the inspection: the CHORUS group shows when the engine found voices, and not
// when the flanger engine found its moving copy (a flanger is a chorus with one short voice; the flanger engine also
// reads its feedback).
inline void mergeChorus (Inspection& I, const chorus::Found& f, const sweep::Result* flanger = nullptr)
{
    I.prob["chorus"] = f.found && ! (flanger != nullptr && flanger->found) ? f.sure : level::sureNone;
}

// ---------------------------------------------------------------- the sides engine's groups (Sides.h)
// HAAS WIDTH (one side late) and STEREO WIDTH (a sound from the middle made wide), as the sides engine read them off
// the sound: the two short filters through which one sound reaches the two sides, or the filter through which a copy of
// the middle was put at the sides. Amount: on the sound itself 0 = taken out, 1 = untouched (bit for bit), 2 = twice;
// on a second sound 0 = none, 1 = as found, 2 = twice. A plain delay between the sides moves in time with the fader
// (the late side comes nearer and the levels come together), so that no comb is heard on the way.
constexpr double sidesSure = 0.95;
inline const char* sidesName (const sides::Found& f) { return f.kind == sides::HAAS ? "haas" : "width"; }
inline FoundFx sidesFx (std::shared_ptr<const sides::Found> found, double prob)
{
    FoundFx F; F.name = sidesName (*found); F.title = titleFor (F.name); F.prob = prob; F.sides = std::move (found);
    Knob a; a.id = F.name + ".amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2; F.knobs = { a };
    return F;
}
// The sides engine's verdict goes into the inspection and the detectors' own guesses about haas and width go out: the
// engine reads the delay or the widening filters off the sound itself, or says that there is none it can read.
// A pair's later taps are reflections on each side too, and where the two sides' do not cancel in the middle the tone
// engine reports them as well. They are one thing: when every reflection it found sits at one of the pair's taps, the
// COMB FILTER group is not shown (the pair says more: which side, and how loud on each).
inline bool sidesHoldComb (const sides::Found& f, const tone::Found& t)
{
    if (f.mode != sides::PAIR || t.echoes.empty()) return false;
    for (const auto& e : t.echoes)
    {
        bool at = false;
        for (const auto* v : { &f.a, &f.b }) for (const auto& tap : *v) at = at || (tap.lag > 0 && std::abs (e.delay - (double) tap.lag) <= 2.0);
        if (! at) return false;
    }
    return true;
}
inline void mergeSides (Inspection& I, const sides::Found& f, const tone::Found* tone = nullptr)
{
    I.prob["haas"] = f.kind == sides::HAAS ? sidesSure : level::sureNone; I.prob["width"] = f.kind == sides::WIDTH ? sidesSure : level::sureNone;
    if (tone != nullptr && sidesHoldComb (f, *tone)) I.prob["comb"] = level::sureNone;
}
// ---------------------------------------------------------------- the ring engine's group (Ring.h)
// RING MOD: the tone the sound was multiplied by, as the ring engine read it off the sound. Amount: on the sound itself
// 0 = the tone out (the sound divided by the gain), 1 = untouched (bit for bit), 2 = the tone part twice as strong; on
// a second sound 0 = none, 1 = as found, 2 = the tone part twice as strong. Carrier: the tone's pitch, an octave down
// to an octave up; on the sound itself a moved Carrier means the found tone out and the new one in.
constexpr double ringSure = 0.95;
inline FoundFx ringFx (std::shared_ptr<const ring::Found> found, double prob)
{
    FoundFx F; F.name = "ringmod"; F.title = titleFor ("ringmod"); F.prob = prob; F.ring = std::move (found); F.est = { { "carrier", 1.0 } };
    Knob a; a.id = "ringmod.amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2;
    Knob c; c.key = "carrier"; c.id = "ringmod.carrier"; c.label = "Carrier"; c.neutral = 0; c.lo = 0; c.hi = 2; c.ownTip = true;
    c.tip = "The pitch of the tone the sound is multiplied by. Middle = as found. Left = lower, down to an octave. Right = higher, up to an octave.";
    F.knobs = { a, c }; return F;
}
// The Carrier fader (0 .. 2, 1 = as found) as a factor on the tone's pitch: half to double.
inline double ringCarrier (const Params& p) { return std::pow (2.0, std::clamp (get (p, "carrier", 1.0), 0.0, 2.0) - 1.0); }
// The ring engine's verdict goes into the inspection and the detector's own guess about ring modulation goes out.
inline void mergeRing (Inspection& I, const ring::Found& f) { I.prob["ringmod"] = f.found ? ringSure : level::sureNone; }
// ---------------------------------------------------------------- the crush engine's group (Crush.h)
// BITCRUSH: the grid of values the samples were rounded to and the hold, as the crush engine read them off the sound.
// On a second sound: Amount 0 = none, 1 = through the same crusher, 2 = what the crusher changes twice over. On the
// sound itself: a crusher that mixed plain sound in can be undone (Amount 0 = the sound before it); a bare one cannot,
// since what the rounding and the hold threw away is gone: there the left half of every fader leaves the sound as found.
// Bit Depth: the steps up to 16 times as large (4 bits fewer) or as small. Sample Rate: the hold up to 4 times as long
// or as short. Each of the two is shown when the crusher has that part.
constexpr double crushSure = 0.95;
inline FoundFx crushFx (std::shared_ptr<const crush::Found> found, double prob)
{
    FoundFx F; F.name = "bitcrush"; F.title = titleFor ("bitcrush"); F.prob = prob; F.est = { { "bits", 1.0 }, { "down", 1.0 } };
    const bool soft = found->soft, grid = found->hasGrid(), hold = found->hasHold(); F.crush = std::move (found);
    Knob a; a.id = "bitcrush.amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2; a.ownTip = true;
    a.tip = soft ? "Middle = as found. Left = take the crusher out, far left = the sound before it: this crusher mixed plain sound in, so that sound can be worked out. Right = what the crusher changed, up to twice over."
                 : std::string ("Middle = as found. Right = more of it: towards ") + (grid && hold ? "steps twice as large and a hold twice as long" : grid ? "steps twice as large" : "a hold twice as long")
                       + ". Left of the middle nothing changes: what the crusher threw away is not in the sound any more.";
    a.tip2 = std::string ("How much of the found crusher goes onto the second sound. Middle = ") + (grid && hold ? "the same steps and the same hold" : grid ? "the same steps" : "the same hold") + " as on the first sound. Left = not applied. Right = what the crusher changes, twice over.";
    F.knobs = { a };
    if (grid)
    {
        Knob b; b.key = "bits"; b.id = "bitcrush.bits"; b.label = "Bit Depth"; b.neutral = 0; b.lo = 0; b.hi = 2; b.ownTip = true;
        b.tip = std::string ("The size of the crusher's steps. Middle = as found. Right = larger steps, up to 4 bits fewer. ")
                    + (soft ? "Left = smaller steps, up to 4 bits more." : "Left of the middle nothing changes: smaller steps than the found ones are not in the sound any more.");
        b.tip2 = "The size of the crusher's steps on the second sound. Middle = as found. Right = larger steps, up to 4 bits fewer. Left = smaller steps, up to 4 bits more.";
        F.knobs.push_back (b);
    }
    if (hold)
    {
        Knob d; d.key = "down"; d.id = "bitcrush.down"; d.label = "Sample Rate"; d.neutral = 0; d.lo = 0; d.hi = 2; d.ownTip = true;
        d.tip = std::string ("How long the crusher holds each value: its sample rate. Middle = as found. Right = longer, down to a quarter of the rate. ")
                    + (soft ? "Left = shorter, up to four times the rate." : "Left of the middle nothing changes: what lay between the held values is not in the sound any more.");
        d.tip2 = "How long the crusher holds each value on the second sound: its sample rate. Middle = as found. Right = longer, down to a quarter of the rate. Left = shorter, up to four times the rate.";
        F.knobs.push_back (d);
    }
    return F;
}
// The Bit Depth fader (0 .. 2, 1 = as found) as a factor on the size of the steps: a sixteenth to sixteen times (4 bits
// either way), and never fewer than two levels over full scale.
inline double crushSteps (const FoundFx& F, const Params& p)
{
    const double f = std::pow (2.0, 4.0 * (std::clamp (get (p, "bits", 1.0), 0.0, 2.0) - 1.0));
    return F.crush != nullptr ? std::min (f, std::max (std::pow (2.0, F.crush->bits - 1.0), 1.0)) : f;
}
// The Sample Rate fader (0 .. 2, 1 = as found) as a factor on the hold: a quarter to four times.
inline double crushHold (const Params& p) { return std::pow (2.0, 2.0 * (std::clamp (get (p, "down", 1.0), 0.0, 2.0) - 1.0)); }
// The crush engine's verdict goes into the inspection and the detector's own guess about bitcrush goes out.
inline void mergeCrush (Inspection& I, const crush::Found& f) { I.prob["bitcrush"] = f.found ? crushSure : level::sureNone; }

// ---------------------------------------------------------------- the stutter engine's group (Stutter.h)
// STUTTER: the slices that are played again and again, as the stutter engine read them off the sound. On a second
// sound: Amount 0 = none, 1 = the same slices at the same times, 2 = each slice twice as many extra times; Slice: the
// slices half to twice as long. On the sound itself: right of the middle each slice is played more often (over what
// follows it); left of it fewer times, and since what the repeats covered is gone the sound closes up and ends early;
// Slice: only shorter (a longer slice is not in the sound).
constexpr double stutterSure = 0.95;
inline FoundFx stutterFx (std::shared_ptr<const stutter::Found> found, double prob)
{
    FoundFx F; F.name = "stutter"; F.title = titleFor ("stutter"); F.prob = prob; F.stutter = std::move (found); F.est = { { "slice", 1.0 } };
    Knob a; a.id = "stutter.amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2; a.ownTip = true;
    a.tip = "Middle = as found. Right = every slice is played more often, up to twice as many repeats, over what follows it. Left = fewer repeats, far left = none: the sound closes up and ends early. What the repeats covered is not in the sound any more, so it cannot come back.";
    a.tip2 = "How much of the found stutter goes onto the second sound. Middle = the same slices at the same times. Left = fewer repeats, far left = none. Right = up to twice as many repeats.";
    Knob c; c.key = "slice"; c.id = "stutter.slice"; c.label = "Slice"; c.neutral = 0; c.lo = 0; c.hi = 2; c.ownTip = true;
    c.tip = "The length of each repeated slice. Middle = as found. Left = shorter, down to half: the start of each slice is repeated faster. Right of the middle nothing changes: a longer slice than the found one is not in the sound any more.";
    c.tip2 = "The length of each repeated slice on the second sound. Middle = as found. Left = shorter, down to half. Right = longer, up to double.";
    F.knobs = { a, c }; return F;
}
// The Slice fader (0 .. 2, 1 = as found) as a factor on the slice length: half to double.
inline double stutterSlice (const Params& p) { return std::pow (2.0, std::clamp (get (p, "slice", 1.0), 0.0, 2.0) - 1.0); }
// The stutter engine's verdict goes into the inspection and the detector's own guess about a stutter goes out.
inline void mergeStutter (Inspection& I, const stutter::Found& f) { I.prob["stutter"] = f.found ? stutterSure : level::sureNone; }

// ---------------------------------------------------------------- the filter engine's group (Filter.h)
// FILTER: the curve of a fixed low-pass, high-pass or band-pass, as the filter engine read it off the sound. Amount: on
// the sound itself 0 = the filter out (what it cut lifted again, as far as the engine's caps allow), 1 = untouched,
// 2 = the filter twice; on a second sound 0 = none, 1 = the same curve, 2 = the curve twice. Cutoff: the whole curve
// two octaves down to two up. Resonance: the peak at the corner from none to twice as high; shown when there is one.
// On the sound itself moved faders mean the found curve out and the one the faders give in, as one filter.
inline const char* filterTitle (const fixed::Found& f) { return f.kind == fixed::LOW ? "LOW-PASS FILTER" : f.kind == fixed::HIGH ? "HIGH-PASS FILTER" : f.kind == fixed::BAND ? "BAND-PASS FILTER" : "FILTER"; }
inline FoundFx filterFx (std::shared_ptr<const fixed::Found> found, double prob)
{
    FoundFx F; F.name = "filter"; F.title = filterTitle (*found); F.prob = prob; F.est = { { "cutoff", 1.0 }, { "resonance", 1.0 } };
    const bool peak = found->peakDb >= fixed::peakShown; F.filter = std::move (found);
    Knob a; a.id = "filter.amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2; a.ownTip = true;
    a.tip = "Middle = as found. Left = take the filter out, far left = the sound before it: what the filter cut is lifted again, by up to 40 dB. What it cut deeper than that stays down, and so does hiss that came after the filter. Right = more of it, up to the filter twice.";
    a.tip2 = "How much of the found filter goes onto the second sound. Middle = the same curve as on the first sound. Left = not applied. Right = the curve twice.";
    Knob c; c.key = "cutoff"; c.id = "filter.cutoff"; c.label = "Cutoff"; c.neutral = 0; c.lo = 0; c.hi = 2; c.ownTip = true;
    c.tip = "Where the filter sits. Middle = as found. Left = lower, down to two octaves. Right = higher, up to two octaves. Away from the middle the found filter comes out and the moved one goes in.";
    c.tip2 = "Where the filter sits on the second sound. Middle = as found. Left = lower, down to two octaves. Right = higher, up to two octaves.";
    F.knobs = { a, c };
    if (peak)
    {
        Knob r; r.key = "resonance"; r.id = "filter.resonance"; r.label = "Resonance"; r.neutral = 0; r.lo = 0; r.hi = 2; r.ownTip = true;
        r.tip = "The peak at the filter's corner. Middle = as found. Left = lower, far left = none. Right = higher, up to twice as high.";
        F.knobs.push_back (r);
    }
    return F;
}
// The filter engine's verdict goes into the inspection and the detector's own guess about a filter goes out. A filter
// that moves is the moving-tone engine's, and the two groups never show side by side: both would work on the same
// filter. A sweep that is sure enough to show where the Sureness slider stands at first wins, and so does a fainter
// one that is at least as sure as the fixed filter. Otherwise the fixed filter wins, and the faint sweep is not shown
// at all. (The moving-tone engine has such a faint suspicion on 4 of 32 fixed filters from outside effects, 0.07 to
// 0.54 sure, where the filter engine is 0.90 to 0.98 sure; on the scoreboard's eight sweeps it is 0.68 to 1.00 sure.
// Until this rule any sweep won, and on those four sounds neither group was shown: Research/filter/both.py.)
inline void mergeFixed (Inspection& I, const fixed::Found& f, const moving::Found* mov = nullptr)
{
    const bool sweep = mov != nullptr && mov->found && mov->kind == "filtersweep";
    const bool sweepWins = sweep && (! f.found || mov->sure >= fixed::shownSure || mov->sure >= f.sure);
    I.prob["filter"] = f.found && ! sweepWins ? f.sure : level::sureNone;
    if (sweep && ! sweepWins) I.prob["filtersweep"] = level::sureNone;
}

// ---------------------------------------------------------------- the bend engine's group (Bend.h)
// DISTORTION, SATURATION: the curve the sound's values were pushed through, as the bend engine read it off the sound:
// DISTORTION where the loudest values were squeezed by 10 dB or more, SATURATION under that. One fader, the Amount, as
// a drive: on the sound itself 0 = the bend walked back (the squeezed levels spread out again, as far as the engine's
// cap allows), 1 = untouched (bit for bit), 2 = pushed as far again into the same curve; on a second sound 0 = none,
// 1 = its loudest values pushed as far into the curve as the first sound's were, 2 = as far again.
// Where the sound's tops were cut flat and the declipper rebuilt them (bend::addFlat), 0 on the sound itself is the
// rebuilt sound; a plain cut whose tops were rebuilt goes onto a second sound as a cut, that sound's peak as far past
// it as the first sound's rebuilt peak stands.
inline const char* bendName (const bend::Found& f) { return f.hard() ? "distortion" : "saturation"; }
inline FoundFx bendFx (std::shared_ptr<const bend::Found> found, double prob)
{
    const bool hard = found->hard(); const bend::Flat* A = found->flat != nullptr && found->flat->did ? found->flat.get() : nullptr;
    const bool plain = A != nullptr && (A->kind[0] == 1 || A->kind[1] == 1), back = A != nullptr && A->rebuilt, stays = plain && ! back && A->tab[0] != nullptr && A->tab[1] != nullptr, tested = A != nullptr && A->tested;
    FoundFx F; F.name = bendName (*found); F.title = titleFor (F.name); F.prob = prob; F.bend = std::move (found);
    Knob a; a.id = F.name + ".amount"; a.label = "Amount"; a.neutral = 0; a.lo = 0; a.hi = 2; a.ownTip = true;
    a.tip = plain && back ? "Middle = as found. Left = the cut-off tops come back: they are rebuilt from the sound around them, and what lies under the cut is not touched. Right = cut deeper, up to as far again."
          : stays ? std::string ("Middle = as found. Left does nothing here: the tops were cut off flat, and this sound is too ") + (tested ? "dense" : "long") + " for them to be rebuilt. Right = cut deeper, up to as far again."
          : back ? "Middle = as found. Left = the bend walked back: the squeezed levels are spread out again, and the tops that were pressed flat are rebuilt from the sound around them. Right = pushed further into the same curve, up to as far again."
          : hard ? "Middle = as found. Left = the bend walked back: the squeezed levels are spread out again, most of the way. Tops that were pressed flat stay flat: this sound gives too little to rebuild them from. Right = pushed further into the same curve, up to as far again."
                 : "Middle = as found. Left = the bend walked back: the squeezed levels are spread out again, most of the way (a little is left in, to be on the safe side). Right = pushed further into the same curve, up to as far again.";
    a.tip2 = plain && back ? "How far the second sound is pushed into the found cut. Middle = its loudest values as far past the cut as the first sound's rebuilt tops stand. Left = less far, far left = not at all. Right = further, up to as far again."
                   : "How far the second sound is pushed into the found curve. Middle = its loudest values as far as the first sound's were. Left = less far, far left = not at all. Right = further, up to as far again.";
    F.knobs = { a }; return F;
}
// The bend engine's verdict goes into the inspection and the detectors' own guesses about distortion and saturation go out.
inline void mergeBend (Inspection& I, const bend::Found& f)
{
    I.prob["distortion"] = f.found && f.hard() ? f.sure : level::sureNone;
    I.prob["saturation"] = f.found && ! f.hard() ? f.sure : level::sureNone;
}

// EXCITER is not shown. What an exciter adds are highs, and highs it made are not told from highs the sound has by
// itself or from an equaliser's lift: of 40 sounds through three outside exciters a reader taught for exactly this
// (Research/exciter) found 8, all of them heavy-handed ones, and it also found 2 of the same 40 sounds before the
// exciter and some clean sounds. The detectors' own guess did no better (one true find in eight, and false ones on
// filtered sounds), so it goes out. With two sounds, TONE MATCH carries the first sound's highs over, an exciter's
// among them.
inline void dropExciter (Inspection& I) { I.prob["exciter"] = level::sureNone; }

// What the engines added in builds 0.4.7 and 0.5.1 found on the sound (each may be missing: then the detectors' verdict and the old handling stay).
struct More
{
    std::shared_ptr<const sides::Found> sides;
    std::shared_ptr<const ring::Found> ring;
    std::shared_ptr<const crush::Found> crush;
    std::shared_ptr<const stutter::Found> stutter;
    std::shared_ptr<const fixed::Found> filter;
    std::shared_ptr<const bend::Found> bend;
};

// Settings of each found effect (estimate.estimate, without the slower sweep fit for now).
inline bool isNewFx (const std::string& f) { return f == "saturation" || f == "exciter" || f == "ringmod" || f == "filter" || f == "stutter" || f == "gatedrhythm"; }
// One cut for every effect: the Sureness slider. The six 0.2.0 additions needed a higher score while the detectors'
// own scores were shown; the verifier (Verify.h) now puts every effect's fixed cut at the slider's default.
inline double cutFor (const std::string&, double cut) { return cut; }
// The effects shown at a given sureness, in chain order.
inline std::vector<std::string> namesAbove (const Inspection& I, double cut)
{
    std::vector<std::string> n;
    for (const auto& f : chainOrder()) { auto p = I.prob.find (f); if (p != I.prob.end() && p->second > cutFor (f, cut)) n.push_back (f); }
    return n;
}
// A gated rhythm also reads as a deep tremolo at the same speed: keep only the gate.
inline void pruneFound (std::vector<FoundFx>& out)
{
    auto gr = std::find_if (out.begin(), out.end(), [] (const FoundFx& F) { return F.name == "gatedrhythm"; });
    if (gr == out.end()) return;
    const double r = get (gr->est, "rate", 0);
    out.erase (std::remove_if (out.begin(), out.end(), [r] (const FoundFx& F) { return F.name == "tremolo" && std::abs (get (F.est, "rate", 0) / std::max (r, 1e-6) - 1) < 0.2; }), out.end());
}
// only: estimate exactly these effects (whatever their score); otherwise all above `cut`.
// room: what the room engine found on the sound; with a room found, the reverb is that room (roomFx).
// tone: what the tone engine found on the sound; with reflections found, they are the COMB FILTER group (combFx).
// echo: what the echo engine found on the sound; with echoes found, they are the DELAY group (echoFx).
// lvl: what the level engine found on the sound; a movement found is the TREMOLO, AUTO PAN or GATED RHYTHM group (levelFx).
// swp: what the flanger engine found on the sound; a moving copy found is the FLANGER group (flangerFx).
// mov: what the moving-tone engine found on the sound; a moving filter found is the PHASER or FILTER SWEEP group (movingFx).
// cho: what the chorus engine found on the sound; voices found are the CHORUS group (chorusFx).
// more: what the engines of build 0.4.7 found: a delay or a widening between the sides is the HAAS WIDTH or STEREO WIDTH group (sidesFx),
// a tone the sound was multiplied by is the RING MOD group (ringFx), a grid of values or a hold is the BITCRUSH group (crushFx),
// slices played again are the STUTTER group (stutterFx); and those of build 0.5.1: a fixed filter's curve is the FILTER group (filterFx),
// a curve the values were pushed through is the DISTORTION or the SATURATION group (bendFx).
inline std::vector<FoundFx> estimate (const Inspection& I, const Stereo& y, double cut = 0.65, const std::set<std::string>* only = nullptr,
                                      const std::shared_ptr<const room::Found>& room = {}, const std::shared_ptr<const tone::Found>& tone = {},
                                      const std::shared_ptr<const echo::Found>& echo = {}, const std::shared_ptr<const level::Movement>& lvl = {},
                                      const std::shared_ptr<const sweep::Result>& swp = {}, const std::shared_ptr<const moving::Found>& mov = {},
                                      const std::shared_ptr<const chorus::Found>& cho = {}, const More& more = {})
{
    std::vector<FoundFx> out;
    (void) est::pinkProbe ((size_t) (6 * SR));      // built once before the fits run in parallel
    auto S = [&] (const std::string& f, const char* k, double d) { auto a = I.settings.find (f); if (a == I.settings.end()) return d; auto b = a->second.find (k); return b == a->second.end() ? d : b->second; };
    auto L = [&] (const std::string& f, const char* k, const std::string& d) { auto a = I.labels.find (f); if (a == I.labels.end()) return d; auto b = a->second.find (k); return b == a->second.end() ? d : b->second; };
    auto R = [&] (const std::string& f, double d) { auto a = I.reg.find (f); return a == I.reg.end() ? d : a->second; };
    for (const auto& f : chainOrder())
    {
        auto pit = I.prob.find (f); if (pit == I.prob.end()) continue;
        if (only ? ! only->count (f) : pit->second <= cutFor (f, cut)) continue;
        if (f == "reverb" && room != nullptr) { if (room->found) out.push_back (roomFx (room, pit->second)); continue; }        // the room engine ran: its room or none
        if (f == "comb") { if (tone != nullptr && ! tone->echoes.empty()) out.push_back (combFx (tone)); continue; }
        if (f == "delay") { if (echo != nullptr && echo->any()) out.push_back (echoFx (echo, pit->second)); continue; }
        if (lvl != nullptr && (f == "tremolo" || f == "autopan" || f == "gatedrhythm")) { if (lvl->found && f == levelName (*lvl)) out.push_back (levelFx (lvl, pit->second)); continue; }
        if (swp != nullptr && f == "flanger") { if (swp->found) out.push_back (flangerFx (swp, pit->second)); continue; }
        if (mov != nullptr && (f == "phaser" || f == "filtersweep")) { if (mov->found && f == mov->kind) out.push_back (movingFx (mov, pit->second)); continue; }
        if (cho != nullptr && f == "chorus") { if (cho->found) out.push_back (chorusFx (cho, pit->second)); continue; }
        if (more.sides != nullptr && (f == "haas" || f == "width")) { if (more.sides->found() && f == sidesName (*more.sides)) out.push_back (sidesFx (more.sides, pit->second)); continue; }
        if (more.ring != nullptr && f == "ringmod") { if (more.ring->found) out.push_back (ringFx (more.ring, pit->second)); continue; }
        if (more.crush != nullptr && f == "bitcrush") { if (more.crush->found) out.push_back (crushFx (more.crush, pit->second)); continue; }
        if (more.stutter != nullptr && f == "stutter") { if (more.stutter->found) out.push_back (stutterFx (more.stutter, pit->second)); continue; }
        if (more.filter != nullptr && f == "filter") { if (more.filter->found) out.push_back (filterFx (more.filter, pit->second)); continue; }
        if (more.bend != nullptr && (f == "distortion" || f == "saturation")) { if (more.bend->found && f == bendName (*more.bend)) out.push_back (bendFx (more.bend, pit->second)); continue; }
        FoundFx F; F.name = f; F.title = titleFor (f); F.prob = pit->second; F.knobs = knobsFor (f); Params& e = F.est;
        if (f == "haas") e["lag"] = (double) hand::gccPhat (y, 30.0).lag;
        else if (f == "width") e["amount"] = std::clamp (S (f, "amount", 0.5), 0.05, 1.2);
        else if (f == "autopan") { const auto s = est::fitSineGrid (est::panLane (y), 200, 0.1, 4); e = { { "rate", s.rate }, { "amp_db", s.amp }, { "phase", s.phase } }; }
        else if (f == "tremolo") { const double r0 = std::clamp (R (f, 5), 0.3, 20.0); const auto s = est::fitSineGrid (est::tremLane (y), 200, r0 / 1.5, r0 * 1.5); e = { { "rate", s.rate }, { "amp_db", std::min (s.amp, 24.0) }, { "phase", s.phase } }; }
        else if (f == "bitcrush") { const auto nl = hand::nonlinear (y); e = { { "bits", nl.bits > 0 ? std::round (nl.bits) : 8.0 }, { "down", std::max (1.0, std::round (nl.hold)) } }; }
        else if (f == "distortion")
        {
            const Vec m = y.mid(); const double pk = fx::peakOf (m); size_t c = 0; for (double v : m) c += std::abs (v) > 0.9 * pk;
            e = { { "top", (double) c / (double) std::max<size_t> (1, m.size()) }, { "kind", L (f, "kind", "tanh") == "tanh" ? 1.0 : 0.0 }, { "drive", S (f, "drive", 8) }, { "drive_scale", 1.0 } };
        }
        else if (f == "reverb") { int n = 0; const double t = est::rt60FromDecays (y, n); e = { { "rt60", std::clamp (t > 0 ? t : S (f, "rt60", 1.5), 0.3, 5.0) }, { "mix", std::clamp (S (f, "mix", 0.35), 0.05, 0.8) } }; }
        else if (f == "saturation") e = { { "drive", std::clamp (S (f, "drive", 8), 0.0, 30.0) }, { "tone", std::clamp (S (f, "tone", 0.5), 0.0, 1.0) }, { "kind", L (f, "kind", "tape") == "tape" ? 1.0 : 0.0 } };
        else if (f == "exciter") e = { { "amount", std::clamp (S (f, "amount", 0.3), 0.05, 1.0) }, { "freq", std::clamp (S (f, "freq", 3000), 800.0, 12000.0) } };
        else if (f == "ringmod") e = { { "carrier", std::clamp (S (f, "carrier", 200), 10.0, 4000.0) }, { "mix", std::clamp (S (f, "mix", 0.8), 0.1, 1.0) }, { "drift", std::clamp (S (f, "drift", 0), 0.0, 0.5) }, { "phase", 0 } };
        else if (f == "filter")
        {
            const std::string t = L (f, "type", "low");
            e = { { "type", t == "band" ? 0.0 : t == "high" ? 1.0 : 2.0 }, { "cutoff", std::clamp (S (f, "cutoff", 1500), 30.0, 18000.0) }, { "resonance", std::clamp (S (f, "resonance", 0.707), 0.3, 8.0) }, { "slope", L (f, "slope", "12") == "24" ? 1.0 : 0.0 } };
        }
        else if (f == "stutter") e = { { "slice_ms", std::clamp (S (f, "slice_ms", 120), 20.0, 600.0) }, { "repeats", std::stod (L (f, "repeats", "4")) }, { "prob", std::clamp (S (f, "prob", 0.4), 0.05, 1.0) }, { "seed", 3 } };
        else if (f == "gatedrhythm") e = { { "rate", std::clamp (S (f, "rate", 8), 0.5, 32.0) }, { "duty", std::clamp (S (f, "duty", 0.5), 0.05, 1.0) }, { "depth", std::clamp (S (f, "depth", 1), 0.1, 1.0) }, { "phase", 0 } };
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
    if (! only) pruneFound (out);
    return out;
}

// Take the six 0.2.0 effects (probability, settings, speed guess) from the 27-effect inspection.
inline void mergeNew (Inspection& I, const Inspection& I27)
{
    I.fx27 = I27.fx; I.win27 = I27.win;        // for the verifier (Verify.h)
    for (const char* f : { "saturation", "exciter", "ringmod", "filter", "stutter", "gatedrhythm" })
    {
        auto p = I27.prob.find (f); if (p != I27.prob.end()) I.prob[f] = p->second;
        auto s = I27.settings.find (f); if (s != I27.settings.end()) I.settings[f] = s->second;
        auto l = I27.labels.find (f); if (l != I27.labels.end()) I.labels[f] = l->second;
        auto r = I27.reg.find (f); if (r != I27.reg.end()) I.reg[f] = r->second;
    }
    // When one of the six is there, the 12-effect detector can mistake it for a relative (a static filter for a
    // filter sweep, a gated rhythm for bitcrush). The 27-effect detector knows both: drop what it clearly rules out.
    bool anyNew = false; for (auto& [f, v] : I.prob) if (isNewFx (f) && v > 0.8) anyNew = true;
    if (anyNew)
        for (auto& [f, v] : I.prob)
            if (! isNewFx (f)) { auto q = I27.prob.find (f); if (q != I27.prob.end() && q->second < 0.15) v = std::min (v, q->second); }
}

// ---------------------------------------------------------------- the MOVEMENT group (Movement.h)
// Always there, whatever was found: five faders for what moves in a sound by itself, with no effect on it. Each runs
// 0 .. 2: far left = flat, middle = the sound's own (nothing is done), far right = the sound's own movement twice (one
// sound) or the first sound's movement in place of the second sound's own (two sounds). Not a found effect, so the
// group shows no sureness. own: what moves in the sound worked on (one sound: the first; two sounds: the second);
// other: in the first sound, with two sounds. Either may still be missing (not measured yet): the faders then do nothing.
// What an engine shows as an effect of its own stays with that group: the first sound is measured without the level
// movement and the moving filter that are shown (movementBase), so a tremolo or a filter sweep is not counted twice.
// One sound: the faders of the found effects come first, the movement faders after them (before a found stutter, whose
// repeats close up or spread the sound). Two sounds: the movement comes first and the first sound's effects go on top,
// as they sit on the first sound itself.
inline FoundFx movementFx (std::shared_ptr<const move::Found> own, std::shared_ptr<const move::Found> other)
{
    FoundFx F; F.name = "movement"; F.title = titleFor ("movement"); F.prob = -1.0; F.moveOwn = std::move (own); F.moveOther = std::move (other);
    F.est = { { "pitch", 1.0 }, { "loudness", 1.0 }, { "tone", 1.0 }, { "stereo", 1.0 }, { "spectral", 1.0 } };
    auto K = [] (const char* key, const char* label, std::string one, std::string two)
    {
        Knob k; k.key = key; k.id = std::string ("movement.") + key; k.label = label; k.neutral = 0; k.lo = 0; k.hi = 2; k.ownTip = true; k.tip = std::move (one); k.tip2 = std::move (two); return k;
    };
    F.knobs = {
        K ("pitch", "Pitch",
           "How the sound's pitch moves over time: a glide, a pass-by, a vibrato. Left = the movement taken out, the pitch stands still. Middle = the sound's own. Right = the movement twice. "
           "It works like tape speed, so nothing warbles; the length stays. It needs a pitch that can be followed: on noise or on several things at once this fader does nothing. A sound of more than ten minutes is not read.",
           "Left = the second sound's own pitch movement taken out. Middle = its own. Right = the first sound's pitch movement in its place, at the same times. "
           "It works like tape speed, so nothing warbles; the length stays. It needs a pitch that can be followed: on noise or on several things at once there is nothing to take out or to bring. A sound of more than ten minutes is not read."),
        K ("loudness", "Loudness",
           "How the sound's level moves over time: a swell, a fade, a pass-by. Left = held even. Middle = the sound's own. Right = the movement twice. Single hits keep their edges.",
           "Left = the second sound's level held even. Middle = its own. Right = the first sound's level movement in its place, at the same times."),
        K ("tone", "Tone",
           "How the sound's tone moves over time: getting brighter or darker, opening up. Left = the tone held still. Middle = the sound's own. Right = the movement twice. The overall colour itself is the Spectral fader.",
           "Left = the second sound's tone held still. Middle = its own. Right = the first sound's tone movement in its place, at the same times. The overall colour itself is the Spectral fader."),
        K ("stereo", "Stereo",
           "How the sound moves between left and right, and how its width changes. Left = held still where it sits on average. Middle = the sound's own. Right = the movement twice.",
           "Left = the second sound held still where it sits on average. Middle = its own. Right = the first sound's movement between the sides and in width in its place, at the same times."),
        K ("spectral", "Spectral",
           "The sound's overall colour: its long-term spectrum in broad strokes. Left = brought to a neutral colour, the same power in every octave. Middle = the sound's own. Right = its own colour once more.",
           "The sound's overall colour: its long-term spectrum in broad strokes. Left = the second sound brought to a neutral colour. Middle = its own. Right = the first sound's colour (Tone Match does the same in detail).") };
    return F;
}
// The five fader values (0 .. 2, 1 = nothing is done) out of the fader values by id.
inline move::Faders movementFaders (const std::map<std::string, double>& faders)
{
    auto v = [&] (const char* id) { auto it = faders.find (id); return it == faders.end() ? 1.0 : std::clamp (it->second, 0.0, 2.0); };
    move::Faders f; f.pitch = v ("movement.pitch"); f.loudness = v ("movement.loudness"); f.tone = v ("movement.tone"); f.stereo = v ("movement.stereo"); f.spectral = v ("movement.spectral");
    return f;
}
// Is this group one whose effect counts as the effect's own movement, not as the sound's: a level movement (TREMOLO,
// AUTO PAN, GATED RHYTHM) or a moving filter (PHASER, FILTER SWEEP)?
inline bool movesByItself (const FoundFx& F) { return F.level != nullptr || F.moving != nullptr; }
// Which of those are among the groups shown, as one word: what movementBase takes out ("" = nothing).
inline std::string movementBaseKey (const std::vector<FoundFx>& found) { std::string k; for (auto& F : found) if (movesByItself (F)) k += F.name + ";"; return k; }

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
    if (F.name == "tonematch") { auto it = faders.find ("tonematch.detail"); p["detail"] = matchDetailAt (F, it == faders.end() ? 1.0 : it->second); }
    if (F.name == "flanger" && F.sweep == nullptr && p["dmin"] >= p["dmax"]) p["dmin"] = p["dmax"] * 0.5;
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
// TONE MATCH comes last, on what the other effects have made of the target: the result then has the first sound's
// long-term tone whatever went on before. shown: if given, receives that group with the curves it used (for the
// picture and Export IR).
// The found crusher and the found stutter were the last things the first sound went through: anything after a crusher
// takes the values off its grid, anything after a stutter makes its repeats unequal, and neither would then have been
// read. So here too the crusher comes after the other effects, and the stutter at the very end (after TONE MATCH,
// whose filter would blur the cuts, and which a stutter hardly changes).
inline Stereo applyTo (const Stereo& target, const std::vector<FoundFx>& found, const std::map<std::string, double>& faders, const std::function<bool()>& cancelled = {},
                       FoundFx* shown = nullptr)
{
    Stereo y = target;
    for (auto& F : found)       // what moves in this sound by itself, and the first sound's movement in its place: before the first sound's effects
    {
        if (F.name != "movement" || F.moveOwn == nullptr || (cancelled && cancelled())) continue;
        const move::Faders mf = movementFaders (faders);
        if (mf.any() && F.moveOwn->n == (long) y.size()) { Stereo r = move::apply (y, *F.moveOwn, F.moveOther.get(), mf, cancelled); if (r.size() == y.size()) y = std::move (r); }
    }
    for (auto& F : found)
    {
        if (cancelled && cancelled()) break;
        if (F.name == "tonematch" || F.name == "movement" || F.crush != nullptr || F.stutter != nullptr) continue;
        if (F.room != nullptr)      // the found room, rebuilt from its numbers, on this sound
        {
            const Params p = scaled (F, faders);
            Stereo r = room::onto (y, *F.room, amountOf (F, faders), get (p, "length", 1.0), get (p, "bright", 1.0), get (p, "width", 1.0), cancelled);
            if (r.size() == y.size()) y = std::move (r);
            continue;
        }
        if (F.name == "comb")       // the found reflections on this sound
        {
            const Vec g = combGain (F, scaled (F, faders), amountOf (F, faders), true);
            Stereo r = tone::gainFilter (y, g, g, cancelled); if (r.size() == y.size()) y = std::move (r);
            continue;
        }
        if (F.echo != nullptr)      // the found echoes on this sound
        {
            const double s = amountOf (F, faders);
            if (s > 1e-9) { Stereo r = echo::render (y, *F.echo, echoShape (scaled (F, faders), s), false, cancelled); if (r.size() == y.size()) y = std::move (r); }
            continue;
        }
        if (F.level != nullptr)     // the found level movement on this sound
        {
            const double s = amountOf (F, faders);
            if (s > 1e-9) { Vec gl, gr; level::ontoDb (*F.level, y.size(), s, levelSpeed (scaled (F, faders)), gl, gr); y = level::withDb (y, gl, gr); }
            continue;
        }
        if (F.sweep != nullptr)     // the found moving copy on this sound
        {
            const double s = amountOf (F, faders); const Params p = scaled (F, faders);
            if (s > 1e-9) y = sweep::apply (y, *F.sweep, std::min (s, 2.0), sweepSpeed (p), sweepReach (p), sweepFeedback (p));
            continue;
        }
        if (F.moving != nullptr)    // the found moving tone curve on this sound, at the same times
        {
            const double s = amountOf (F, faders);
            if (s > 1e-9) y = moving::apply (y, *F.moving, std::min (s, 2.0), movingSpeed (scaled (F, faders)), false);
            continue;
        }
        if (F.chorus != nullptr)    // the found voices on this sound
        {
            const double s = amountOf (F, faders); const Params p = scaled (F, faders);
            if (s > 1e-9) y = chorus::apply (y, *F.chorus, std::min (s, 2.0), chorusSpeed (p), chorusDepth (p), false);
            continue;
        }
        if (F.sides != nullptr)     // the found delay or widening on this sound: each side through its own filter
        {
            const double s = amountOf (F, faders);
            if (s > 1e-9) y = sides::onto (y, *F.sides, std::min (s, 2.0));
            continue;
        }
        if (F.ring != nullptr)      // this sound multiplied by the found gain
        {
            const double s = amountOf (F, faders);
            if (s > 1e-9) y = ring::onto (y, *F.ring, std::min (s, 2.0), ringCarrier (scaled (F, faders)));
            continue;
        }
        if (F.bend != nullptr)      // this sound pushed through the found curve
        {
            const double s = amountOf (F, faders);
            if (s > 1e-9) y = bend::onto (y, *F.bend, std::min (s, 2.0), cancelled);
            continue;
        }
        if (F.filter != nullptr)    // this sound through the found filter
        {
            const double s = amountOf (F, faders); const Params p = scaled (F, faders);
            if (s > 1e-9) y = fixed::onto (y, *F.filter, std::min (s, 2.0), fixed::cutoffOctaves (get (p, "cutoff", 1.0)), get (p, "resonance", 1.0), cancelled);
            continue;
        }
        y = applyAmount (F.name, y, scaled (F, faders), amountOf (F, faders));
    }
    for (auto& F : found)       // this sound through the found crusher
    {
        if (F.crush == nullptr || (cancelled && cancelled())) continue;
        const double s = amountOf (F, faders); const Params p = scaled (F, faders);
        if (s > 1e-9) y = crush::onto (y, *F.crush, std::min (s, 2.0), crushSteps (F, p), crushHold (p));
    }
    for (auto& F : found)
    {
        if (F.name != "tonematch" || (cancelled && cancelled())) continue;
        const double s = amountOf (F, faders), detail = get (scaled (F, faders), "detail", matchDetail);
        const tone::Spectra sy = tone::spectra (y, cancelled); auto [gl, gr] = matchCurves (F, sy, detail); if (gl.empty()) continue;
        if (shown != nullptr)
        {
            *shown = F; shown->curveL = gl; shown->curveR = gr; const double base = matchDetailAt (F, 1.0);
            if (std::abs (detail - base) < 1e-9) { shown->foundL = gl; shown->foundR = gr; }
            else { auto [fl, fr] = matchCurves (F, sy, base); shown->foundL = std::move (fl); shown->foundR = std::move (fr); }
        }
        for (auto& v : gl) v *= s; for (auto& v : gr) v *= s;
        Stereo r = tone::gainFilter (y, gl, gr, cancelled); if (r.size() == y.size()) y = std::move (r);
    }
    for (auto& F : found)       // the found slices of this sound played again, at the same times
    {
        if (F.stutter == nullptr || (cancelled && cancelled())) continue;
        const double s = amountOf (F, faders);
        if (s > 1e-9) y = stutter::onto (y, *F.stutter, std::min (s, 2.0), stutterSlice (scaled (F, faders)));
    }
    limit (y);
    return y;
}

// One sound: take the found effects out (left), leave as is (middle) or double (right).
inline bool canRemove (const std::string& f) { return f != "reverb" && f != "distortion" && f != "bitcrush" && ! isNewFx (f); }
inline bool canRemove (const FoundFx& F) { return F.room != nullptr || F.tone != nullptr || F.echo != nullptr || F.level != nullptr || F.sweep != nullptr || F.moving != nullptr || F.chorus != nullptr || F.sides != nullptr || F.ring != nullptr || (F.crush != nullptr && F.crush->soft) || F.filter != nullptr || F.bend != nullptr || canRemove (F.name); }      // the room engine takes its reverb out, the tone engine its reflections, the echo engine its echoes, the level engine its movement, the flanger engine its moving copy, the moving-tone engine its curve, the chorus engine its voices, the sides engine its delay or widening, the ring engine its tone, the crush engine a crusher that mixed plain sound in, the filter engine its curve, the bend engine its bend (as far as the sound still says how)
// The found effect turned down to s (0 = out as far as this effect allows, 1 = untouched).
inline Stereo takeOut (const FoundFx& F, const Stereo& in, double s)
{
    Stereo y = in; const std::string& f = F.name;
    if (f == "haas")
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
    return y;
}
// The found crusher is worked first: it was the last thing the sound went through, and its engine's answer belongs to
// the sound as it was found. The found bend comes right after it for the same reason: its curve is told by the height
// of each value in the sound as found, and a bend that the wave still shows had little after it. The found stutter is
// worked last: with fewer plays the sound closes up, and the other engines' answers belong to the sound's own times.
// limited: the result is held under full scale (not wanted where it is only measured).
inline Stereo reshape (const Stereo& src, const std::vector<FoundFx>& found, const std::map<std::string, double>& faders, const std::function<bool()>& cancelled = {}, bool limited = true)
{
    Stereo y = src;
    for (const FoundFx& F : found)
    {
        if (F.crush == nullptr || (cancelled && cancelled())) continue;
        // a crusher that mixed plain sound in: the sound before it, and from there to where the faders are; a bare one: more of it only; the crush engine's answer belongs to this very sound (same length)
        const Params p = scaled (F, faders);
        if (F.crush->n == y.size()) y = crush::own (y, *F.crush, std::clamp (amountOf (F, faders), 0.0, 2.0), crushSteps (F, p), crushHold (p));
    }
    for (const FoundFx& F : found)
    {
        if (F.bend == nullptr || (cancelled && cancelled())) continue;
        // the bend walked back, or the sound pushed further into the found curve; the bend engine's answer belongs to this very sound (same length)
        if (F.bend->n == y.size()) y = bend::own (y, *F.bend, std::clamp (amountOf (F, faders), 0.0, 2.0), cancelled);
    }
    for (auto it = found.rbegin(); it != found.rend(); ++it)      // undo in reverse chain order
    {
        if (cancelled && cancelled()) break;
        const FoundFx& F = *it; const double s = amountOf (F, faders); const bool moved = settingsMoved (F, faders);
        if (F.crush != nullptr || F.bend != nullptr || F.stutter != nullptr || F.name == "movement") continue;
        if (! moved && std::abs (s - 1.0) < 1e-9) continue;
        if (F.room != nullptr)
        {
            // the sound's own reverb turned down or up and reshaped; the room engine's answer belongs to this very sound (same length)
            const Params p = scaled (F, faders);
            if (F.room->picture.frames == room::frames (y.size()))
            {
                Stereo r = roomFader (y, *F.room, s, get (p, "length", 1.0), get (p, "bright", 1.0), get (p, "width", 1.0), cancelled);
                if (r.size() == y.size()) y = std::move (r);
            }
        }
        else if (F.name == "comb")
        {
            // the found reflections out and those the faders give in, as one filter
            const Vec g = combGain (F, scaled (F, faders), s, false);
            Stereo r = tone::gainFilter (y, g, g, cancelled); if (r.size() == y.size()) y = std::move (r);
        }
        else if (F.echo != nullptr)
        {
            // the found echoes out and those the faders give in, as one filter per channel
            Stereo r = echo::render (y, *F.echo, echoShape (scaled (F, faders), s), true, cancelled); if (r.size() == y.size()) y = std::move (r);
        }
        else if (F.level != nullptr)
        {
            // the found movement out and what the faders give in, as one gain per side; the level engine's answer belongs to this very sound (same length)
            if (F.level->n == y.size()) { Vec gl, gr; level::ownDb (*F.level, y.size(), s, levelSpeed (scaled (F, faders)), gl, gr); y = level::withDb (y, gl, gr); }
        }
        else if (F.sweep != nullptr)
        {
            // the found copy out (the exact inverse moving filter) and what the faders give in; the flanger engine's answer belongs to this very sound (same length)
            if (F.sweep->n == y.size())
            {
                const Params p = scaled (F, faders);
                y = moved || s > 1.0 ? sweep::render (y, *F.sweep, true, s, sweepSpeed (p), sweepReach (p), sweepFeedback (p)) : sweep::apply (y, *F.sweep, s - 1.0);
            }
        }
        else if (F.moving != nullptr)
        {
            // the found curve off and what the faders give on, as one gain per bin and moment; the moving-tone engine's answer belongs to this very sound (same length)
            if (F.moving->n == y.size()) y = moving::apply (y, *F.moving, std::min (s, 2.0) - 1.0, movingSpeed (scaled (F, faders)), true);
        }
        else if (F.chorus != nullptr)
        {
            // the found voices out (each range's own inverse) and what the faders give in; the chorus engine's answer belongs to this very sound (same length)
            if (F.chorus->n == y.size()) { const Params p = scaled (F, faders); y = chorus::apply (y, *F.chorus, std::min (s, 2.0) - 1.0, chorusSpeed (p), chorusDepth (p), true); }
        }
        else if (F.sides != nullptr)
        {
            // the part of the sound that the found pair of filters (or the copy at the sides) explains, changed; the rest stays; the sides engine's answer belongs to this very sound (same length)
            if (F.sides->n == y.size()) y = sides::own (y, *F.sides, std::clamp (s, 0.0, 2.0));
        }
        else if (F.ring != nullptr)
        {
            // the sound without the found gain, times the difference between the gain the faders give and the found one, added; the ring engine's answer belongs to this very sound (same length)
            if (F.ring->n == y.size()) y = ring::own (y, *F.ring, std::clamp (s, 0.0, 2.0), ringCarrier (scaled (F, faders)));
        }
        else if (F.filter != nullptr)
        {
            // the found curve out and the one the faders give in, as one filter; the filter engine's answer belongs to this very sound (same length)
            if (F.filter->n == y.size()) { const Params p = scaled (F, faders); y = fixed::own (y, *F.filter, std::clamp (s, 0.0, 2.0), fixed::cutoffOctaves (get (p, "cutoff", 1.0)), get (p, "resonance", 1.0), cancelled); }
        }
        else if (F.name == "tonematch") {}      // two sounds only
        else if (canRemove (F.name))
        {
            // settings changed: take the found effect out, then put it back with the new settings at strength s
            if (moved) y = applyAmount (F.name, takeOut (F, y, 0.0), scaled (F, faders), s);
            else if (s < 1.0) y = takeOut (F, y, s);
            else y = applyAmount (F.name, y, F.est, s - 1.0);
        }
        else
        {
            // cannot be taken out yet: more of it (right half), and settings changes as "the difference they make"
            if (moved)
            {
                const Stereo a = applyAmount (F.name, y, F.est, 1.0), b = applyAmount (F.name, y, scaled (F, faders), 1.0);
                for (size_t i = 0; i < y.size(); ++i) { y.l[i] += 0.5 * (b.l[i] - a.l[i]); y.r[i] += 0.5 * (b.r[i] - a.r[i]); }
            }
            if (s > 1.0) y = applyAmount (F.name, y, scaled (F, faders), s - 1.0);
        }
    }
    for (const FoundFx& F : found)
    {
        // what moves in the sound by itself, after the effects' faders; what was measured belongs to this very sound (same length)
        if (F.name != "movement" || F.moveOwn == nullptr || (cancelled && cancelled())) continue;
        const move::Faders mf = movementFaders (faders);
        if (mf.any() && F.moveOwn->n == (long) y.size()) { Stereo r = move::apply (y, *F.moveOwn, nullptr, mf, cancelled); if (r.size() == y.size()) y = std::move (r); }
    }
    for (const FoundFx& F : found)
    {
        if (F.stutter == nullptr || (cancelled && cancelled())) continue;
        // more plays of each slice over what follows, or fewer with the sound closed up; the stutter engine's answer belongs to this very sound (same length)
        if (F.stutter->n == y.size()) y = stutter::own (y, *F.stutter, std::clamp (amountOf (F, faders), 0.0, 2.0), stutterSlice (scaled (F, faders)));
    }
    if (limited) limit (y);
    return y;
}

// The first sound as its movement is measured: without the level movement and the moving filter that are shown as
// effects of their own (see movementFx). The sound itself when none is shown.
inline Stereo movementBase (const Stereo& src, const std::vector<FoundFx>& found, const std::function<bool()>& cancelled = {})
{
    std::vector<FoundFx> out; std::map<std::string, double> zero;
    for (auto& F : found) if (movesByItself (F)) { out.push_back (F); zero[F.name + ".amount"] = 0.0; }
    return out.empty() ? src : reshape (src, out, zero, cancelled, false);
}

// An echo found on a sound that is multiplied by a tone. A sound that repeats itself in time (a loop, a bar played
// twice) is told from an echo by how its repeats differ, and a ring modulation changes just that: every pitch of the
// sound is split in two. The ring engine gives the sound without the tone, so the echo engine looks at that sound as
// well. An echo that was there before the tone is found there too, at the same time, and there the sound is as the
// echo engine's rule knows sounds: the less sure of the two verdicts counts. An echo that is not found there at all was
// put on after the tone (dividing by the tone's gain tears it apart): the verdict on the sound as it is stays.
// Returned: the second verdict, or -1 when there was nothing to look at. The echoes that are shown and worked with
// are still the ones found on the sound as it is. Costs one more look of the echo engine, and only when a tone and an
// echo are both found.
// Measured with the app's own code (Research/ring/rc1.py: 16 clips, a tone and an echo put on in both orders at
// several levels, 112 sounds): the echo engine looks again on 48 of them, and on none does the DELAY group come or go
// at the default Sureness; the largest drop of an echo that was put on is 0.98 to 0.71 (still shown). On the
// scoreboard the one DELAY group on a sound that has a tone only (a loop that repeats every half second) goes
// (0.87 to 0.52). Not done: an echo under a tone that the echo engine sees clearly only without the tone (one clip
// of the 112: 0.20 on the sound, 0.99 without) stays hidden, since the echoes that are worked with are the ones
// measured on the sound as it is.
inline double recheckEchoWithout (Inspection& I, const std::shared_ptr<const echo::Found>& echo, const More& more, const echo::Stop& stop = {})
{
    if (more.ring == nullptr || ! more.ring->found || echo == nullptr || ! echo->any()) return -1.0;
    auto p = I.prob.find ("delay"); if (p == I.prob.end()) return -1.0;
    Stereo x; x.l.assign (more.ring->xl.begin(), more.ring->xl.end()); x.r.assign (more.ring->xr.begin(), more.ring->xr.end());
    if (x.size() == 0 || x.l.size() != x.r.size()) return -1.0;
    const echo::Found e = echo::analyse (x, stop); if (stop && stop()) return -1.0;
    const double s = e.any() ? e.sure : echo::sureNone; bool same = false;
    for (int c = 0; c < 2; ++c) for (int d = 0; d < 2; ++d) if (echo->ch[c].found && e.ch[d].found && std::abs (echo->ch[c].rec.ms - e.ch[d].rec.ms) <= 2.0) same = true;
    if (same) p->second = std::min (p->second, s);
    return s;
}

// A room found on a sound that shows other effects as well. A train of echoes dies away as a room does; the copies
// of a flanger or a chorus, a filter that sweeps, one side a few milliseconds behind the other: each of them can look
// like a room to the room engine. So the room must still be there with those effects taken out. Echoes the echo
// engine finds, and what the app shows at the default Sureness and can take out cleanly (level movement, flanger,
// phaser, filter sweep, chorus, a delay or a widening between the sides, each by its engine) are taken out,
// the room engine looks at that sound as well, and the less sure of its two verdicts counts (returned: the second
// one, or -1 when there was nothing to look at; what: the groups that were taken out). The room that is shown and
// worked with is still the one found on the sound as it is. Costs one render of those groups and one more look of
// the room engine (on a sound of more than 9 s: the look in pieces, roomInPieces), and only when a room and one of
// those effects are both found.
// Measured with the app's own code (Research/room/bench/recheck_eval.py): on 912 sounds with one other effect and no
// reverb the REVERB groups made by the effect go from 50 to 26 (Haas 12 -> 4, delay 15 -> 8, auto pan 6 -> 3); of 340
// sounds with a real reverb that is shown, 78 are looked at again and one loses its group. On the scoreboard's 136
// cases the false REVERB groups go from 16 to 13.
// With the room model of 0.4.6 (which reads fewer rooms into such sounds to begin with): on the 912 sounds REVERB
// shows on 96 before this look and on 82 after it (0.4.5: 119 and 93; Haas 9 -> 3, delay 11 -> 8); of 334 sounds
// with a real reverb that is shown, 85 are looked at again and two lose their group; scoreboard: 9.
// (Since 0.4.2 this was done for echoes alone: on the scoreboard's 16 sounds with a delay the room engine showed a
// REVERB group on 5, only one of which has a room of its own, and the look without the echoes took 3 of the 4 away
// and added none; Research/echo/roomcheck.py, roomcheck2.py.)
constexpr double roomWithoutCut = 0.65;
inline double recheckRoomWithout (Inspection& I, const Stereo& y, const std::shared_ptr<const echo::Found>& echo, const std::shared_ptr<const level::Movement>& lvl,
                                  const std::shared_ptr<const sweep::Result>& swp, const std::shared_ptr<const moving::Found>& mov, const std::shared_ptr<const chorus::Found>& cho,
                                  const More& more, const room::Net& net, const room::Net* second, const room::Stop& stop = {}, std::string* what = nullptr)
{
    auto r = I.prob.find ("reverb"); if (r == I.prob.end() || ! (r->second > roomNone)) return -1.0;
    std::set<std::string> only;
    for (const char* f : { "tremolo", "autopan", "gatedrhythm", "flanger", "phaser", "filtersweep", "chorus", "haas", "width" }) { auto p = I.prob.find (f); if (p != I.prob.end() && p->second > roomWithoutCut) only.insert (f); }
    if (echo != nullptr && echo->any() && I.prob.count ("delay")) only.insert ("delay");      // echoes: whenever the echo engine finds any
    if (only.empty()) return -1.0;
    std::vector<FoundFx> out; std::map<std::string, double> faders;
    for (FoundFx& F : estimate (I, y, roomWithoutCut, &only, {}, {}, echo, lvl, swp, mov, cho, more))
        if (F.echo != nullptr || F.level != nullptr || F.sweep != nullptr || F.moving != nullptr || F.chorus != nullptr || F.sides != nullptr || F.name == "haas")
        {
            faders[F.name + ".amount"] = 0.0; if (what != nullptr) *what += (what->empty() ? "" : ",") + F.name;
            out.push_back (std::move (F));
        }
    if (out.empty()) return -1.0;
    const Stereo dry = reshape (y, out, faders, stop); if (dry.size() != y.size() || (stop && stop())) return -1.0;
    double s;
    if (room::frames (dry.size()) > roomWholeFrames) { s = roomInPieces (dry, net, second, stop); if (s < 0.0) return -1.0; }     // a long sound: in pieces, as its first verdict
    else { const room::Found f = room::find (dry, net, second, stop); if (f.stopped) return -1.0; s = roomSureness (f); }
    r->second = std::min (r->second, s); return s;
}

} // namespace dr
