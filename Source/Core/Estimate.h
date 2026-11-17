#pragma once
// Reading each found effect's settings from the source sound (port of estimate.py, revest.py), the controls
// that take an effect back out of the source (uneffect.py, "do its best"), and the fader layout: per effect an
// Amount fader plus its most useful settings. Reverb, comb filter, tone match and delay are the work of their
// own engines (RoomEngine.h, Tone.h, Echo.h); their groups are built here.

#include "Echo.h"
#include "Level.h"
#include "Sweep.h"
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
    Vec curveL, curveR;                        // TONE MATCH, for the picture and Export IR: the curves of the last render (left, right; Amount in the middle)
    Vec foundL, foundR;                        // the same with Detail in the middle too
};

inline std::string titleFor (const std::string& f)
{
    static const std::map<std::string, std::string> t { { "flanger", "FLANGER" }, { "chorus", "CHORUS" }, { "phaser", "PHASER" }, { "haas", "HAAS WIDTH" },
        { "width", "STEREO WIDTH" }, { "delay", "DELAY" }, { "reverb", "REVERB" }, { "distortion", "DISTORTION" }, { "bitcrush", "BITCRUSH" },
        { "filtersweep", "FILTER SWEEP" }, { "tremolo", "TREMOLO" }, { "autopan", "AUTO PAN" },
        { "saturation", "SATURATION" }, { "exciter", "EXCITER" }, { "ringmod", "RING MOD" }, { "filter", "FILTER" }, { "stutter", "STUTTER" }, { "gatedrhythm", "GATED RHYTHM" },
        { "comb", "COMB FILTER" }, { "tonematch", "TONE MATCH" } };
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
// How sure it is that there is a room on the sound, 0 .. 1: what the Sureness slider compares. Fitted on the
// scoreboard's 240 sounds with reverb against 368 without (the same sounds dry, and with tone effects only;
// Research/room/train/sure_stats.py and sure_fit2.py with PRIOR=0.62), as a probability for a sound that is somewhat
// more likely to carry reverb than not (62 in 100). Most of the answer is whether the engine finds a room at all (it
// does on 96 % of the sounds with reverb and on 34 % of those without); among the found ones, more cells that are
// clearly room and more bands with all-room moments make it surer. At the slider's 65 % the reverb then shows on
// 86 % of the sounds that have one, on 17 % of the dry ones and on 12 % of those with a tone effect only
// (cross-validated); the "dry" ones it shows on are mostly recorded loops and recordings, which may carry a room of
// their own. The numbers belong to the model in Resources (0.4.1: first look unet_w, second look unet_wp2) and are
// fitted again with every new model.
// roomNone: what is left when the engine finds no room (then about 9 in 10 sounds have none).
constexpr double roomSure0 = 2.100, roomSureDeep = 1.215, roomSureBands = 2.264, roomNone = 0.09;
inline double roomSureness (const room::Found& f)
{
    if (! f.found) return roomNone;
    const RoomEvidence e = roomEvidence (f);
    return 1.0 / (1.0 + std::exp (-(roomSure0 + roomSureDeep * std::log10 (e.deep + 0.003) + roomSureBands * (double) e.bands / (double) room::NB)));
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
// The Amount fader of a found room on the sound itself (s 0 .. 2, 1 = untouched). From the middle to half way left
// the engine's own answer comes in, and half way left it is in full. How deep the cut should go depends on the
// sound (a wet one wants more, a nearly dry one less), so from there to far left the answer is made bolder, up to
// depth roomDeep: the cells the engine is sure about stay where they are and the doubtful ones move most. The ear
// picks. Right of the middle: more reverb. (The prototype's engine.fader.)
constexpr double roomHalf = 0.5, roomDeep = 1.5;
// What that fader does to the room's level, for the display (dB): how far the room comes down where it is on its own
// is the typical value over the scoreboard's reverb cases, 4.5 dB with the engine's own answer and 10.6 dB at far
// left (it differs from sound to sound; under the sound itself less can be taken). Right of the middle: plain level.
constexpr double roomDownOwn = 4.5, roomDownDeep = 10.6;
inline double roomAmountDb (double s)
{
    if (s >= 1.0) return 20.0 * std::log10 (s);
    return s >= roomHalf ? -roomDownOwn * (1.0 - s) / (1.0 - roomHalf) : -(roomDownOwn + (roomDownDeep - roomDownOwn) * (roomHalf - s) / roomHalf);
}
inline Stereo roomFader (const Stereo& y, const room::Found& f, double s, double length, double bright, double width, const room::Stop& stop = {})
{
    if (s >= 1.0) return room::oneSound (y, f, s, length, bright, width, stop);
    if (s >= roomHalf) return room::oneSound (y, f, (s - roomHalf) / (1.0 - roomHalf), length, bright, width, stop);
    room::Found d; d.picture.frames = f.picture.frames; d.picture.LE = f.picture.LE; d.gains = f.gains; d.found = f.found; d.kdb = f.kdb; d.t60 = f.t60; d.ev = f.ev; d.width = f.width;
    const double b = roomDeep * (roomHalf - s) / roomHalf;
    for (float& g : d.gains) { const double p = std::clamp (-(double) g / 30.0, 1e-6, 1.0 - 1e-6); g = (float) (-30.0 / (1.0 + std::exp (-(std::log (p / (1.0 - p)) + b)))); }
    return room::oneSound (y, d, 0.0, length, bright, width, stop);
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
inline bool mergeRoom (Inspection& I, const room::Found& f)
{
    I.prob["reverb"] = roomSureness (f); return true;
}

// A sound that is cut off while it is still playing has no ending. The model learned from sounds that end inside
// the file, where a room shows as the tail after the last note, and where the file simply stops it reads a room into
// many dry sounds: of the scoreboard's 120 dry sounds cut off at 6 s it showed a room on 54 (on 18 when silence
// follows them, as in the sounds it learned from), of the corpus's 198 dry clips as files on 90.
// So a found room is looked at a second time with silence after the sound, and the less sure of the two verdicts
// counts (returned: the second one, or -1 when there was nothing to look at). Then a room shows on 18 of the 120
// and on 29 of the 198. Sounds that end inside the file lose nothing (120 with a reverb whose tail rings out: 104
// shown by either look; their dry versions: 18 and 18). The price is paid by sounds with a real reverb that are cut
// off: 68 of 120 shown instead of 88 (Research/room/bench/cut_test.py). The room that is shown and worked with is
// still the one found on the sound as it is. Costs one more look of the model.
constexpr double roomEndSilence = 3.0;      // seconds
inline double recheckRoomEnd (Inspection& I, const Stereo& y, const room::Net& net, const room::Net* second, const room::Stop& stop = {})
{
    auto r = I.prob.find ("reverb"); if (r == I.prob.end() || ! (r->second > roomNone)) return -1.0;
    Stereo z = y; const size_t n = y.size() + (size_t) (roomEndSilence * SR); z.l.resize (n, 0.0); z.r.resize (n, 0.0);
    const room::Found f = room::find (z, net, second, stop); if (f.stopped) return -1.0;
    const double s = roomSureness (f); r->second = std::min (r->second, s); return s;
}

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

// A train of echoes dies away as a room does, and the room engine can take it for one: on the scoreboard's 16 sounds
// with a delay it showed a REVERB group (rule above 65 %) on 5, and only one of those sounds has a room of its own.
// So when the echo engine finds echoes and the room engine a room, the room must still be there with the echoes
// taken out: the room engine looks at that sound as well, and the less sure of its two verdicts counts (returned:
// the second one, or -1 when there was nothing to look at). On the 16 sounds that takes 3 of the 4 away and adds
// none (Research/echo/roomcheck.py; that rooms with echoes on top keep their group: roomcheck2.py). The room that
// is shown and worked with is still the one found on the sound as it is. Costs about a second.
inline double recheckRoom (Inspection& I, const Stereo& y, const echo::Found& e, const room::Net& net, const room::Net* second, const room::Stop& stop = {})
{
    auto r = I.prob.find ("reverb"); if (! e.any() || r == I.prob.end() || ! (r->second > roomNone)) return -1.0;
    echo::Shape sh; sh.amount = 0.0; const Stereo dry = echo::render (y, e, sh, true, stop); if (dry.size() != y.size()) return -1.0;
    const room::Found f = room::find (dry, net, second, stop); if (f.stopped) return -1.0;
    const double s = roomSureness (f); r->second = std::min (r->second, s); return s;
}

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
inline std::vector<FoundFx> estimate (const Inspection& I, const Stereo& y, double cut = 0.65, const std::set<std::string>* only = nullptr,
                                      const std::shared_ptr<const room::Found>& room = {}, const std::shared_ptr<const tone::Found>& tone = {},
                                      const std::shared_ptr<const echo::Found>& echo = {}, const std::shared_ptr<const level::Movement>& lvl = {},
                                      const std::shared_ptr<const sweep::Result>& swp = {})
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
        if (f == "reverb" && room != nullptr && room->found) { out.push_back (roomFx (room, pit->second)); continue; }
        if (f == "comb") { if (tone != nullptr && ! tone->echoes.empty()) out.push_back (combFx (tone)); continue; }
        if (f == "delay") { if (echo != nullptr && echo->any()) out.push_back (echoFx (echo, pit->second)); continue; }
        if (lvl != nullptr && (f == "tremolo" || f == "autopan" || f == "gatedrhythm")) { if (lvl->found && f == levelName (*lvl)) out.push_back (levelFx (lvl, pit->second)); continue; }
        if (swp != nullptr && f == "flanger") { if (swp->found) out.push_back (flangerFx (swp, pit->second)); continue; }
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
inline Stereo applyTo (const Stereo& target, const std::vector<FoundFx>& found, const std::map<std::string, double>& faders, const std::function<bool()>& cancelled = {},
                       FoundFx* shown = nullptr)
{
    Stereo y = target;
    for (auto& F : found)
    {
        if (cancelled && cancelled()) break;
        if (F.name == "tonematch") continue;
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
        y = applyAmount (F.name, y, scaled (F, faders), amountOf (F, faders));
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
    limit (y);
    return y;
}

// One sound: take the found effects out (left), leave as is (middle) or double (right).
inline bool canRemove (const std::string& f) { return f != "reverb" && f != "distortion" && f != "bitcrush" && ! isNewFx (f); }
inline bool canRemove (const FoundFx& F) { return F.room != nullptr || F.tone != nullptr || F.echo != nullptr || F.level != nullptr || F.sweep != nullptr || canRemove (F.name); }      // the room engine takes its reverb out, the tone engine its reflections, the echo engine its echoes, the level engine its movement, the flanger engine its moving copy
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
inline Stereo reshape (const Stereo& src, const std::vector<FoundFx>& found, const std::map<std::string, double>& faders, const std::function<bool()>& cancelled = {})
{
    Stereo y = src;
    for (auto it = found.rbegin(); it != found.rend(); ++it)      // undo in reverse chain order
    {
        if (cancelled && cancelled()) break;
        const FoundFx& F = *it; const double s = amountOf (F, faders); const bool moved = settingsMoved (F, faders);
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
    limit (y);
    return y;
}

} // namespace dr
