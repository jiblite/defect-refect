#pragma once
// The echo engine: finds echoes (30 ms to 2.5 s) on ONE sound, takes them out, puts them on another sound and
// reshapes them. Port of Research/echo/echo2.py (Tools/echo_parity.py compares the two).
//
// A sound plus quieter, later copies of itself is the sound run through a filter, and that filter shows in the
// sound's own long-term spectrum as a ripple: fine, evenly spaced waves whose spacing is one over the echo time.
// The picture of all ripples is the cepstrum (the transform of the log spectrum), with a peak at the echo's lag and
// smaller ones at its multiples. The pieces of that picture around those lags ARE the log of the echo filter (a
// sound plus quieter copies is a minimum-phase filter, and such a filter is given by its ripple alone). So the same
// numbers take the echo out exactly (the inverse filter) and put it on another sound.
// Many things make peaks in that picture: a steady pitch, a drum loop, a note played twice. So every lag that
// stands out is measured in several cheap ways (measure()) and a small fitted rule says how sure it is an echo
// (EchoRule.h, fitted by Research/echo/fit3.py).
// Limits: one echo time per channel (the surest); echoes that cross from one channel to the other on a wide sound
// are not covered; under 30 ms it is a comb filter (Tone.h).

#include "Tone.h"
#include "EchoRule.h"
#include <array>
#include <limits>
#include <map>
#include <memory>

namespace dr
{
namespace echo
{
constexpr int QMIN = 1440, QMAX = 120000;       // lags looked at, samples (30 ms .. 2.5 s)
constexpr int HALF = 96, WIN = 2 * HALF + 1;    // the piece of the picture kept around a lag
constexpr int NMAX = 912000;                    // the longest stretch that is analysed (19 s)
constexpr int MC = 4096;                        // grid for "where the sound has its level"
using Stop = room::Stop;

// Transforms of the sizes in use, made once per job.
struct Fft
{
    std::map<int, std::unique_ptr<room::Transform>> made;
    const room::Transform& of (int n) { auto& p = made[n]; if (! p) p = std::make_unique<room::Transform> (n); return *p; }
};
inline long pow2 (long n) { long p = 1; while (p < n) p *= 2; return p; }
// The transform of a real signal, zero-padded to N points.
inline void forward (Fft& F, const double* x, long len, int N, Vec& re, Vec& im)
{
    re.assign ((size_t) N, 0.0); im.assign ((size_t) N, 0.0); std::copy (x, x + std::min (len, (long) N), re.begin()); F.of (N).run (re, im);
}

// The part of the sound that is looked at: all of it up to 19 s, else the 19 s with the most energy.
inline std::pair<long, long> stretch (const Stereo& x)
{
    const long n = (long) x.size(); if (n <= NMAX) return { 0, n };
    const long hop = 24000, nb = n / hop, k = NMAX / hop; Vec c ((size_t) nb + 1, 0.0);
    for (long b = 0; b < nb; ++b)
    {
        double e = 0; for (long i = b * hop; i < (b + 1) * hop; ++i) e += x.l[(size_t) i] * x.l[(size_t) i] + x.r[(size_t) i] * x.r[(size_t) i];
        c[(size_t) b + 1] = c[(size_t) b] + e;
    }
    long best = 0; double top = -1.0;
    for (long s = 0; s + k <= nb; ++s) { const double v = c[(size_t) (s + k)] - c[(size_t) s]; if (v > top) { top = v; best = s; } }
    return { best * hop, (long) NMAX };
}

// ---------------------------------------------------------------- the ripple picture
// Cepstrum of one channel: the log spectrum with its trend taken off and the dead parts of the spectrum left out.
// PsC: the smoothed power spectrum summed onto MC/2+1 coarse bins (where the sound has its level).
// xr, xi: the channel's own transform, kept when the caller can use this size again.
struct Cep { Vec c, PsC, xr, xi; int nfft = 0; };
inline Cep cep (Fft& F, const double* x, long n, int keepIf = 0)
{
    Cep o; o.nfft = (int) pow2 (2 * n + 2 * QMAX); const int N = o.nfft, H = N / 2;
    Vec re, im; forward (F, x, n, N, re, im);
    Vec P ((size_t) H + 1); for (int k = 0; k <= H; ++k) P[(size_t) k] = re[(size_t) k] * re[(size_t) k] + im[(size_t) k] * im[(size_t) k];
    if (keepIf == N) { o.xr = re; o.xi = im; }
    const int w = std::max (3, (int) std::nearbyint ((double) N / QMIN / 2.0) | 1);     // half the widest ripple period we look at, in bins
    const Vec Ps = uniformFilter (P, 4 * w + 1, 1); const double top = *std::max_element (Ps.begin(), Ps.end()); const int r = N / MC;
    o.PsC.assign ((size_t) MC / 2 + 1, 0.0);
    if (! (top > 0)) { o.c.assign ((size_t) N, 0.0); return o; }
    Vec m ((size_t) H + 1); for (int k = 0; k <= H; ++k) m[(size_t) k] = Ps[(size_t) k] > top * 1e-10 ? 1.0 : 0.0;        // where the file holds anything at all (100 dB under the top)
    m = uniformFilter (m, 4 * w + 1, 1);                                                                             // .. with soft edges
    Vec L ((size_t) H + 1); for (int k = 0; k <= H; ++k) L[(size_t) k] = std::log (std::max ({ P[(size_t) k], Ps[(size_t) k] * 1e-4, 1e-300 }));
    { const Vec trend = uniformFilter (L, 8 * w + 1, 1); for (int k = 0; k <= H; ++k) L[(size_t) k] = (L[(size_t) k] - trend[(size_t) k]) * m[(size_t) k]; }
    for (int j = 0; j < MC / 2; ++j)                // coarse bin j holds the fine bins j r - r/2 .. j r + r/2 - 1
    {
        double s = 0; for (int i = std::max (0, j * r - r / 2); i < j * r + r / 2; ++i) s += Ps[(size_t) i];
        o.PsC[(size_t) j] = s;
    }
    { double s = 0; for (int i = H - r / 2; i <= H; ++i) s += Ps[(size_t) i]; o.PsC[(size_t) MC / 2] = s; }
    std::fill (im.begin(), im.end(), 0.0); for (int k = 0; k <= H; ++k) re[(size_t) k] = L[(size_t) k]; for (int k = 1; k < H; ++k) re[(size_t) (N - k)] = L[(size_t) k];
    F.of (N).back (re, im); o.c = std::move (re);
    return o;
}

// Typical size of the picture around each lag (median of its size over about 85 ms), for telling a peak from the rest.
inline Vec background (const Vec& c, int q1)
{
    const int nd = (q1 + 15) / 16; Vec a ((size_t) nd), med ((size_t) nd), w (257);
    for (int i = 0; i < nd; ++i) a[(size_t) i] = std::abs (c[(size_t) i * 16]);
    for (int i = 0; i < nd; ++i)
    {
        for (int k = 0; k < 257; ++k) w[(size_t) k] = a[(size_t) std::clamp (i - 128 + k, 0, nd - 1)];
        std::nth_element (w.begin(), w.begin() + 128, w.end()); med[(size_t) i] = w[128];
    }
    Vec bg ((size_t) q1);
    for (int x = 0; x < q1; ++x)
    {
        const int i = x / 16; const double v = i + 1 < nd ? (med[(size_t) i + 1] - med[(size_t) i]) / 16.0 * (double) (x - 16 * i) + med[(size_t) i] : med[(size_t) nd - 1];
        bg[(size_t) x] = std::max (v, 1e-12) / 0.6745;        // as a standard deviation
    }
    return bg;
}

// The picture around lag q, with what does not stand out of the background turned down (soft threshold at t times it).
inline const Vec& taper() { static const Vec t = [] { Vec h = hanningSym ((size_t) WIN); for (auto& v : h) v = std::sqrt (v); return h; }(); return t; }
inline Vec window (const Vec& c, int q, const Vec* bg = nullptr, double t = 3.0)
{
    Vec w ((size_t) WIN); const Vec& tp = taper();
    for (int i = 0; i < WIN; ++i)
    {
        double v = c[(size_t) (q - HALF + i)] * tp[(size_t) i];
        if (bg != nullptr) { const double b = t * (*bg)[(size_t) (q - HALF + i)], r = b / std::max (std::abs (v), 1e-30); v *= std::max (0.0, 1.0 - r * r); }
        w[(size_t) i] = v;
    }
    return w;
}
// The size of the ripple a lag window makes, over frequency (coarse bins).
inline Vec ripple (Fft& F, const Vec& w)
{
    Vec re ((size_t) MC, 0.0), im ((size_t) MC, 0.0);
    for (int i = 0; i <= HALF; ++i) re[(size_t) i] = w[(size_t) (HALF + i)];
    for (int i = 0; i < HALF; ++i) re[(size_t) (MC - HALF + i)] = w[(size_t) i];
    F.of (MC).run (re, im); Vec R ((size_t) MC / 2 + 1);
    for (int k = 0; k <= MC / 2; ++k) R[(size_t) k] = re[(size_t) k] * re[(size_t) k] + im[(size_t) k] * im[(size_t) k];
    return R;
}
// How loud is the copy with lag window w against the sound itself, where the sound has its level (1 = as loud).
inline double loudness (Fft& F, const Vec& w, const Vec& PsC)
{
    const Vec R = ripple (F, w); double a = 0, b = 0;
    for (size_t k = 0; k < R.size(); ++k) { a += PsC[k] * R[k]; b += PsC[k]; }
    return std::sqrt (a / std::max (b, 1e-30));
}

// Lags from QMIN up where the picture stands out, by how far (at most `most`).
struct Cand { int q = 0; double score = 0, h = 0, a = 0; bool rah = false; };      // lag, how far it stands out, value at the peak, loudness of the copy; rah: one of a pitch's series
inline std::vector<Cand> candidates (Fft& F, const Cep& C, Vec& bg, double thr = 8.0, int most = 8)
{
    const int lim = std::min (QMAX, C.nfft / 2 - 2 * HALF), q1 = std::min (QMAX + HALF, C.nfft / 2 - 2 * HALF);
    bg = background (C.c, q1 + HALF);
    Vec a ((size_t) q1), sc ((size_t) q1); for (int i = 0; i < q1; ++i) { a[(size_t) i] = std::abs (C.c[(size_t) i]); sc[(size_t) i] = a[(size_t) i] * a[(size_t) i]; }
    sc = uniformFilter (sc, 9, 1); for (int i = 0; i < q1; ++i) sc[(size_t) i] = std::sqrt (std::max (0.0, sc[(size_t) i]) * 9.0) / bg[(size_t) i];       // size over +-4 samples
    for (int i = 0; i < std::min (QMIN, q1); ++i) sc[(size_t) i] = 0.0;
    for (int i = std::max (0, lim - 2 * HALF - 3); i < q1; ++i) sc[(size_t) i] = 0.0;      // a base lag must leave room for its window in series()
    std::vector<Cand> out;
    for (int it = 0; it < most; ++it)
    {
        const int k = (int) (std::max_element (sc.begin(), sc.end()) - sc.begin());
        if (sc[(size_t) k] < thr) break;
        int j = k - 4; for (int i = k - 4; i <= k + 4; ++i) if (a[(size_t) i] > a[(size_t) j]) j = i;
        Cand cd; cd.q = j; cd.score = sc[(size_t) k]; cd.h = C.c[(size_t) j]; cd.a = loudness (F, window (C.c, j, &bg), C.PsC); out.push_back (cd);
        for (int i = std::max (0, k - 2 * HALF); i < std::min (q1, k + 2 * HALF); ++i) sc[(size_t) i] = 0.0;
    }
    return out;
}
// Is the peak at lag q one of a series that starts at a shorter lag (a steady pitch gives peaks at its period and
// every multiple)? True when the picture also stands out at q/k for some k = 2 .. kmax, at least `part` as far as at q.
inline bool rahmonic (const Vec& c, const Vec& bg, int q, int lo = 96, int kmax = 8, double part = 0.5)
{
    auto peak = [&] (int j, int r) { double m = 0; for (int i = j - r; i <= j + r; ++i) m = std::max (m, std::abs (c[(size_t) i])); return m; };
    const double here = peak (q, 2) / bg[(size_t) q];
    for (int k = 2; k <= kmax; ++k)
    {
        const int j = (int) std::nearbyint ((double) q / k); if (j < lo) break;
        const int r = std::max (2, (int) std::nearbyint (0.004 * j));
        if (peak (j, r) / bg[(size_t) j] > part * here) return true;
    }
    return false;
}

// The echo part of the picture for base lag D: the windows at the lag itself and at its multiples (those that stand out).
struct Peak { int q = 0; double v = 0; int k = 0; };
struct Part { std::vector<std::pair<int, Vec>> win; std::vector<Peak> peaks; };
inline Part series (const Cep& C, const Vec& bg, int D, int kmax = 8)
{
    Part e; const long lim = std::min (QMAX, C.nfft / 2 - 2 * HALF);
    for (int k = 1; k <= kmax; ++k)
    {
        const long q = (long) D * k; if (q + 2 * HALF > lim) break;
        int j = (int) q;
        if (k > 1) { j = (int) q - 6; for (int i = (int) q - 6; i <= (int) q + 6; ++i) if (std::abs (C.c[(size_t) i]) > std::abs (C.c[(size_t) j])) j = i; }
        Vec w = window (C.c, j, &bg);
        if (k > 1 && std::all_of (w.begin(), w.end(), [] (double v) { return v == 0.0; })) continue;
        e.peaks.push_back ({ j, C.c[(size_t) j], k }); e.win.emplace_back (j, std::move (w));
    }
    return e;
}

// The windows as the log of the echo filter on a grid of N points (lags times `tscale`): its transform.
inline void logFilter (Fft& F, const Part& e, int N, double tscale, Vec& re, Vec& im)
{
    re.assign ((size_t) N, 0.0); im.assign ((size_t) N, 0.0);
    for (const auto& [q0, w] : e.win)
    {
        const long q = std::lround ((double) q0 * tscale);
        if (q - HALF >= 0 && q + HALF + 1 <= N / 2) for (int i = 0; i < WIN; ++i) re[(size_t) (q - HALF + i)] += w[(size_t) i];
    }
    F.of (N).run (re, im);
}
// exp of a transform's values (sign -1: of their negatives), for a real signal's transform.
inline void expOf (const Vec& lr, const Vec& li, double sign, Vec& hr, Vec& hi)
{
    const int N = (int) lr.size(); hr.resize ((size_t) N); hi.resize ((size_t) N);
    for (int k = 0; k <= N / 2; ++k) { const double m = std::exp (sign * lr[(size_t) k]); hr[(size_t) k] = m * std::cos (li[(size_t) k]); hi[(size_t) k] = sign * m * std::sin (li[(size_t) k]); }
    for (int k = 1; k < N / 2; ++k) { hr[(size_t) (N - k)] = hr[(size_t) k]; hi[(size_t) (N - k)] = -hi[(size_t) k]; }
}
// The sound (its transform on a grid of N points) with the echo part taken out: its first n samples.
inline Vec without (Fft& F, const Vec& xr, const Vec& xi, const Part& e, int N, long n)
{
    Vec lr, li, hr, hi; logFilter (F, e, N, 1.0, lr, li); expOf (lr, li, -1.0, hr, hi);
    for (int k = 0; k < N; ++k) { const double a = xr[(size_t) k], b = xi[(size_t) k]; lr[(size_t) k] = a * hr[(size_t) k] - b * hi[(size_t) k]; li[(size_t) k] = a * hi[(size_t) k] + b * hr[(size_t) k]; }
    F.of (N).back (lr, li); lr.resize ((size_t) n); return lr;
}

// ---------------------------------------------------------------- what is measured about a lag
// Taking a true echo out only ever takes level away. Returns (added, removed): energy that appears in y where x had
// less, and energy that went, over 50 ms steps, as shares of x's energy.
inline std::pair<double, double> onlyRemoves (const double* x, const double* y, long n, long hop = 2400)
{
    const long nb = n / hop; double tot = 0, add = 0, rem = 0;
    for (long b = 0; b < nb; ++b)
    {
        double ea = 0, eb = 0; for (long i = b * hop; i < (b + 1) * hop; ++i) { ea += x[i] * x[i]; eb += y[i] * y[i]; }
        ea /= (double) hop; eb /= (double) hop; tot += ea; add += std::max (eb - ea, 0.0); rem += std::max (ea - eb, 0.0);
    }
    tot = std::max (tot, 1e-30); return { add / tot, rem / tot };
}
// How evenly is the copy there across the spectrum (a true echo copies everything)? The ripple's size in bands of
// equal width on a log scale from 100 Hz to 16 kHz, over the bands where the sound has level: (share of those bands
// where the copy is at least half its median size, their number).
inline std::pair<double, int> evenness (Fft& F, const Vec& w, const Vec& PsC, int bands = 24)
{
    const Vec R = ripple (F, w); Vec lev ((size_t) bands), siz ((size_t) bands);
    for (int b = 0; b < bands; ++b)
    {
        const double lo = 100.0 * std::pow (160.0, (double) b / bands), hi = 100.0 * std::pow (160.0, (double) (b + 1) / bands); double sl = 0, sr = 0; int n = 0;
        for (int k = 0; k <= MC / 2; ++k) { const double f = (double) k * SR / MC; if (f >= lo && f < hi) { sl += PsC[(size_t) k]; sr += R[(size_t) k]; ++n; } }
        lev[(size_t) b] = n ? sl / n : 0.0; siz[(size_t) b] = n ? std::sqrt (sr / n) : 0.0;
    }
    const double top = *std::max_element (lev.begin(), lev.end()); Vec s; for (int b = 0; b < bands; ++b) if (lev[(size_t) b] > top * 1e-4) s.push_back (siz[(size_t) b]);
    if (s.size() < 3) return { 0.0, (int) s.size() };
    const double med = median (s); int ok = 0; for (double v : s) ok += v >= 0.5 * med;
    return { (double) ok / (double) s.size(), (int) s.size() };
}
// Every moment of the sound against the moment D later, in bands of 8 bins up to 16 kHz: where the later one is a
// copy of the earlier one (coherence over cmin), how much quieter or louder is it? An echo's copies are all quieter
// by the same amount; notes that are played again are about as loud as before. Only moments that can testify
// count: a held note is "a copy of itself" at every lag, so cells that are also coherent half a frame before or
// after the lag are left out.
// info = part of the sound's energy that can testify, share = part of that with a copy at the lag, med = typical
// level of the copies (dB against the original), iqr = how much that varies (dB), up = part of the copies that are
// as loud as the original or louder (within 1 dB), n = number of copies, eff = how many of them carry the weight (a
// thousand copies of which one is loud are one witness, not a thousand).
struct Pairs { double share = 0, med = 0, iqr = 0, up = 0, info = 0, eff = 0; int n = 0; };
inline Pairs pairs (Fft& F, const double* x, long len, int D, double cmin = 0.9, double ctl = 0.7)
{
    Pairs z; const int n = D >= 2730 ? 2048 : 1024, hop = n / 4, h2 = n / 2; const long m = len - D - n - h2;
    if (m < h2 + 4 * hop) return z;
    const long T = (m - h2 + hop - 1) / hop; const int nb = ((int) (16000.0 * n / SR) - 4) / 8, nk = nb * 8;
    const Vec w = hanningSym ((size_t) n); const room::Transform& tr = F.of (n);
    // the transforms (bins 4 .. 4 + nk) of the frames that start at first, first + hop, ..
    auto frames = [&] (long first, long count, Vec& R, Vec& I)
    {
        R.resize ((size_t) (count * nk)); I.resize (R.size()); Vec re ((size_t) n), im ((size_t) n);
        for (long f = 0; f < count; ++f)
        {
            const double* p = x + first + f * hop; for (int i = 0; i < n; ++i) { re[(size_t) i] = p[i] * w[(size_t) i]; im[(size_t) i] = 0.0; }
            tr.run (re, im); std::copy (re.begin() + 4, re.begin() + 4 + nk, R.begin() + f * nk); std::copy (im.begin() + 4, im.begin() + 4 + nk, I.begin() + f * nk);
        }
    };
    // the sound's own frames, and those D later: two frames earlier (half a frame before the lag) up to two frames later
    Vec XR, XI, YR, YI; frames (h2, T, XR, XI); frames (D, T + 4, YR, YI);
    auto energy = [&] (const Vec& R, const Vec& I, long count)
    {
        Vec E ((size_t) (count * nb));
        for (long f = 0; f < count; ++f) for (int b = 0; b < nb; ++b)
        { double s = 0; const size_t o = (size_t) (f * nk + b * 8); for (size_t j = o; j < o + 8; ++j) s += R[j] * R[j] + I[j] * I[j]; E[(size_t) (f * nb + b)] = s; }
        return E;
    };
    const Vec EX = energy (XR, XI, T), EY = energy (YR, YI, T + 4);
    auto coh = [&] (long f, long g, int b)
    {
        double cr = 0, ci = 0; const size_t ox = (size_t) (f * nk + b * 8), oy = (size_t) (g * nk + b * 8);
        for (size_t j = 0; j < 8; ++j) { cr += YR[oy + j] * XR[ox + j] + YI[oy + j] * XI[ox + j]; ci += YI[oy + j] * XR[ox + j] - YR[oy + j] * XI[ox + j]; }
        return std::sqrt (cr * cr + ci * ci) / std::sqrt (std::max (EX[(size_t) (f * nb + b)] * EY[(size_t) (g * nb + b)], 1e-60));
    };
    const double top = *std::max_element (EX.begin(), EX.end()); double tot = 0, ti = 0, sok = 0;
    std::vector<std::pair<double, double>> cp;      // the copies: level against the original (dB), weight
    for (long f = 0; f < T; ++f) for (int b = 0; b < nb; ++b)
    {
        const double ex = EX[(size_t) (f * nb + b)]; if (! (ex > top * 1e-5)) continue;
        tot += ex; if (std::max (coh (f, f + 4, b), coh (f, f, b)) > ctl) continue;
        ti += ex; const double ey = EY[(size_t) ((f + 2) * nb + b)];
        if (coh (f, f + 2, b) > cmin && ey > top * 1e-7) { sok += ex; cp.emplace_back (10.0 * std::log10 (ey / ex), ex); }
    }
    z.info = tot > 0 ? ti / tot : 0.0; if (! (ti > 0)) return z;
    z.share = sok / ti; z.n = (int) cp.size(); if (cp.size() < 8) return z;
    double wsum = 0, wup = 0, wsq = 0; for (auto& [r, wt] : cp) { wsum += wt; wsq += wt * wt; if (r > -1.0) wup += wt; }
    z.eff = wsum * wsum / std::max (wsq, 1e-300);
    std::stable_sort (cp.begin(), cp.end(), [] (const auto& a, const auto& b) { return a.first < b.first; });
    Vec cw (cp.size()); { double s = 0; for (size_t i = 0; i < cp.size(); ++i) { s += cp[i].second; cw[i] = s / wsum; } }
    auto q = [&] (double v) { const size_t i = (size_t) (std::lower_bound (cw.begin(), cw.end(), v) - cw.begin()); return cp[std::min (i, cp.size() - 1)].first; };
    z.med = q (0.5); z.iqr = q (0.75) - q (0.25); z.up = wup / wsum;
    return z;
}

// Everything measured about one lag, and what the rule looks at (nan = cannot be said: fewer than 8 copies).
struct Record
{
    int q = 0; double ms = 0, score = 0, a = 0, h = 0, a2 = 0, added = 0, removed = 0, even = 0; int bands = 0, others = 0, npk = 0;
    double secs = 0, pshare = 0, pmed = 0, piqr = 0, pup = 0; int pn = 0; double pinfo = 0, peff = 0;
    double p = 0;       // the rule's answer
};
inline std::array<double, rule::features> feats (const Record& r)
{
    const double nan = std::numeric_limits<double>::quiet_NaN(), a = std::max (r.a, 1e-6); const bool en = r.pn >= 8; const double pd = en ? r.pmed - 20.0 * std::log10 (a) : nan;
    return { std::log10 (r.score), r.a, r.a2 / a, std::min (r.added / std::max (r.removed, 1e-4), 3.0), std::log10 (std::max (r.removed, 1e-4)), (double) std::min (r.others, 6), std::log10 (r.ms), r.even,
             (double) std::min (r.npk, 12), (double) r.bands, r.pshare, pd, std::abs (pd), en ? r.piqr : nan, en ? r.pup : nan, std::log10 ((double) r.pn + 1.0), r.secs * 1000.0 / r.ms, r.pinfo,
             std::log10 (r.peff + 1.0) };
}
inline double ruleProb (const Record& r)
{
    const auto x = feats (r); double s = rule::base;
    for (int t = 0; t < rule::trees; ++t)
    {
        const rule::Node* nd = rule::nodes + rule::start[t]; int i = 0;
        while (! nd[i].leaf) { const double v = x[(size_t) nd[i].feat]; i = std::isnan (v) ? (nd[i].missLeft ? nd[i].left : nd[i].right) : (v <= nd[i].v ? nd[i].left : nd[i].right); }
        s += nd[i].v;
    }
    return 1.0 / (1.0 + std::exp (-s));
}

// One channel (the stretch): the most echo-like lags (loudest copy first, at most `most`), each with everything
// measured about it and its part of the picture. seen: receives all lags that stood out.
struct Measured { Record rec; Part part; };
inline std::vector<Measured> measure (Fft& F, const double* x, long n, std::vector<Cand>* seen = nullptr, const Stop& stop = {}, double thr = 8.0, double amin = 0.05, double amax = 0.95, int most = 3)
{
    std::vector<Measured> out; if (seen != nullptr) seen->clear(); if (n < 4800) return out;
    const int NA = (int) pow2 (n + 2 * QMAX + (long) SR);
    Cep C = cep (F, x, n, NA); if (stop && stop()) return {};
    Vec bg; std::vector<Cand> cand = candidates (F, C, bg, thr);
    for (auto& k : cand) k.rah = rahmonic (C.c, bg, k.q);
    if (seen != nullptr) *seen = cand;
    std::vector<int> ok; for (int i = 0; i < (int) cand.size(); ++i) if (cand[(size_t) i].a >= amin && cand[(size_t) i].a <= amax && ! cand[(size_t) i].rah) ok.push_back (i);
    std::stable_sort (ok.begin(), ok.end(), [&] (int p, int q) { return cand[(size_t) p].a > cand[(size_t) q].a; }); if ((int) ok.size() > most) ok.resize ((size_t) most);
    if (ok.empty()) return out;
    if (C.xr.empty()) forward (F, x, n, NA, C.xr, C.xi);
    for (int bi : ok)
    {
        if (stop && stop()) return {};
        const Cand& base = cand[(size_t) bi]; Measured M; M.part = series (C, bg, base.q); if (M.part.win.empty()) continue;
        Record& r = M.rec; r.q = base.q; r.ms = base.q / 48.0; r.score = base.score; r.a = base.a; r.h = base.h;
        r.a2 = M.part.peaks.size() > 1 && M.part.peaks[1].k == 2 ? loudness (F, M.part.win[1].second, C.PsC) : 0.0;
        { const Vec y = without (F, C.xr, C.xi, M.part, NA, n); const auto ar = onlyRemoves (x, y.data(), n); r.added = ar.first; r.removed = ar.second; }
        { const auto ev = evenness (F, M.part.win[0].second, C.PsC); r.even = ev.first; r.bands = ev.second; }
        for (int j = 0; j < (int) cand.size(); ++j)
            if (j != bi && cand[(size_t) j].score > 0.5 * base.score) { const double ratio = (double) cand[(size_t) j].q / base.q; if (std::abs (ratio - std::nearbyint (ratio)) > 0.02) ++r.others; }
        r.npk = (int) M.part.peaks.size(); r.secs = (double) n / SR;
        const Pairs pr = pairs (F, x, n, base.q); r.pshare = pr.share; r.pmed = pr.med; r.piqr = pr.iqr; r.pup = pr.up; r.pn = pr.n; r.pinfo = pr.info; r.peff = pr.eff;
        r.p = ruleProb (r); out.push_back (std::move (M));
    }
    return out;
}

// ---------------------------------------------------------------- the faders
// Fader values, 1 = as found. amount: 0 = no echoes, 2 = twice as loud. time: a factor on the echo time. feedback:
// how long the repeats go on (0 = only the first one, 2 = each repeat keeps twice as much of the one before; an
// echo that does not go on by feedback gets repeats added right of the middle). bright: the tone of the echoes
// (0 = dark: the highs cut, 2 = bright: the highs lifted).
struct Shape { double amount = 1.0, time = 1.0, feedback = 1.0, bright = 1.0; };
constexpr double fbAdd = 0.35, fbTop = 0.85, loopTop = 0.95;            // feedback an echo without any gets at far right; the most the fader asks for; the most a loop may hand on at any frequency
constexpr double brightDown = -24.0, brightUp = 6.0, brightHz = 2000.0;     // Brightness: a shelf over the echoes' highs, dB at far left and far right, its middle
constexpr double trainDb = 30.0;                                        // repeats within this of the first one count as the echo's own
struct Repeat { double ms = 0, db = 0; };

struct Channel
{
    bool found = false; Record rec; Part part;
    std::vector<double> level;      // as found: the repeats' levels against the sound itself, dB (first, second, ..)
    int count = 1;                  // how many of them count (within trainDb of the first)
    bool train = false;             // they go on by feedback (three or more, each keeping about the same share of the one before)
    double phi = 0;                 // that share (for a train: the typical one; else the second against the first; 0 = a single echo)
};
// The share each repeat keeps of the one before when the Feedback fader is at v (1 = as found, phi).
inline double feedbackAt (double phi, double v) { return std::min (fbTop, v * phi + std::max (0.0, v - 1.0) * std::max (0.0, fbAdd - phi)); }

// The filter (impulse response on a grid of N points; what is longer wraps around) that gives the echoes as the
// faders ask. own = false: for another sound (it comes out with the echoes on it). own = true: for the sound the
// echoes were found on: the found echoes out and the asked ones in (Amount 0 is the exact inverse of the found
// echo filter, everything in the middle leaves the sound as it is).
inline Vec responseOn (Fft& F, const Channel& c, const Shape& sh, bool own, int N)
{
    const room::Transform& tr = F.of (N); const double s = sh.amount, t = sh.time; const size_t n = (size_t) N;
    Vec lr, li, er, ei; logFilter (F, c.part, N, t, lr, li); expOf (lr, li, 1.0, er, ei);
    for (auto& v : er) v -= 1.0;                                    // the echoes alone
    if (std::abs (sh.feedback - 1.0) > 1e-9)
    {
        // The echoes over time, repeat by repeat (repeat k is the stretch around k times the echo time).
        Vec hr = er, hi = ei; tr.back (hr, hi); std::fill (hi.begin(), hi.end(), 0.0);
        const long D = std::max (2L, std::lround ((double) c.rec.q * t)); const double v = sh.feedback;
        auto from = [&] (int k) { return std::min ((long) N, (long) k * D - D / 2); };
        if (v <= 1.0)       // fewer repeats: the k-th one keeps v^(k-1) of itself
        {
            double g = 1.0; for (int k = 1; from (k) < N; ++k) { if (k > 1) for (long i = from (k); i < from (k + 1); ++i) hr[(size_t) i] *= g; g *= v; }
            tr.run (hr, hi); er = hr; ei = hi;
        }
        else
        {
            // More repeats. A train that goes on by feedback: every repeat that counts gets u^(k-1) (u = the asked
            // share over the found one), and after the last of them the train goes on by itself, each repeat made
            // from the one before the way the second was made from the first (R = E2 / E1, times u). An echo that
            // does not go on by feedback keeps what was found and gets followers of its last repeat, each made from
            // the one before by the first repeat itself (scaled to the asked share).
            const int K = std::max (1, c.count); const double phi = c.train ? std::max (c.phi, 1e-3) : 0.0, phiT = feedbackAt (phi, v), u = c.train ? phiT / phi : 1.0;
            Vec r1 (n, 0.0), i1 (n, 0.0), rK (n, 0.0), iK (n, 0.0), r2, i2; double e1 = 0;
            for (long i = from (1); i < from (2); ++i) { r1[(size_t) i] = hr[(size_t) i]; e1 += hr[(size_t) i] * hr[(size_t) i]; }
            if (c.train) { r2.assign (n, 0.0); i2.assign (n, 0.0); for (long i = from (2); i < from (3); ++i) r2[(size_t) i] = hr[(size_t) i]; }
            double g = 1.0;
            for (int k = 1; k <= K; ++k) { for (long i = from (k); i < from (k + 1); ++i) { hr[(size_t) i] *= g; if (k == K) rK[(size_t) i] = hr[(size_t) i]; } g *= u; }
            if (c.train) for (long i = from (K + 1); i < N; ++i) hr[(size_t) i] = 0.0;
            tr.run (hr, hi); er = hr; ei = hi;
            if (e1 > 0)
            {
                tr.run (r1, i1); tr.run (rK, iK); if (c.train) tr.run (r2, i2);
                Vec Qr (n), Qi (n); double top = 0; const double eps = 1e-3 * e1, lvl = std::sqrt (e1);
                for (size_t k = 0; k < n; ++k)
                {
                    if (c.train) { const double d = r1[k] * r1[k] + i1[k] * i1[k] + eps; Qr[k] = u * (r2[k] * r1[k] + i2[k] * i1[k]) / d; Qi[k] = u * (i2[k] * r1[k] - r2[k] * i1[k]) / d; }
                    else { Qr[k] = phiT * r1[k] / lvl; Qi[k] = phiT * i1[k] / lvl; }
                    top = std::max (top, Qr[k] * Qr[k] + Qi[k] * Qi[k]);
                }
                top = std::sqrt (top); const double cap = top > loopTop ? loopTop / top : 1.0;
                for (size_t k = 0; k < n; ++k)
                {
                    const std::complex<double> Q (cap * Qr[k], cap * Qi[k]), d = std::complex<double> (rK[k], iK[k]) * Q / (1.0 - Q);
                    er[k] += d.real(); ei[k] += d.imag();
                }
            }
        }
    }
    if (std::abs (sh.bright - 1.0) > 1e-9)
    {
        // a first-order shelf over the highs (as a proper sampled filter, so the echoes stay where they are in time)
        const double gdb = sh.bright < 1.0 ? brightDown * (1.0 - sh.bright) : brightUp * (sh.bright - 1.0), sg = std::pow (10.0, gdb / 40.0), w0 = std::tan (PI * brightHz / SR);
        for (int k = 0; k <= N / 2; ++k)
        {
            const std::complex<double> E0 (er[(size_t) k], ei[(size_t) k]); std::complex<double> E;
            if (k == N / 2) E = { E0.real() * sg * sg, 0.0 };
            else { const double uu = std::tan (PI * (double) k / N) / w0; E = E0 * std::complex<double> (1.0, uu * sg) / std::complex<double> (1.0, uu / sg); }
            er[(size_t) k] = E.real(); ei[(size_t) k] = E.imag();
            if (k > 0 && k < N / 2) { er[(size_t) (N - k)] = E.real(); ei[(size_t) (N - k)] = -E.imag(); }
        }
    }
    for (size_t k = 0; k < n; ++k) { er[k] = 1.0 + s * er[k]; ei[k] = s * ei[k]; }       // the sound itself and the echoes as asked
    if (own)
    {
        if (std::abs (t - 1.0) > 1e-9) logFilter (F, c.part, N, 1.0, lr, li);
        Vec gr, gi; expOf (lr, li, -1.0, gr, gi);                   // the found echoes out
        for (size_t k = 0; k < n; ++k) { const double a = er[k], b = ei[k]; er[k] = a * gr[k] - b * gi[k]; ei[k] = a * gi[k] + b * gr[k]; }
    }
    tr.back (er, ei); return er;
}
// .. on a grid long enough for it (the last eighth must be 50 dB under the whole; at most 44 s), cut where it has died away.
inline Vec response (Fft& F, const Channel& c, const Shape& sh, bool own)
{
    Vec h;
    for (int N = 1 << 19; ; N *= 2)
    {
        h = responseOn (F, c, sh, own, N); double tot = 0, late = 0;
        for (int i = 0; i < N; ++i) { const double e = h[(size_t) i] * h[(size_t) i]; tot += e; if (i >= N - N / 8) late += e; }
        if (N >= (1 << 21) || ! (late > 1e-5 * tot)) break;
    }
    double pk = 0; for (double v : h) pk = std::max (pk, std::abs (v));
    size_t last = 0; for (size_t i = 0; i < h.size(); ++i) if (std::abs (h[i]) > 1e-5 * pk) last = i;
    h.resize (last + 1); return h;
}

// ---------------------------------------------------------------- one sound's echoes
struct Found
{
    Channel ch[2];
    std::vector<Record> seen[2];        // every lag that was measured, per channel
    std::vector<Cand> cand[2];          // every lag that stood out
    double p = 0, sure = 0;             // the rule's answer for the surer channel, and what the Sureness slider compares
    long start = 0, length = 0;         // the stretch that was looked at
    bool twin = false;                  // left and right are the same signal
    bool any() const { return ch[0].found || ch[1].found; }
};
// The rule's answer p is fitted on clips where one lag in three is an echo. Of the sounds a person drops on the app
// far fewer carry one: taken as one in ten, p becomes "how sure is it that this sound has an echo", which is what
// the Sureness slider compares (65 % on the slider is p = 0.89). On whole sounds the rule's p then behaves like
// this (Research/echo/judge3.py): of the scoreboard's 16 sounds with an outside delay 15 are found at p >= 0.85 and
// none with a wrong echo time; of 198 clean sounds 2 are reported at p >= 0.93 (both guitar loops with a quieter
// copy exactly 1 s later) and 4 at p >= 0.78.
// pFound: under this the surest lag is not offered at all. pOwn: what a second channel's lag needs when it does not
// fit the surer channel's echo time. pJoin: .. and when it does (the same time, or one a multiple of the other).
constexpr double priorFit = 0.334, priorUse = 0.10, pFound = 0.5, pOwn = 0.8, pJoin = 0.2, sureNone = 0.03;
inline double sureness (double p) { const double k = priorUse / (1.0 - priorUse) / (priorFit / (1.0 - priorFit)); return p * k / (p * k + (1.0 - p)); }

// The repeats of a channel's echo filter as found (levels against the sound itself), for the picture and the Feedback fader.
inline void repeatsOf (Fft& F, Channel& c)
{
    const int N = 1 << 19; const Vec h = responseOn (F, c, Shape(), false, N); const long D = c.rec.q; c.level.clear(); c.phi = 0; c.count = 1; c.train = false;
    Vec e;
    for (int k = 1; k <= 24; ++k)
    {
        const long a = (long) k * D - D / 2, b = std::min ((long) N, (long) (k + 1) * D - D / 2); if (a >= N) break;
        double s = 0; for (long i = a; i < b; ++i) s += h[(size_t) i] * h[(size_t) i];
        const double db = 10.0 * std::log10 (std::max (s, 1e-30)); if (k > 1 && db < -60.0) break;
        e.push_back (s); c.level.push_back (db);
    }
    if (e.empty() || ! (e[0] > 0)) return;
    while (c.count < (int) e.size() && c.level[(size_t) c.count] > c.level[0] - trainDb) ++c.count;
    if (e.size() > 1) c.phi = std::sqrt (e[1] / e[0]);
    if (c.count >= 3 && std::sqrt (e[2] / e[1]) >= 0.5 * c.phi) { c.train = true; c.phi = std::pow (e[(size_t) c.count - 1] / e[0], 0.5 / (double) (c.count - 1)); }
}
inline Found analyse (const Stereo& x, const Stop& stop = {})
{
    Found f; const auto st = stretch (x); f.start = st.first; f.length = st.second; f.twin = x.l == x.r; Fft F;
    std::vector<Measured> got[2]; int best[2] = { -1, -1 };
    for (int c = 0; c < 2; ++c)
    {
        if (c == 1 && f.twin) { got[1] = got[0]; f.cand[1] = f.cand[0]; }
        else got[c] = measure (F, (c ? x.r : x.l).data() + f.start, f.length, &f.cand[c], stop);
        if (stop && stop()) return {};
        for (int i = 0; i < (int) got[c].size(); ++i) { f.seen[c].push_back (got[c][(size_t) i].rec); if (best[c] < 0 || got[c][(size_t) i].rec.p > got[c][(size_t) best[c]].rec.p) best[c] = i; }
    }
    // The surest lag of each channel. The surer channel counts from pFound up; the other one when it is sure by
    // itself, or when it shows a lag that fits the surer one's echo time.
    const int lead = best[0] < 0 ? 1 : best[1] < 0 ? 0 : got[1][(size_t) best[1]].rec.p > got[0][(size_t) best[0]].rec.p ? 1 : 0;
    if (best[lead] >= 0 && got[lead][(size_t) best[lead]].rec.p >= pFound)
    {
        const double q0 = got[lead][(size_t) best[lead]].rec.q; int use[2] = { -1, -1 }; use[lead] = best[lead]; const int c = 1 - lead;
        for (int i = 0; i < (int) got[c].size(); ++i)
        {
            const Record& r = got[c][(size_t) i].rec; const double ratio = r.q >= q0 ? r.q / q0 : q0 / r.q; const bool fits = std::abs (ratio - std::nearbyint (ratio)) < 0.01;
            if (r.p >= (fits ? pJoin : pOwn) && (use[c] < 0 || r.p > got[c][(size_t) use[c]].rec.p)) use[c] = i;
        }
        for (int k = 0; k < 2; ++k) if (use[k] >= 0) { f.ch[k].found = true; f.ch[k].rec = got[k][(size_t) use[k]].rec; f.ch[k].part = std::move (got[k][(size_t) use[k]].part); }
        f.p = got[lead][(size_t) best[lead]].rec.p; f.sure = sureness (f.p);
    }
    else f.sure = sureNone;
    for (int c = 0; c < 2; ++c) if (f.ch[c].found) { if (c == 1 && f.twin && f.ch[0].found) { f.ch[1].level = f.ch[0].level; f.ch[1].phi = f.ch[0].phi; f.ch[1].count = f.ch[0].count; f.ch[1].train = f.ch[0].train; } else repeatsOf (F, f.ch[c]); }
    return f;
}

// The repeats as the faders have them, for the picture (time after the sound, level against it): what the Feedback
// fader does to them as responseOn() does it, with the added repeats as a plain model (each one the asked share of
// the one before). The tone of the echoes is not in it.
inline double shareAt (const Channel& c, double v) { return std::abs (v - 1.0) < 1e-9 ? c.phi : v < 1.0 ? v * c.phi : feedbackAt (c.train ? c.phi : 0.0, v); }
inline std::vector<Repeat> repeats (const Channel& c, const Shape& sh)
{
    std::vector<Repeat> out; if (! c.found || c.level.empty() || ! (sh.amount > 0)) return out;
    const double step = (double) c.rec.q * sh.time / SR * 1e3, up = 20.0 * std::log10 (sh.amount), v = sh.feedback; const int K = std::min (std::max (1, c.count), (int) c.level.size());
    auto put = [&] (int k, double db) { if (k == 0 || db >= -60.0) out.push_back ({ step * (double) (k + 1), db + up }); return k == 0 || db >= -60.0; };
    if (std::abs (v - 1.0) < 1e-9) { for (size_t k = 0; k < c.level.size(); ++k) put ((int) k, c.level[k]); return out; }
    if (v < 1.0) { const double drop = v > 1e-6 ? 20.0 * std::log10 (v) : -400.0; for (size_t k = 0; k < c.level.size(); ++k) if (! put ((int) k, c.level[k] + drop * (double) k)) break; return out; }
    const double phiT = feedbackAt (c.train ? c.phi : 0.0, v), more = c.train ? 20.0 * std::log10 (phiT / std::max (c.phi, 1e-3)) : 0.0, next = 20.0 * std::log10 (std::max (phiT, 1e-6));
    for (int k = 0; k < K; ++k) put (k, c.level[(size_t) k] + more * k);
    for (int k = K; k < 24; ++k) if (! put (k, c.level[(size_t) K - 1] + more * (K - 1) + next * (k - K + 1))) break;
    return out;
}

// The sound with the echoes as the faders ask. own = true: x is the sound they were found on (reshape, take out).
// own = false: x is another sound (put on). Stopped on the way: comes back empty.
inline Stereo render (const Stereo& x, const Found& f, const Shape& sh, bool own, const Stop& stop = {})
{
    if (! f.any()) return x;
    Fft F; Vec h[2];
    for (int c = 0; c < 2; ++c)
    {
        if (stop && stop()) return {};
        if (! f.ch[c].found) h[c] = { 1.0 };
        else if (c == 1 && f.twin && f.ch[0].found) h[1] = h[0];
        else h[c] = response (F, f.ch[c], sh, own);
    }
    return tone::filter (x, h[0], h[1], stop);
}

} // namespace echo
} // namespace dr
