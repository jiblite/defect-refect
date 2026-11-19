#pragma once
// The crush engine (BITCRUSH): a sound whose samples were rounded to a grid of values (fewer bits) and held for
// several samples each (a lower sample rate):
//     wet[t] = Q (dry[b(t)]),   Q(v) = offset + step * round ((v - offset) / step),   b(t) = the last hold boundary at or before t
//   find()    reads the grid and the hold off one sound
//   onto()    another sound through the same crusher: the same grid of values (in absolute terms, as the same crusher
//             would) and the same hold; Amount 0 = none, 1 = as found, 2 = what the crusher changes twice over; the
//             steps and the hold can be made finer or coarser
//   own()     the found sound with the group's faders moved (see there for what can be had)
// How:
//  1. The grid. A sound on a grid of values has an amplitude distribution made of spikes a step apart, and the Fourier
//     transform of that distribution (how well exp (2 pi i u v) lines up over all samples) has a line at u = 1 / step:
//     a narrow one, as wide as one over the range of the values, standing out of its surroundings. The step and the
//     offset are then made exact by a straight line through the values themselves (grid).
//  2. The hold: the places where the sound changes at all. With a hold of k samples they all lie k apart or a multiple
//     of it; with a hold that is not a whole number (a sample rate that does not divide the file's) they lie on the
//     pattern ceil (j P + c), whose slope and start are fitted (hold).
//  3. A crusher that mixes plain sound back in, with soft steps:
//         v = m hold(x) + (1 - m) x,    wet = (1 - m) v + m Q(v)
//     Q: the staircase, flat on the plateaus with a sine blend between them over the share aa of a step. The values
//     then lie around the grid values, not on them: each grid value has a stretch of width (1 - m)(1 - aa) of a step
//     filled evenly, with sharp ends. Read off: the step at which the values folded onto one step show the sharpest
//     ends (sharpen), where those ends lie and how fast the values thin out beyond them (box, edge), the hold from
//     where the large changes fall (rhythm), and with a hold the blend from the trace a wrong one leaves at the hold's
//     boundaries (soft). Because of the plain share nothing is lost: the map is one to one, and softUndo gives the
//     sound before the crusher.
// What it does not do: a crusher with a filter after it (its grid is read where it still shows, and put on as a bare
// one); which of the dry samples a held value was, and the crusher's rounding rule (nearest, down, towards zero), which
// the sound does not tell: the nearest value and the sample at the boundary are taken.
// A long sound: the settings are read from its loudest 30 s.
// A port of the Python prototype Research/crush/crush.py (Tools/crush_parity.py). Plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <vector>

#include "Dsp.h"
#include "Fft.h"

namespace dr
{
namespace crush
{
using Stop = std::function<bool()>;                     // asked now and then while measuring: true = give up (the answer is then not used)
constexpr double BITS_MAX = 13.0;                       // a grid finer than this many bits (against full scale, 2 / step) is an ordinary file, not a crush
constexpr double ON = 0.02;                             // a sample is on the grid if it is within this share of a step of a grid value
constexpr double SHARE = 0.995;                         // .. and this share of the live samples has to be
constexpr long HB = 1L << 18;                           // bins of the amplitude histogram
constexpr int KMAX = 256;                               // the longest hold looked for (samples)
constexpr double SOFT_HALF = 0.002;                     // the crusher with plain sound mixed in fills a stretch at least this wide (in steps) around each grid value
constexpr double SOFT_DENSE = 8.0;                      // .. and counts if the values lie this many times denser around the grid values than between them
constexpr int NBF = 4096;                               // bins of a step when the values are folded onto one
constexpr double LONG = 30.0;                           // seconds the settings are read from
constexpr double TWO_PI = 2.0 * PI;

struct Grid { bool ok = false; double step = 0.0, offset = 0.0, share = 0.0, bits = 0.0; long levels = 0; };
struct Hold { double P = 1.0, c = 0.0, share = 1.0, room = 0.0; long changes = 0; };
struct Found
{
    bool found = false; size_t n = 0; double rate = 48000.0; long from = 0;      // from: where the stretch the settings were read from starts
    bool soft = false;                                      // plain sound mixed in (the sound before the crusher can be had)
    bool through = false;                                   // the grid shows only in part: something came after the crusher
    double step[2] = { 0.0, 0.0 }, offset[2] = { 0.0, 0.0 };       // the grid of each side (step 0: none)
    double bits = 24.0, P = 1.0, c = 0.0;                   // bits against full scale; the hold: a held value lasts P samples, the boundaries are ceil (j P + c)
    double mix = 1.0, blend = 0.0;                          // soft: m and aa
    double half = 0.0, inside = 0.0, dense = 0.0, lam = 0.0, rhythm = 0.0, aaMass = 0.0, aaLam = 0.0;
    Grid grids[2]; Hold hold;
    std::vector<float> xl, xr;                              // soft: the sound before the crusher
    bool hasGrid() const { return step[0] > 0.0 || step[1] > 0.0; }
    bool hasHold() const { return P > 1.0; }
};

inline double frac (double x) { return x - std::floor (x); }
inline double sign (double x) { return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0; }
// numpy's percentile (linear between the two nearest); v is put in order
inline double percentile (Vec& v, double q)
{
    if (v.empty()) return 0.0;
    std::sort (v.begin(), v.end()); const double pos = q / 100.0 * (double) (v.size() - 1); const size_t i = (size_t) std::floor (pos);
    return i + 1 < v.size() ? v[i] + (pos - (double) i) * (v[i + 1] - v[i]) : v[i];
}

// ---------------------------------------------------------------- the grid
// The grid of values of one side. offset is the grid value nearest zero (between -step / 2 and step / 2).
inline Grid grid (const Vec& v)
{
    Grid G; Vec live; live.reserve (v.size()); for (double x : v) if (x != 0.0) live.push_back (x);
    if (live.empty()) live = v;
    const size_t n = live.size(); if (n < 1000) return G;
    double lo = live[0], hi = live[0]; for (double x : live) { lo = std::min (lo, x); hi = std::max (hi, x); }
    if (! (hi - lo > 1e-9)) return G;
    const double A = std::max (std::abs (lo), std::abs (hi)) * (1.0 + 1e-9), h = 2.0 * A / (double) HB; const size_t nf = (size_t) (4 * HB);
    Vec re (nf, 0.0), im (nf, 0.0);
    for (double x : live) re[(size_t) std::min ((long) ((x + A) / h), HB - 1)] += 1.0;
    fft (re, im); const long lenF = (long) (nf / 2 + 1); Vec F ((size_t) lenF);                 // F[j]: how well exp (2 pi i u v) lines up over the samples, u = j / (nf h)
    for (long j = 0; j < lenF; ++j) F[(size_t) j] = std::hypot (re[(size_t) j], im[(size_t) j]) / (double) n;
    const double du = 1.0 / ((double) nf * h); const long j0 = std::max ((long) std::ceil (0.9 / (hi - lo) / du), 2L), j1 = std::min ((long) (std::pow (2.0, BITS_MAX - 1.0) / du) + 1, lenF - 2);      // steps from the whole range down to BITS_MAX
    if (j1 - j0 < 8) return G;
    // The small samples line up for every u up to one over their size, grid or no grid, and in most sounds they are
    // the many: F is high there anyway. A grid shows as a narrow line on top of that (as wide as one over the range of
    // the values): z = how much of what does not line up anyway lines up at the line, against the lowest F close by on
    // either side.
    const long W = std::max ((long) std::ceil (1.0 / (hi - lo) / du), 2L), g = 2 * W, sp = 6 * W, w = sp - g + 1;
    auto at = [&] (long i) { return F[(size_t) std::clamp (i, 0L, lenF - 1)]; };              // F with its ends carried on
    // mn[i]: lowest of F[i - sp .. i - g] (the same window, moved on by sp + g, is the one to the right)
    const long len = lenF + sp + g; Vec mn ((size_t) len); std::vector<long> dq; dq.reserve ((size_t) w + 1); size_t head = 0;
    for (long i = -sp; i < len - g; ++i)                                                      // i: the window's last sample, F[i - w + 1 .. i]
    {
        while (dq.size() > head && at (dq.back()) >= at (i)) dq.pop_back();
        dq.push_back (i); if (dq[head] <= i - w) ++head;
        if (head > 4096) { dq.erase (dq.begin(), dq.begin() + (long) head); head = 0; }
        const long k = i + g; if (k >= 0 && k < len) mn[(size_t) k] = at (dq[head]);          // window F[k - sp .. k - g]
    }
    double top = -1e300; Vec z ((size_t) (j1 - j0 + 1));
    for (long j = j0; j <= j1; ++j) { const double base = std::max (mn[(size_t) j], mn[(size_t) (j + sp + g)]); z[(size_t) (j - j0)] = (F[(size_t) j] - base) / std::max (1.0 - base, 1e-9); top = std::max (top, z[(size_t) (j - j0)]); }
    if (top < 0.5) return G;
    long j = -1; for (size_t i = 1; i + 1 < z.size(); ++i) if (z[i] >= z[i - 1] && z[i] >= z[i + 1] && z[i] >= 0.8 * top) { j = (long) i + j0; break; }
    if (j < 0) return G;
    { long b = j - 1; for (long q = j; q <= j + 1; ++q) if (F[(size_t) q] > F[(size_t) b]) b = q; j = b; }
    const double y0 = F[(size_t) (j - 1)], y1 = F[(size_t) j], y2 = F[(size_t) (j + 1)], den0 = y0 - 2.0 * y1 + y2;
    double step = 1.0 / (((double) j + (den0 < 0.0 ? 0.5 * (y0 - y2) / den0 : 0.0)) * du);
    double sc = 0.0, ss = 0.0; for (double x : live) { sc += std::cos (TWO_PI * x / step); ss += std::sin (TWO_PI * x / step); }
    double off = std::atan2 (ss, sc) / TWO_PI * step;
    for (double win : { 0.25, 0.25, 0.12, 0.06, 0.03, 0.03 })                                 // the straight line through the values: value = offset + step * (whole number); the window narrows, so that values which only lie around the grid (a crusher that mixes plain sound in) do not pull it
    {
        double sk = 0.0, sv = 0.0; long m = 0;
        for (double x : live) { const double k = std::nearbyint ((x - off) / step), r = (x - off) / step - k; if (std::abs (r) < win) { sk += k; sv += x; ++m; } }
        if (m < 100) break;
        const double km = sk / (double) m, vm = sv / (double) m; double den = 0.0, num = 0.0;
        for (double x : live) { const double k = std::nearbyint ((x - off) / step), r = (x - off) / step - k; if (std::abs (r) < win) { den += (k - km) * (k - km); num += (k - km) * (x - vm); } }
        if (! (den > 0.0)) return G;
        step = num / den; off = vm - step * km;
    }
    off -= step * std::nearbyint (off / step); long on = 0; double kmin = 1e300, kmax = -1e300;
    for (double x : live) { const double k = std::nearbyint ((x - off) / step), r = (x - off) / step - k; on += std::abs (r) < ON; kmin = std::min (kmin, k); kmax = std::max (kmax, k); }
    G.ok = true; G.step = step; G.offset = off; G.share = (double) on / (double) n; G.levels = (long) (kmax - kmin) + 1; G.bits = std::log2 (2.0 / step); return G;
}

// ---------------------------------------------------------------- the hold
// The hold pattern from the places where the sound changes: a held value lasts P samples (1: no hold), the boundaries
// are ceil (j P + c), j = 0, 1, ..; share: how many of the changes lie on it.
inline Hold hold (const Stereo& y)
{
    const long n = (long) y.size(); std::vector<long> ch;
    for (long t = 1; t < n; ++t) if (y.l[(size_t) t] != y.l[(size_t) t - 1] || y.r[(size_t) t] != y.r[(size_t) t - 1]) ch.push_back (t);
    Hold none; none.changes = (long) ch.size(); const size_t m = ch.size();
    if (m < 50) return none;
    std::vector<long> cnt;
    for (int k = KMAX; k > 1; --k)                                                            // a whole number of samples: the longest that fits
    {
        cnt.assign ((size_t) k, 0); for (long t : ch) ++cnt[(size_t) (t % k)];
        size_t ph = 0; for (size_t i = 1; i < cnt.size(); ++i) if (cnt[i] > cnt[ph]) ph = i;
        if ((double) cnt[ph] >= 0.99 * (double) m) { Hold H; H.P = (double) k; H.c = (double) ph; H.share = (double) cnt[ph] / (double) m; H.changes = (long) m; return H; }
    }
    Vec gaps (m - 1); for (size_t i = 0; i + 1 < m; ++i) gaps[i] = (double) (ch[i + 1] - ch[i]);
    Vec gs = gaps; const long g = (long) percentile (gs, 2.0);
    if (g < 2) return none;                                                                   // changes one sample apart: no hold
    // not a whole number: the line in the spectrum of the changes, then a straight-line fit of position against count
    size_t nf = 1; while ((long) nf < n) nf <<= 1;
    Vec re (nf, 0.0), im (nf, 0.0); const double mean = (double) m / (double) n;
    for (long t = 0; t < n; ++t) re[(size_t) t] = -mean;
    for (long t : ch) re[(size_t) t] += 1.0;
    fft (re, im); const long half = (long) (nf / 2);
    auto S = [&] (long j) { return std::hypot (re[(size_t) j], im[(size_t) j]); };
    const long lo = std::max ((long) ((double) nf / ((double) g + 1.5)), 1L), hi = std::min ((long) ((double) nf / std::max ((double) g - 0.5, 1.5)) + 1, half - 1);
    if (hi < lo) return none;
    long j = lo; for (long q = lo; q <= hi; ++q) if (S (q) > S (j)) j = q;
    const double y0 = S (j - 1), y1 = S (j), y2 = S (j + 1), den0 = y0 - 2.0 * y1 + y2; double P = (double) nf / ((double) j + (den0 < 0.0 ? 0.5 * (y0 - y2) / den0 : 0.0));
    Vec idx (m, 0.0); double chm = 0.0; for (long t : ch) chm += (double) t; chm /= (double) m;
    for (int it = 0; it < 3; ++it)
    {
        for (size_t i = 1; i < m; ++i) idx[i] = idx[i - 1] + std::nearbyint (gaps[i - 1] / P);
        double im_ = 0.0; for (double v : idx) im_ += v; im_ /= (double) m; double den = 0.0, num = 0.0;
        for (size_t i = 0; i < m; ++i) { den += (idx[i] - im_) * (idx[i] - im_); num += (idx[i] - im_) * ((double) ch[i] - chm); }
        if (! (den > 0.0)) return none;
        P = num / den;
    }
    // the start: ceil (j P + c) = position  <=>  position - 1 - j P < c <= position - j P, for every change; P moved a hair so that the room for c is widest
    auto room = [&] (double p, double& top, double& bot) { top = 1e300; bot = -1e300; for (size_t i = 0; i < m; ++i) { const double r = (double) ch[i] - idx[i] * p; top = std::min (top, r); bot = std::max (bot, r - 1.0); } };
    double a = P * (1.0 - 2e-5), b = P * (1.0 + 2e-5), h1, l1, h2, l2;
    for (int it = 0; it < 60; ++it)
    {
        const double p1 = a + (b - a) / 3.0, p2 = b - (b - a) / 3.0; room (p1, h1, l1); room (p2, h2, l2);
        if (h1 - l1 < h2 - l2) a = p1; else b = p2;
    }
    P = 0.5 * (a + b); room (P, h1, l1); const double c0 = 0.5 * (h1 + l1); long on = 0;
    for (size_t i = 0; i < m; ++i) on += std::ceil (idx[i] * P + c0) == (double) ch[i];
    if (! (P >= 2.0)) return none;
    Hold H; H.P = P; H.c = c0; H.share = (double) on / (double) m; H.changes = (long) m; H.room = h1 - l1; return H;
}

// For every sample whether a held value is taken there (always at sample 0 and at the last one).
inline std::vector<char> boundaries (long n, double P, double c)
{
    std::vector<char> b ((size_t) std::max (n, 0L), 0); if (n <= 0) return b;
    if (P <= 1.0) { std::fill (b.begin(), b.end(), 1); return b; }
    const long j0 = (long) std::floor (-c / P) - 1, j1 = (long) std::ceil (((double) n - c) / P) + 1;
    for (long j = j0; j <= j1; ++j) b[(size_t) std::clamp ((long) std::ceil ((double) j * P + c - 1e-9), 0L, n - 1)] = 1;
    return b;
}
// q with every sample replaced by the one at the last hold boundary at or before it.
inline void held (Stereo& q, double P, double c)
{
    if (P <= 1.0) return;
    const long n = (long) q.size(); const std::vector<char> b = boundaries (n, P, c); double l = 0.0, r = 0.0;
    for (long t = 0; t < n; ++t) { if (b[(size_t) t] || t == 0) { l = q.l[(size_t) t]; r = q.r[(size_t) t]; } else { q.l[(size_t) t] = l; q.r[(size_t) t] = r; } }
}

// ---------------------------------------------------------------- the crusher that mixes plain sound back in
struct Soft { double mix = 1.0, blend = 0.0, step[2] = { 0.0, 0.0 }, offset[2] = { 0.0, 0.0 }, P = 1.0, c = 0.0; };
// The staircase: a whole number of steps on the plateaus and a sine blend between them over the share aa of a step.
inline double qsoft (double u, double aa)
{
    const double k = std::nearbyint (u), fr = u - k; if (aa <= 1e-9) return k;
    const double a1 = (1.0 - aa) / 2.0; return std::abs (fr) > a1 ? k + sign (fr) * 0.5 * (std::sin (PI * (std::abs (fr) - a1) / aa - PI / 2.0) + 1.0) : k;
}
// fr with (1 - m) fr + m S(fr) = rho (rho between -0.5 and 0.5; S: the blend, 0 on the plateau).
inline double g0inv (double rho, double m, double aa)
{
    const double a = std::min (std::abs (rho), 0.5), a1 = (1.0 - aa) / 2.0; double fr = std::min (a / (1.0 - m), a1);
    if (a > (1.0 - m) * a1)
    {
        if (aa <= 1e-9) fr = 0.5;
        else
        {
            // (1 - m) x + m (1 - cos (pi (x - a1) / aa)) / 2 = a, which rises all the way: Newton's steps, kept between the two ends that are known
            double lo = a1, hi = 0.5, x = 0.5 * (lo + hi);
            for (int it = 0; it < 60; ++it)
            {
                const double th = PI * (x - a1) / aa, f = (1.0 - m) * x + m * 0.5 * (1.0 - std::cos (th)) - a;
                if (f < 0.0) lo = x; else hi = x;
                double xn = x - f / ((1.0 - m) + m * 0.5 * std::sin (th) * PI / aa);
                if (! (xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
                if (std::abs (xn - x) < 1e-15 || hi - lo < 1e-15) { x = xn; break; }
                x = xn;
            }
            fr = x;
        }
    }
    return sign (rho) * fr;
}
inline double softV (double y, double step, double off, double m, double aa) { const double w = (y - off) / step, K = std::nearbyint (w); return off + step * (K + g0inv (w - K, m, aa)); }
// x through the crusher.
inline Stereo softForward (const Stereo& x, const Soft& f)
{
    Stereo hx = x; held (hx, f.P, f.c); Stereo o = x; const double m = f.mix;
    for (int ch = 0; ch < 2; ++ch)
    {
        const Vec& in = ch ? x.r : x.l; const Vec& hh = ch ? hx.r : hx.l; Vec& out = ch ? o.r : o.l; const double st = f.step[ch], of = f.offset[ch];
        for (size_t i = 0; i < in.size(); ++i) { const double v = m * hh[i] + (1.0 - m) * in[i]; out[i] = st > 0.0 ? (1.0 - m) * v + m * (of + st * qsoft ((v - of) / st, f.blend)) : v; }
    }
    return o;
}
// The sound before the crusher.
inline Stereo softUndo (const Stereo& y, const Soft& f)
{
    const long n = (long) y.size(); Stereo v = y; const double m = f.mix;
    for (int ch = 0; ch < 2; ++ch) { const Vec& in = ch ? y.r : y.l; Vec& out = ch ? v.r : v.l; const double st = f.step[ch], of = f.offset[ch]; if (st > 0.0) for (size_t i = 0; i < in.size(); ++i) out[i] = softV (in[i], st, of, m, f.blend); }
    if (f.P <= 1.0) return v;
    const std::vector<char> b = boundaries (n, f.P, f.c); Stereo x = v; double l = 0.0, r = 0.0;
    for (long t = 0; t < n; ++t)
    {
        if (b[(size_t) t] || t == 0) { l = v.l[(size_t) t]; r = v.r[(size_t) t]; }
        else { x.l[(size_t) t] = (v.l[(size_t) t] - m * l) / (1.0 - m); x.r[(size_t) t] = (v.r[(size_t) t] - m * r) / (1.0 - m); }
    }
    return x;
}
// A hold under plain sound mixed in: the changes are large where a held value is taken and small between. The largest
// change of every k samples lies at the same place (start) in this share of them.
struct Rhythm { int k = 1; int start = 0; double share = 0.0; };
inline Rhythm rhythm (const Stereo& y)
{
    const long n = (long) y.size(); Vec e ((size_t) std::max (n, 1L), 0.0); Rhythm best;
    for (long t = 1; t < n; ++t) { const double dl = y.l[(size_t) t] - y.l[(size_t) t - 1], dr_ = y.r[(size_t) t] - y.r[(size_t) t - 1]; e[(size_t) t] = dl * dl + dr_ * dr_; }
    std::vector<long> c;
    for (int k = 2; k <= KMAX; ++k)
    {
        const long rows = n / k; long live = 0; c.assign ((size_t) k, 0);
        for (long r = 0; r < rows; ++r)
        {
            const double* p = &e[(size_t) (r * k)]; int am = 0; for (int i = 1; i < k; ++i) if (p[i] > p[am]) am = i;
            if (p[am] > 1e-18) { ++live; ++c[(size_t) am]; }
        }
        if (live < 50) break;
        int ph = 0; for (int i = 1; i < k; ++i) if (c[(size_t) i] > c[(size_t) ph]) ph = i;
        const double sh = (double) c[(size_t) ph] / (double) live;
        if (sh > best.share + 0.02 || (sh >= best.share - 0.02 && best.k > 1 && k % best.k == 0 && sh >= 0.85)) best = { k, ph, sh };
    }
    return best;
}
inline Vec folded (const Vec& v, double st)
{
    Vec h ((size_t) NBF, 0.0); for (double x : v) h[(size_t) std::min ((long) (frac (x / st) * NBF), (long) NBF - 1)] += 1.0; return h;
}
// The steepest way up and the steepest way down of a folded distribution (smoothed over 3 bins, read around the
// circle): the bin where it goes up, the bin where it comes down, how steep both are together.
struct Ends { int up = 0, dn = 0; double steep = 0.0; };
inline Ends ends (const Vec& h)
{
    const int N = (int) h.size(); Vec g ((size_t) N), d ((size_t) N);
    for (int j = 0; j < N; ++j) g[(size_t) j] = (h[(size_t) ((j + N - 1) % N)] + h[(size_t) j] + h[(size_t) ((j + 1) % N)]) / 3.0;
    for (int j = 0; j < N; ++j) d[(size_t) j] = g[(size_t) ((j + 2) % N)] - g[(size_t) ((j + N - 2) % N)];
    Ends E; for (int j = 1; j < N; ++j) { if (d[(size_t) j] > d[(size_t) E.up]) E.up = j; if (d[(size_t) j] < d[(size_t) E.dn]) E.dn = j; }
    E.steep = d[(size_t) E.up] - d[(size_t) E.dn]; return E;
}
// The step made exact for values that lie around the grid and not on it. Inside each step the values lean towards zero
// (there are more small samples than large ones), which makes every mean too small; but the stretch around a grid value
// that the crusher fills has sharp ends, at the same place for every grid value. So: the step at which the values,
// folded onto one step, show the sharpest ends.
inline double sharpen (const Vec& live, double step)
{
    Vec K (live.size()); double Kmax = 0.0; for (size_t i = 0; i < live.size(); ++i) { K[i] = std::abs (std::nearbyint (live[i] / step)); Kmax = std::max (Kmax, K[i]); }
    struct Stage { double kmax, span; int num; };
    for (const Stage& s : { Stage { 4.0, 0.02, 161 }, Stage { 32.0, 5e-4, 81 }, Stage { 1073741824.0, 5e-5, 81 }, Stage { 1073741824.0, 2e-6, 41 } })
    {
        double kmax = s.kmax;
        auto count = [&] { long c = 0; for (double k : K) c += k <= kmax && k >= 1.0; return c; };
        while (kmax < 1073741824.0 && count() < 5000 && kmax < Kmax) kmax *= 2.0;              // enough values to see the ends
        Vec v; for (size_t i = 0; i < live.size(); ++i) if (K[i] <= kmax && K[i] >= 1.0) v.push_back (live[i]);
        if (v.size() < 500) continue;
        double best = -1e300, at = step;
        for (int i = 0; i < s.num; ++i) { const double cand = step * (1.0 + (-s.span + 2.0 * s.span * (double) i / (double) (s.num - 1))), sc = ends (folded (v, cand)).steep; if (sc > best) { best = sc; at = cand; } }
        if (best > 1.1 * ends (folded (v, step)).steep || s.span < 1e-3) step = at;           // (the wide look only moves the step if the ends get clearly sharper)
    }
    return step;
}
// Where the dense part ends, from the distances d (one side, in order) of the values from the grid value: inside the
// values lie evenly, outside they thin out as 1 / sqrt (1 + (d - hw) / lam) (the blend begins with the plateau's own
// slope). hw and lam of the best fit (likelihood) between 0.6 hw0 and 1.6 hw0.
inline void edge (const Vec& all, double hw0, double& hwOut, double& lamOut)
{
    const double lo = 0.6 * hw0, hi = 1.6 * hw0; Vec d; for (double x : all) if (x >= lo && x <= hi) d.push_back (x);
    const double n = (double) d.size(); hwOut = hw0; lamOut = 0.0; if (d.size() < 100) return;
    double best = -9e99, bh = hw0, bl = 1e-6, span = 0.2 * hw0, ctr = hw0; Vec lg (d.size());
    for (int rnd = 0; rnd < 4; ++rnd)
    {
        for (int i = 0; i < 41; ++i)
        {
            const double hw = ctr + (-span + 2.0 * span * (double) i / 40.0); if (! (lo < hw && hw < hi)) continue;
            const size_t from = (size_t) (std::lower_bound (d.begin(), d.end(), hw) - d.begin());
            for (int j = 0; j < 23; ++j)
            {
                const double lam = hw0 * std::pow (10.0, -6.0 + 5.5 * (double) j / 22.0), il = 1.0 / lam; double s = 0.0, pr = 1.0; int k = 0;
                for (size_t q = from; q < d.size(); ++q) { pr *= 1.0 + (d[q] - hw) * il; if (++k == 32) { s += std::log (pr); pr = 1.0; k = 0; } }      // the sum of the logarithms, 32 at a time as the logarithm of their product (each factor is between 1 and a few million)
                s += std::log (pr);
                const double ll = -0.5 * s - n * std::log ((hw - lo) + 2.0 * lam * (std::sqrt (1.0 + (hi - hw) / lam) - 1.0));
                if (ll > best) { best = ll; bh = hw; bl = lam; }
            }
        }
        ctr = bh; span *= 0.1;
    }
    hwOut = bh; lamOut = bl;
}
// The stretch around each grid value that the values fill: the grid's offset, its half width in steps, the share of
// the values inside, how many times denser inside than outside, and how fast they thin out beyond the ends; from the
// values of all grid values but the one at zero (silence sits there).
struct Box { bool ok = false; double offset = 0.0, half = 0.0, inside = 0.0, dense = 0.0, lam = 0.0; };
inline Box box (const Vec& live, double step)
{
    Box B; Vec v; for (double x : live) if (std::nearbyint (x / step) != 0.0) v.push_back (x);
    if (v.size() < 500) return B;
    const Ends E = ends (folded (v, step)); const double w = (double) (((E.dn - E.up) % NBF + NBF) % NBF);
    double c = frac (((double) E.up + w / 2.0) / (double) NBF), hw = w / 2.0 / (double) NBF, lam = 0.0;
    if (! (2.0 / (double) NBF < hw && hw < 0.25)) return B;
    Vec right, left;
    for (int it = 0; it < 2; ++it)
    {
        right.clear(); left.clear();
        for (double x : v) { const double d = frac (x / step - c + 0.5) - 0.5; if (d >= 0.0) right.push_back (d); else left.push_back (-d); }
        std::sort (right.begin(), right.end()); std::sort (left.begin(), left.end());
        double r, l1, l, l2; edge (right, hw, r, l1); edge (left, hw, l, l2); c = frac (c + 0.5 * (r - l)); hw = 0.5 * (r + l); lam = 0.5 * (l1 + l2);
    }
    long in = 0; for (double x : v) in += std::abs (frac (x / step - c + 0.5) - 0.5) <= hw;
    B.inside = (double) in / (double) v.size(); const double off = frac (c + 0.5) - 0.5;
    B.dense = B.inside / std::max (2.0 * hw, 1e-9) / std::max ((1.0 - B.inside) / std::max (1.0 - 2.0 * hw, 1e-9), 1e-9);
    B.ok = true; B.offset = off * step; B.half = hw; B.lam = lam; return B;
}
// The crusher that mixes plain sound in, read off the sound (G: the grids of the two sides). ok = false: not one.
struct SoftRead { bool ok = false; Soft f; double half = 0.0, inside = 0.0, dense = 0.0, lam = 0.0, rhythm = 0.0, aaMass = 0.0, aaLam = 0.0, bits = 0.0; };
inline SoftRead soft (const Stereo& y, const Grid G[2], double rate, const Stop& stop = {})
{
    SoftRead R; if (! (G[0].ok && G[1].ok)) return R;
    const long n = (long) y.size(); double st[2], of[2], hws = 0.0, ins = 0.0, lams = 0.0, dens = 1e300;
    auto side = [&] (int ch, double& s1)                                                      // one side: its step made exact, then the stretch its values fill
    {
        const Vec& v = ch ? y.r : y.l; Vec live; live.reserve (v.size()); for (double x : v) if (x != 0.0) live.push_back (x);
        if (live.size() < 2000) return Box();
        s1 = sharpen (live, G[ch].step); if (stop && stop()) return Box();
        return box (live, s1);
    };
    double s1[2] = { 0.0, 0.0 }; auto other = std::async (std::launch::async, [&] { return side (1, s1[1]); });      // the two sides at the same time
    const Box bx[2] = { side (0, s1[0]), other.get() };
    for (int ch = 0; ch < 2; ++ch)
    {
        const Box& b = bx[ch]; if (! b.ok) return R;
        st[ch] = s1[ch]; of[ch] = b.offset; hws += 0.5 * b.half; ins += 0.5 * b.inside; lams += 0.5 * b.lam; dens = std::min (dens, b.dense);
    }
    if (stop && stop()) return R;
    const double hw = hws, inside = ins, lam = lams;
    auto mixOf = [&] (double a) { return std::clamp (1.0 - 2.0 * hw / std::max (1.0 - a, 0.2), 0.3, 0.9995); };
    const double aaMass = std::clamp (1.0 - inside, 0.0, 0.8); double m = mixOf (aaMass), aaLam = aaMass;      // half width (1 - m) (1 - aa) / 2; share inside 1 - aa where the values lie evenly over a step
    for (int it = 0; it < 6; ++it) { const double t = PI * std::sqrt (lam * m) / (2.0 * hw); aaLam = t / (1.0 + t); m = mixOf (aaLam); }      // lam = aa^2 (1 - m)^2 / (m pi^2): the blend's width from how fast the values thin out beyond the ends
    double least = 1e300;
    for (int ch = 0; ch < 2; ++ch) { const Vec& v = ch ? y.r : y.l; Vec k (v.size()); for (size_t i = 0; i < v.size(); ++i) k[i] = std::nearbyint (v[i] / st[ch]); const double a = percentile (k, 1.0), b = percentile (k, 99.0); least = std::min (least, b - a); }
    const bool fine = least >= 40.0; double aa = fine ? aaMass : aaLam; m = mixOf (aa);                         // the share where the values spread over many steps, the thinning where over few
    const Rhythm rh = rhythm (y); const double P = rh.share >= 0.85 ? (double) rh.k : 1.0, c = rh.share >= 0.85 ? (double) rh.start : 0.0;
    if (P > 1.0)
    {
        // With a hold the blend can be made exact: the half width ties mix and blend together, and of the pairs it
        // allows the right one leaves no trace of the hold in the undone sound: its second difference is as large
        // across the hold's boundaries as between them (any other pair leaves teeth, and the boundaries stand out).
        const long SRl = std::lround (rate), ip = (long) P, pl = (SRl / 2 / ip) * ip; Stereo yy;
        if (n <= 3 * SRl) yy = y;
        else for (int k = 0; k < 6; ++k)                                                       // six pieces of a long sound, each a whole number of holds long and starting on one: the hold's pattern goes on from one to the next
        {
            const long at = (long) ((double) (n - pl) * (double) k / 5.0) / ip * ip;
            yy.l.insert (yy.l.end(), y.l.begin() + at, y.l.begin() + at + pl); yy.r.insert (yy.r.end(), y.r.begin() + at, y.r.begin() + at + pl);
        }
        const long ny = (long) yy.size(); const std::vector<char> isb = boundaries (ny, P, c);
        auto teeth = [&] (double a)
        {
            Soft f; f.mix = 1.0 - 2.0 * hw / (1.0 - a); f.blend = a; f.P = P; f.c = c; for (int ch = 0; ch < 2; ++ch) { f.step[ch] = st[ch]; f.offset[ch] = of[ch]; }
            const Stereo o = softUndo (yy, f); double sb = 0.0, so = 0.0; long nb = 0, no = 0;
            for (long i = 1; i + 1 < ny; ++i)
            {
                const double dl = o.l[(size_t) i + 1] - 2.0 * o.l[(size_t) i] + o.l[(size_t) i - 1], dr_ = o.r[(size_t) i + 1] - 2.0 * o.r[(size_t) i] + o.r[(size_t) i - 1], e = dl * dl + dr_ * dr_;
                if (isb[(size_t) i] || isb[(size_t) i + 1] || i == 0) { sb += e; ++nb; } else { so += e; ++no; }
            }
            if (nb == 0 || no == 0) return std::numeric_limits<double>::quiet_NaN();
            return (sb / (double) nb) / std::max (so / (double) no, 1e-300);
        };
        const double base = teeth (aa); double bt = base, ba = aa, span = 0.06;
        for (int rnd = 0; rnd < 3; ++rnd)
        {
            const double ctr = ba; double as[13]; bool use[13]; std::vector<std::future<double>> jobs;      // thirteen blends around the best so far, measured at the same time
            for (int i = 0; i < 13; ++i)
            {
                const double a = ctr + (-span + 2.0 * span * (double) i / 12.0); as[i] = a; use[i] = a >= 0.0 && a <= 0.8 && 1.0 - 2.0 * hw / (1.0 - a) >= 0.3;
                if (use[i]) jobs.push_back (std::async (std::launch::async, teeth, a));
            }
            size_t j = 0; for (int i = 0; i < 13; ++i) if (use[i]) { const double t = jobs[j++].get(); if (t < bt) { bt = t; ba = as[i]; } }
            if (stop && stop()) return R;
            span /= 6.0;
        }
        if (bt < 0.98 * base) { aa = ba; m = std::clamp (1.0 - 2.0 * hw / (1.0 - aa), 0.3, 0.9995); }
    }
    R.ok = true; R.f.mix = m; R.f.blend = aa; R.f.P = P; R.f.c = c; for (int ch = 0; ch < 2; ++ch) { R.f.step[ch] = st[ch]; R.f.offset[ch] = of[ch]; }
    R.half = hw; R.inside = inside; R.dense = dens; R.lam = lam; R.rhythm = rh.share; R.aaMass = aaMass; R.aaLam = aaLam; R.bits = 0.5 * (std::log2 (2.0 / st[0]) + std::log2 (2.0 / st[1]));
    return R;
}

// ---------------------------------------------------------------- find
inline Soft softOf (const Found& F) { Soft f; f.mix = F.mix; f.blend = F.blend; f.P = F.P; f.c = F.c; for (int ch = 0; ch < 2; ++ch) { f.step[ch] = F.step[ch]; f.offset[ch] = F.offset[ch]; } return f; }
// The stretch the settings are read from: the whole sound up to 30 s, of a longer one its loudest 30 s.
inline long part (const Stereo& y, double rate)
{
    const long n = (long) y.size(), len = std::lround (LONG * rate), hop = std::lround (rate); if (n <= len) return 0;
    Vec c ((size_t) n + 1, 0.0); for (long i = 0; i < n; ++i) c[(size_t) i + 1] = c[(size_t) i] + y.l[(size_t) i] * y.l[(size_t) i] + y.r[(size_t) i] * y.r[(size_t) i];
    long best = 0; double e = -1.0; for (long a = 0; a + len <= n; a += hop) { const double v = c[(size_t) (a + len)] - c[(size_t) a]; if (v > e) { e = v; best = a; } }
    return best;
}
inline Found findPart (const Stereo& y, double rate, const Stop& stop)
{
    Found F; F.n = y.size(); F.rate = rate;
    if ((long) y.size() < std::lround (0.25 * rate)) return F;
    F.grids[0] = grid (y.l); if (stop && stop()) return F;
    F.grids[1] = grid (y.r); if (stop && stop()) return F;
    F.hold = hold (y);
    bool ok[2]; for (int ch = 0; ch < 2; ++ch) ok[ch] = F.grids[ch].ok && F.grids[ch].share >= SHARE && F.grids[ch].bits <= BITS_MAX;
    const bool hasHold = F.hold.P >= 2.0 && F.hold.share >= 0.99 && F.hold.changes >= 50;
    if (! ((ok[0] && ok[1]) || hasHold))
    {
        const SoftRead R = soft (y, F.grids, rate, stop); if (stop && stop()) return F;
        if (R.ok && R.dense >= SOFT_DENSE && R.inside >= 0.3 && R.half <= 0.2 && R.bits <= BITS_MAX)
        {
            F.found = true; F.bits = R.bits; F.half = R.half; F.inside = R.inside; F.dense = R.dense; F.lam = R.lam; F.rhythm = R.rhythm; F.aaMass = R.aaMass; F.aaLam = R.aaLam;
            for (int ch = 0; ch < 2; ++ch) { F.step[ch] = R.f.step[ch]; F.offset[ch] = R.f.offset[ch]; }
            if (R.half >= SOFT_HALF) { F.soft = true; F.mix = R.f.mix; F.blend = R.f.blend; F.P = R.f.P; F.c = R.f.c; }
            else F.through = true;                          // the values sit on the grid itself but only some of them: a crusher with something after it (a filter) that is not read; its grid is put onto the second sound as it is
        }
        return F;
    }
    F.found = true; double bits = 0.0; int nb = 0;
    for (int ch = 0; ch < 2; ++ch) if (ok[ch]) { F.step[ch] = F.grids[ch].step; F.offset[ch] = F.grids[ch].offset; bits += F.grids[ch].bits; ++nb; }
    F.bits = nb ? bits / (double) nb : 24.0; if (hasHold) { F.P = F.hold.P; F.c = F.hold.c; }
    return F;
}
inline Found find (const Stereo& y, double rate = 48000.0, const Stop& stop = {})
{
    const long n = (long) y.size(), len = std::lround (LONG * rate); Found F;
    if (n <= len) F = findPart (y, rate, stop);
    else
    {
        const long a = part (y, rate); F = findPart (y.slice ((size_t) a, (size_t) len), rate, stop); F.n = y.size(); F.from = a;
        if (F.found && F.P > 1.0) { const double c = F.c + (double) a; F.c = c - F.P * std::floor (c / F.P); }      // the hold's pattern counted from the sound's own start
    }
    if (stop && stop()) return Found();
    if (F.found && F.soft) { const Stereo x = softUndo (y, softOf (F)); F.xl.assign (x.l.begin(), x.l.end()); F.xr.assign (x.r.begin(), x.r.end()); }
    return F;
}

// ---------------------------------------------------------------- the faders
// z through the found crusher, its steps sf times as large and its hold hf times as long (a crusher without a hold
// gets one of hf samples).
inline Stereo through (const Stereo& z, const Found& F, double sf = 1.0, double hf = 1.0)
{
    Soft g = softOf (F); for (int ch = 0; ch < 2; ++ch) { g.step[ch] *= sf; g.offset[ch] *= sf; } g.P = std::max (F.P * hf, 1.0);
    if (F.soft) return softForward (z, g);
    Stereo q = z;
    for (int ch = 0; ch < 2; ++ch) { Vec& v = ch ? q.r : q.l; const double st = g.step[ch], of = g.offset[ch]; if (st > 0.0) for (double& x : v) x = of + st * std::nearbyint ((x - of) / st); }
    held (q, g.P, g.c); return q;
}
// The second sound through the same crusher. s: Amount (1 = as found, 2 = what the crusher changes twice over).
inline Stereo onto (const Stereo& z, const Found& F, double s = 1.0, double sf = 1.0, double hf = 1.0)
{
    if (! F.found || s < 1e-12) return z;
    Stereo q = through (z, F, sf, hf); if (std::abs (s - 1.0) < 1e-12) return q;
    for (size_t i = 0; i < z.size(); ++i) { q.l[i] = z.l[i] + s * (q.l[i] - z.l[i]); q.r[i] = z.r[i] + s * (q.r[i] - z.r[i]); }
    return q;
}
// The found sound with the faders moved. With plain sound mixed in (soft) the sound before the crusher is known, so
// everything can be had: Amount 0 = that sound, 1 = as found, 2 = what the crusher changed twice over; other steps or
// another hold: that sound through the crusher so changed. A bare crusher: what the rounding and the hold threw away
// is gone. Left of the middle nothing happens; a finer grid or a shorter hold than the found one cannot be had; coarser
// steps and a longer hold are put on the sound as it is; right of the middle: towards steps twice as large and a hold
// twice as long (which doubles what the crusher did wrong).
inline Stereo own (const Stereo& y, const Found& F, double s = 1.0, double sf = 1.0, double hf = 1.0)
{
    if (! F.found) return y;
    const bool moved = std::abs (sf - 1.0) >= 1e-9 || std::abs (hf - 1.0) >= 1e-9;
    if (F.soft)
    {
        if (F.xl.size() != y.size()) return y;
        if (! moved && std::abs (s - 1.0) < 1e-12) return y;
        Stereo x; x.l.assign (F.xl.begin(), F.xl.end()); x.r.assign (F.xr.begin(), F.xr.end());
        const Stereo q = moved ? through (x, F, sf, hf) : y; Stereo o = x;
        for (size_t i = 0; i < y.size(); ++i) { o.l[i] = x.l[i] + s * (q.l[i] - x.l[i]); o.r[i] = x.r[i] + s * (q.r[i] - x.r[i]); }
        return o;
    }
    sf = std::max (sf, 1.0); hf = std::max (hf, 1.0);
    Stereo q = std::abs (sf - 1.0) < 1e-9 && std::abs (hf - 1.0) < 1e-9 ? y : through (y, F, sf, hf);
    if (s > 1.0 + 1e-12)
    {
        const Stereo q2 = through (q, F, 2.0 * sf, F.P * hf > 1.0 ? 2.0 * hf : 1.0);
        for (size_t i = 0; i < q.size(); ++i) { q.l[i] += (s - 1.0) * (q2.l[i] - q.l[i]); q.r[i] += (s - 1.0) * (q2.r[i] - q.r[i]); }
    }
    return q;
}

// What was found on a sound at one rate, for the same sound at another (the app works at 48 kHz, and a file at another
// rate is read at its own, since its values lie on the grid only there): the same grid of values, a hold as long in
// time. n: the sound's length at that rate; xl, xr: the sound before the crusher at that rate (the caller's conversion
// of F.xl, F.xr; without it the found sound itself cannot be changed).
inline Found atRate (Found F, double rate, size_t n, std::vector<float> xl = {}, std::vector<float> xr = {})
{
    const double r = rate / F.rate; F.n = n; if (std::abs (r - 1.0) < 1e-9) return F;
    F.from = std::lround ((double) F.from * r);
    if (F.P > 1.0 && F.P * r > 1.0) { F.P *= r; F.c *= r; } else { F.P = 1.0; F.c = 0.0; }
    F.xl = std::move (xl); F.xr = std::move (xr); F.rate = rate; return F;
}

} // namespace crush
} // namespace dr
