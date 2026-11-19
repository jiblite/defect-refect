#pragma once
// The stutter engine (STUTTER): a slice of the sound played several times over, in place of what came after it:
//     wet[s0 + j L + i] = wet[s0 + i],   j = 1, 2, ..,  up to the event's end        (one event: start s0, slice L, end)
//   find()    reads the events off one sound: where, how long a slice, up to where it is played again
//   onto()    the found events played on another sound: Amount 0 = none, 1 = as found, 2 = each slice twice as many
//             extra times; Slice: the slices half to twice as long, over the same stretch
//   own()     the found sound with the group's faders moved: right of the middle each slice is played more often (over
//             what follows); left of it fewer times. A stutter cannot be taken out: what the repeats cover is gone. With
//             fewer plays the sound closes up and ends early, which is the nearest thing there is.
// How: the places where the sound is, sample for sample, what it was L samples before, for a stretch at least half as
// long as L. They are looked for with a table of all windows of 64 samples (two windows with the same samples land in
// the same place of the table), then each stretch is walked out to its first and last equal sample (stretches).
// What is not an event although it repeats exactly:
//   - silence, and a sound that stands still (a steady offset): equal whatever was played. Such a stretch counts only
//     beside one of the same slice length that moves, and at an event's two ends such samples say nothing (edges);
//   - a sound that repeats of its own accord, a loop (ownWay);
//   - a steady wave, a held note of a machine: a slice of it "played again" shows nothing where one play ends and the
//     next begins (tone);
//   - the same note struck again with silence between (apart);
//   - a sound that goes on the same way before the slice or after the last play, at another loudness: a note that is
//     itself a short loop fading in or out, a loop through an effect that has to settle (goesOn).
// What it cannot see: a slice that is silent or stands still; where a cut inside silence lies; two events one after the
// other whose slices hold the same samples; a repeat that is not exact (after another effect, after a change of the
// sample rate or of the word length with dither).
// A port of the Python prototype Research/stutter/stutter.py (Tools/stutter_parity.py). Plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "Dsp.h"
#include "Fft.h"

namespace dr
{
namespace stutter
{
using Stop = std::function<bool()>;                     // asked now and then while looking: true = give up (the answer is then not used)
constexpr long WIN = 64;                                // samples of a window in the table
constexpr double TOL = 1.0 / 262144.0;                  // two samples are "the same" if they are this close (2^-18)
constexpr double KEEP = 0.5;                            // a stretch counts if it is at least this share of its slice long (an event: one and a half plays or more)
constexpr double OWN_OUT = 0.10;                        // a slice over 0.3 s is the sound's own way if this share of the sound repeats at that distance outside the stretches

// Lengths in samples at the sound's rate.
struct Sizes { long lmin = 960, lmax = 96000, extra = 960, loop = 14400, fade = 96, gap = 240, near = 4800, least = 12000; };
inline Sizes sizes (double rate)
{
    Sizes z; z.lmin = std::lround (0.020 * rate); z.lmax = std::lround (2.0 * rate); z.extra = std::lround (0.020 * rate); z.loop = std::lround (0.3 * rate);
    z.fade = std::lround (0.002 * rate); z.gap = std::lround (0.005 * rate); z.near = std::lround (0.1 * rate); z.least = std::lround (0.25 * rate); return z;
}

struct Event { long s0 = 0, L = 0, end = 0; };          // the slice s0 .. s0 + L is played again and again up to sample end
struct Stretch { long lo = 0, hi = 0, L = 0; };         // samples lo + L .. hi + L - 1 are what samples lo .. hi - 1 were
struct LeftOut { long s0 = 0, L = 0, end = 0; std::string why; };
struct Found
{
    bool found = false; size_t n = 0; double rate = 48000.0;
    std::vector<Event> events; bool fades = false;      // fades: the plays start and end at nothing (2 ms)
    double share = 0.0;                                 // how much of the sound the repeats cover
    std::vector<long> own;                              // the distances at which the sound repeats of its own accord
    std::vector<Stretch> stretches; std::vector<LeftOut> leftOut;
    // for the picture and the faders' words
    double slice() const { std::vector<long> v; for (auto& e : events) v.push_back (e.L); if (v.empty()) return 0.0; std::sort (v.begin(), v.end()); return (double) v[v.size() / 2] / rate; }      // the slice in the middle, seconds
    double plays() const { std::vector<double> v; for (auto& e : events) v.push_back ((double) (e.end - e.s0) / (double) e.L); if (v.empty()) return 0.0; std::sort (v.begin(), v.end()); return v[v.size() / 2]; }
};

// ---------------------------------------------------------------- small pieces
struct View                                              // the sound and what its samples say
{
    const Stereo& y; long n;
    explicit View (const Stereo& s) : y (s), n ((long) s.size()) {}
    bool quiet (long i) const { return std::abs (y.l[(size_t) i]) <= 16 * TOL && std::abs (y.r[(size_t) i]) <= 16 * TOL; }
    bool same (long i, long j) const { return std::abs (y.l[(size_t) i] - y.l[(size_t) j]) <= TOL && std::abs (y.r[(size_t) i] - y.r[(size_t) j]) <= TOL; }
    bool eq (long i, long L) const { return same (i + L, i); }                 // sample i + L is what sample i was
    // A sample that says nothing by being equal to the one a slice before: silence, and a sound that stands still (the sample is what its neighbour is).
    bool dull (long i) const { return quiet (i) || (i > 0 && same (i - 1, i)) || (i + 1 < n && same (i, i + 1)); }
    double peak (long a, long b) const { double p = 0.0; for (long i = std::max (a, 0L); i < std::min (b, n); ++i) p = std::max ({ p, std::abs (y.l[(size_t) i]), std::abs (y.r[(size_t) i]) }); return p; }
    long live (long a, long b) const { long c = 0; for (long i = a; i < b; ++i) c += ! quiet (i); return c; }
    long moving (long a, long b) const { long c = 0; for (long i = a + 1; i <= b; ++i) c += ! same (i - 1, i); return c; }      // how many of the samples a + 1 .. b differ from the one before
};

// ---------------------------------------------------------------- the table
// The runs of windows of 64 samples that occurred before, each window at the nearest earlier place with the same
// samples (rounded to 2^-16): for every run the distance and the sample in its middle. Windows of one value over and
// over (silence) say nothing and are left out. Two tables that take turns hold the windows of the last two stretches
// of lmax samples: a window further back than that is of no use.
struct Run { long L = 0, mid = 0; };
inline std::vector<Run> runs (const Stereo& y, const Sizes& z, const Stop& stop = {})
{
    const long n = (long) y.size(); std::vector<Run> out; if (n < 2 * WIN) return out;
    const uint64_t A = 0x9E3779B97F4A7C15ULL, K = 0x100000001B3ULL;
    uint64_t top = 1; for (long i = 0; i < WIN - 1; ++i) top *= A;             // A^(WIN - 1)
    auto ql = [&] (long i) { return (int64_t) std::nearbyint (y.l[(size_t) i] * 65536.0); };
    auto qr = [&] (long i) { return (int64_t) std::nearbyint (y.r[(size_t) i] * 65536.0); };
    auto mix = [&] (long i) { return (uint64_t) ql (i) * K + (uint64_t) qr (i); };
    auto changes = [&] (long i) { return ql (i + 1) != ql (i) || qr (i + 1) != qr (i); };       // between samples i and i + 1
    const long block = z.lmax + WIN; size_t cap = 1; while ((long) cap < 4 * block) cap <<= 1;
    struct Gen
    {
        std::vector<uint64_t> key; std::vector<int32_t> val; uint64_t mask = 0;
        void init (size_t c) { key.assign (c, 0); val.assign (c, -1); mask = c - 1; }
        void clear() { std::fill (val.begin(), val.end(), -1); }
        size_t slot (uint64_t h) const { size_t s = (size_t) ((h ^ (h >> 29)) & mask); while (val[s] >= 0 && key[s] != h) s = (s + 1) & mask; return s; }
        long get (uint64_t h) const { return val[slot (h)]; }
        void put (uint64_t h, long t) { const size_t s = slot (h); key[s] = h; val[s] = (int32_t) t; }
    };
    Gen g[2]; g[0].init (cap); g[1].init (cap); int cur = 0; long genStart = 0;
    uint64_t h = 0; for (long i = 0; i < WIN; ++i) h = h * A + mix (i);
    long ch = 0; for (long i = 0; i + 1 < WIN; ++i) ch += changes (i);          // changes inside the window
    std::vector<long> ts; long curL = 0, lastT = -1;
    auto close = [&] { if (! ts.empty()) out.push_back ({ curL, ts[ts.size() / 2] }); ts.clear(); };
    const long nw = n - WIN + 1;
    for (long t = 0; t < nw; ++t)
    {
        if ((t & 0xFFFF) == 0 && stop && stop()) return {};
        if (t - genStart >= block) { cur ^= 1; g[cur].clear(); genStart = t; }
        if (ch > 0)
        {
            long before = g[cur].get (h); if (before < 0) before = g[cur ^ 1].get (h);
            if (before >= 0)
            {
                const long lag = t - before;
                if (lag >= z.lmin && lag <= z.lmax)
                {
                    if (ts.empty() || t - lastT > WIN || lag != curL) { close(); curL = lag; }
                    ts.push_back (t); lastT = t;
                }
            }
            g[cur].put (h, t);
        }
        if (t + 1 < nw)
        {
            h = (h - mix (t) * top) * A + mix (t + WIN);
            ch += (long) changes (t + WIN - 1) - (long) changes (t);
        }
    }
    close(); return out;
}

// ---------------------------------------------------------------- the stretches
// The stretches where the sound is what it was L samples before. Silence equals silence, which proves nothing: a
// stretch is cut where a whole slice of it is silent, and it starts and ends no further out than one slice from its
// first and last live sample. A sound that stands still proves nothing by being equal either, and a stretch with
// little sound in it not much: such a stretch counts only if its slice length is that of a stretch that moves.
inline std::vector<Stretch> stretches (const View& v, const std::vector<Run>& R, const Sizes& z)
{
    const long n = v.n; std::vector<Stretch> out, weak; std::vector<std::pair<long, long>> seen;       // (L, hi)
    for (const Run& r : R)
    {
        const long L = r.L, i0 = r.mid - L;
        long was = -1; for (auto& s : seen) if (s.first == L) was = s.second;
        if (i0 < was || i0 < 0 || i0 + L >= n) continue;
        if (! v.eq (i0, L)) continue;
        long lo = i0; while (lo > 0 && v.eq (lo - 1, L)) --lo;
        long hi = i0; while (hi < n - L && v.eq (hi, L)) ++hi;
        bool had = false; for (auto& s : seen) if (s.first == L) { s.second = hi; had = true; }
        if (! had) seen.push_back ({ L, hi });
        // the live ones among them, in groups with no gap over one slice
        std::vector<long> first, last; long prevLive = -1;
        for (long i = lo; i < hi; ++i)
            if (! v.quiet (i + L))
            {
                if (prevLive < 0 || i - prevLive > L) { first.push_back (i); last.push_back (i); }
                else last.back() = i;
                prevLive = i;
            }
        for (size_t k = 0; k < first.size(); ++k)
        {
            const long fl = first[k], ll = last[k];
            const long a0 = std::max (k == 0 ? lo : last[k - 1] + 1, fl - L + 1), b0 = std::min (k + 1 == first.size() ? hi : first[k + 1], ll + L);
            const long live = v.live (fl + L, ll + L + 1), moving = v.moving (fl + L, std::min (ll + L, n - 1));
            const double need = std::max (0.2 * (double) std::min (L, b0 - a0), 4.0 * WIN);
            if (((double) (b0 - a0) >= KEEP * (double) L || (b0 + L == n && (double) (b0 - a0) >= 0.25 * (double) L)) && b0 - a0 >= z.extra && live >= 4 * WIN)      // cut off by the end of the sound: a quarter of a slice is enough
                ((double) std::min (live, moving) >= need ? out : weak).push_back ({ a0, b0, L });
        }
    }
    std::set<long> have; for (auto& s : out) have.insert (s.L);
    for (auto& s : weak) if (have.count (s.L)) out.push_back (s);
    std::sort (out.begin(), out.end(), [] (const Stretch& a, const Stretch& b) { return a.lo != b.lo ? a.lo < b.lo : a.hi != b.hi ? a.hi < b.hi : a.L < b.L; });
    return out;
}

// The distances at which the sound repeats of its own accord (a loop), which are not events: (1) one stretch that runs
// from the start of the sound to its end, whatever its length; and for distances over 0.3 s: (2) two stretches of which
// the second goes on with what the first repeats (a loop with a break in it); (3) the sound repeating at that distance
// in many places outside the stretches (a loop with many breaks).
inline std::set<long> ownWay (const View& v, const std::vector<Stretch>& S, const Sizes& z)
{
    const long n = v.n; std::set<long> res; long f0 = -1, f1 = -1, nl = 0;
    for (long i = 0; i < n; ++i) if (! v.quiet (i)) { if (f0 < 0) f0 = i; f1 = i; ++nl; }
    if (nl == 0) return res;
    const double span = (double) std::max (f1 - f0, 1L);
    std::set<long> Ls; for (auto& s : S) Ls.insert (s.L);
    for (long L : Ls)
    {
        std::vector<Stretch> P; for (auto& s : S) if (s.L == L) P.push_back (s);
        bool whole = false; for (auto& s : P) whole = whole || ((double) s.lo <= (double) f0 + 0.05 * span && (double) (s.hi + L) >= (double) f1 - 0.10 * span);
        if (whole) { res.insert (L); continue; }
        if (L <= z.loop) continue;
        bool hit = false;
        for (size_t i = 0; i < P.size() && ! hit; ++i)
            for (size_t j = i + 1; j < P.size() && ! hit; ++j)
            {
                const long a2 = P[j].lo, D = L * ((a2 - P[i].lo) / L); if (D <= 0) continue;
                const long e = std::min (a2 + L, n); long m = 0, same = 0;
                for (long t = a2; t < e; ++t) if (! v.quiet (t)) { ++m; same += v.same (t, t - D); }
                if (m > 0 && (double) same >= 0.9 * (double) m) hit = true;
            }
        if (hit) { res.insert (L); continue; }
        long c = 0; size_t p = 0;                                                  // equal at that distance outside the stretches (P is in order of lo; stretches of one length do not lie in each other)
        for (long i = 0; i < n - L; ++i)
        {
            bool inside = false; for (p = 0; p < P.size(); ++p) if (i >= P[p].lo && i < P[p].hi) { inside = true; break; }
            if (! inside && ! v.quiet (i + L) && v.eq (i, L)) ++c;
        }
        if ((double) c >= OWN_OUT * (double) nl) res.insert (L);
    }
    return res;
}

// Whether what looks like a slice played again is a steady wave (a held note of a machine, a test tone): the slice is
// itself the same thing over and over at a much shorter distance, and nothing happens where one play ends and the next
// begins. A slice of such a note that was really played again shows a jump (or a fade) there. The shorter distance is
// one at which the wave comes back after having gone away (a slow wave is like itself a few samples on: that is not a
// period).
inline bool tone (const View& v, long s0, long L, long end)
{
    const long n = v.n, half = L / 2; size_t nf = 1; while ((long) nf < 2 * L) nf <<= 1;
    Vec ac (nf, 0.0), c2 ((size_t) L + 1, 0.0);
    double ml = 0.0, mr = 0.0; for (long i = 0; i < L; ++i) { ml += v.y.l[(size_t) (s0 + i)]; mr += v.y.r[(size_t) (s0 + i)]; }
    ml /= (double) L; mr /= (double) L;                                            // (without its mean: a steady offset is not a wave)
    for (int ch = 0; ch < 2; ++ch)
    {
        Vec re (nf, 0.0), im (nf, 0.0); const Vec& src = ch ? v.y.r : v.y.l; const double m = ch ? mr : ml;
        for (long i = 0; i < L; ++i) re[(size_t) i] = src[(size_t) (s0 + i)] - m;
        fft (re, im); for (size_t i = 0; i < nf; ++i) ac[i] += re[i] * re[i] + im[i] * im[i];
    }
    { Vec im (nf, 0.0); fft (ac, im, true); }
    for (long i = 0; i < L; ++i) { const double a = v.y.l[(size_t) (s0 + i)] - ml, b = v.y.r[(size_t) (s0 + i)] - mr; c2[(size_t) i + 1] = c2[(size_t) i] + a * a + b * b; }
    Vec r ((size_t) half + 1, 0.0);
    for (long p = 8; p <= half; ++p) r[(size_t) p] = ac[(size_t) p] / std::sqrt (std::max (c2[(size_t) (L - p)] * (c2[(size_t) L] - c2[(size_t) p]), 1e-60));
    long low = -1; for (long p = 8; p <= half; ++p) if (r[(size_t) p] < 0.5) { low = p; break; }
    if (low < 0) return false;
    long ps = low; for (long p = low; p <= half; ++p) if (r[(size_t) p] > r[(size_t) ps]) ps = p;
    if (r[(size_t) ps] <= 0.0 || r[(size_t) ps] < 0.999) return false;
    const long W = std::max (ps, 128L); long seams = 0, stand = 0;
    auto d2 = [&] (long a, long b) { double s = 0.0; for (long i = a; i < b; ++i) { const double dl = v.y.l[(size_t) i] - v.y.l[(size_t) (i - ps)], dr_ = v.y.r[(size_t) i] - v.y.r[(size_t) (i - ps)]; s += dl * dl + dr_ * dr_; } return s; };
    for (long q = s0 + L; q < end; q += L)
    {
        if (q + W > std::min (end, n) || q - W < s0) break;                         // (the sound after the event's end is not a seam)
        const double ds = d2 (q - W + ps, q + W) / (double) (2 * W - ps);           // around the seam
        const double di = d2 (q - L + ps, q - W) / (double) std::max (L - W - ps, 1L);     // inside the play before it
        ++seams; stand += ds > 16.0 * di + 2.0 * TOL * TOL;
    }
    return seams > 0 && 4 * stand <= seams;
}

// Whether the sound before the slice or after the last play is the slice once more at another loudness (a held note
// of an instrument that is itself a short loop, fading in or out; a loop through an effect that has to settle). A slice
// that was played again is a cut: the sound before and after it is another. The 100 ms before and after are held
// against the samples one slice length on (or back), with a loudness that may change evenly; alike if what is left over
// is under a hundredth of the power. Each without its mean: a steady offset is not a sound that goes on.
inline std::string goesOn (const View& v, long s0, long L, long end, const Sizes& z)
{
    const long n = v.n; const double floor = (16 * TOL) * (16 * TOL);
    auto alike = [&] (long a0, long b0, long w)
    {
        if (w < z.extra) return false;
        double mal = 0, mar = 0, mbl = 0, mbr = 0;
        for (long i = 0; i < w; ++i) { mal += v.y.l[(size_t) (a0 + i)]; mar += v.y.r[(size_t) (a0 + i)]; mbl += v.y.l[(size_t) (b0 + i)]; mbr += v.y.r[(size_t) (b0 + i)]; }
        mal /= (double) w; mar /= (double) w; mbl /= (double) w; mbr /= (double) w;
        double ea = 0, eb = 0, bt = 0, tt = 0, ab = 0, abt = 0;
        for (long i = 0; i < w; ++i)
        {
            const double t = w > 1 ? -1.0 + 2.0 * (double) i / (double) (w - 1) : -1.0;
            const double al = v.y.l[(size_t) (a0 + i)] - mal, ar = v.y.r[(size_t) (a0 + i)] - mar, bl = v.y.l[(size_t) (b0 + i)] - mbl, br = v.y.r[(size_t) (b0 + i)] - mbr;
            ea += al * al + ar * ar; const double bb = bl * bl + br * br, xy = al * bl + ar * br;
            eb += bb; bt += bb * t; tt += bb * t * t; ab += xy; abt += xy * t;
        }
        if (ea <= floor * (double) (2 * w) || eb <= floor * (double) (2 * w)) return false;
        const double det = eb * tt - bt * bt; double g0 = ab / eb, g1 = 0.0;
        if (det > 1e-12 * eb * tt) { g0 = (ab * tt - abt * bt) / det; g1 = (abt * eb - ab * bt) / det; }
        double res = 0.0;
        for (long i = 0; i < w; ++i)
        {
            const double t = w > 1 ? -1.0 + 2.0 * (double) i / (double) (w - 1) : -1.0, g = g0 + g1 * t;
            const double rl = (v.y.l[(size_t) (a0 + i)] - mal) - g * (v.y.l[(size_t) (b0 + i)] - mbl), rr = (v.y.r[(size_t) (a0 + i)] - mar) - g * (v.y.r[(size_t) (b0 + i)] - mbr);
            res += rl * rl + rr * rr;
        }
        return res < 0.01 * ea && g0 > 0.0;
    };
    long w = std::min ({ L, z.near, n - end });
    if (w > 0 && alike (end, end - L, w)) return "the sound goes on the same way after it";
    w = std::min ({ L, z.near, s0 });
    if (w > 0 && alike (s0 - w, s0 - w + L, w)) return "the sound went the same way before it";
    return "";
}

// Whether every play stands by itself, with silence before and after it (5 ms each, 60 dB under the slice's loudest
// sample): then it is the same note or stroke played again, as a player or a machine does, and not a slice cut out of
// a sound and repeated.
inline bool apart (const View& v, long s0, long L, long end, const Sizes& z)
{
    const double thr = std::max (16 * TOL, 1e-3 * v.peak (s0, s0 + L)); long seams = 0;
    for (long q = s0 + L; q < end; q += L) { ++seams; if (v.peak (q - z.gap, q + z.gap) > thr) return false; }
    return seams > 0;
}

// How far the start of a stretch may lie after lo, and its end before hi: the samples there that say nothing on both
// sides (they are equal whatever was played).
inline void edges (const View& v, const Stretch& s, long& a, long& b)
{
    a = s.lo; while (a < s.hi - 1 && a + s.L < v.n && v.dull (a) && v.dull (a + s.L)) ++a;
    b = s.hi; while (b - 1 > a && v.dull (b - 1) && v.dull (b - 1 + s.L)) --b;
}
// Where an event that starts at s0 ends: no later than the stretch (hi + L), no earlier than its last sample that says
// something (hi2 + L). Whole plays where the samples allow it, and as few as they allow.
inline long endsAt (long s0, long L, long hi, long hi2)
{
    long m = (long) std::ceil ((double) (hi2 + L - s0) / (double) L);
    if (m >= 2) return s0 + m * L <= hi + L ? s0 + m * L : hi + L;
    m = (long) std::floor ((double) (hi + L - s0) / (double) L + 0.5);              // nothing in it says anything after the first play: the nearest whole number
    return hi + L - L / 2 <= s0 + m * L && s0 + m * L <= hi + L ? s0 + m * L : hi + L;
}

// ---------------------------------------------------------------- find
inline Found find (const Stereo& y, double rate = 48000.0, const Stop& stop = {})
{
    Found F; F.n = y.size(); F.rate = rate; const Sizes z = sizes (rate); const View v (y); const long n = v.n;
    if (n < z.least) return F;
    if (v.peak (0, n) <= 16 * TOL) return F;
    const std::vector<Run> R = runs (y, z, stop); if (stop && stop()) return F;
    const std::vector<Stretch> S = stretches (v, R, z); F.stretches = S;
    const std::set<long> O = ownWay (v, S, z); F.own.assign (O.begin(), O.end());
    std::vector<long> A (S.size()), B (S.size()); for (size_t i = 0; i < S.size(); ++i) edges (v, S[i], A[i], B[i]);
    long prevEnd = -1;
    for (size_t i = 0; i < S.size(); ++i)
    {
        const long lo = S[i].lo, hi = S[i].hi, L = S[i].L;
        if (O.count (L)) continue;
        if (stop && stop()) return Found();
        long s0 = (lo <= prevEnd && prevEnd <= A[i]) ? prevEnd : lo;                 // a slice that follows straight on another event starts where that one ends
        if (lo < prevEnd && (double) (hi + L - prevEnd) >= (1.0 + KEEP) * (double) L) s0 = prevEnd;       // ... also when the sound before it happens to go the same way for a few samples (the ends of two fades; a held note of a machine): what is left of the stretch after the event before is an event of its own
        long end = endsAt (s0, L, hi, B[i]); bool any = false; long nxt = 0;
        for (size_t j = 0; j < S.size(); ++j)                                        // another event that starts a whole number of plays on: this one ends there (the sound may go on being equal for a while, where both slices begin alike)
            if (lo < S[j].lo && S[j].lo <= hi + L)
            {
                const long k = std::max ((long) std::ceil ((double) (S[j].lo - s0) / (double) L), 1L), w = s0 + k * L;
                if (S[j].lo <= w && w <= A[j] && (S[j].L == L || hi < w)) { nxt = any ? std::min (nxt, w) : w; any = true; }      // (one of another slice length: only within the last play)
            }
        if (any) end = nxt;
        const long whole = s0 + L * (long) std::ceil ((double) (end - s0) / (double) L - 0.02);      // the last play is taken as whole when what is missing of it is silence (in it and in the slice)
        if (end < whole && whole <= n && whole - end <= L / 2 && v.peak (end, whole) <= 16 * TOL && v.peak (end - L, whole - L) <= 16 * TOL) end = whole;
        end = std::min (end, n);
        if (! ((double) (end - s0) >= (1.0 + KEEP) * (double) L || end >= n)) continue;
        if (s0 < prevEnd) { F.leftOut.push_back ({ s0, L, end, "inside the event before" }); continue; }      // two events cannot lie in each other: the one that starts first is the event, the other an echo of it
        if (tone (v, s0, L, end)) { F.leftOut.push_back ({ s0, L, end, "a steady wave" }); continue; }
        const std::string why = apart (v, s0, L, end, z) ? std::string ("every play stands by itself in silence") : goesOn (v, s0, L, end, z);
        if (! why.empty()) { F.leftOut.push_back ({ s0, L, end, why }); continue; }
        F.events.push_back ({ s0, L, end }); prevEnd = end;
    }
    if (F.events.empty()) return F;
    double cover = 0.0; long plays = 0, at0 = 0;
    for (const Event& e : F.events)
    {
        cover += (double) (e.end - e.s0 - e.L);
        const long m = (long) std::ceil ((double) (e.end - e.s0) / (double) e.L);
        for (long j = 1; j < m; ++j)
        {
            const long t = e.s0 + j * e.L; if (t + 8 >= n) continue;
            if (v.peak (t, t + z.fade) <= 64 * TOL) continue;                        // the plays that begin in silence anyway say nothing about a fade
            ++plays; at0 += v.quiet (t);                                             // a play that starts at nothing and comes up at once: a fade
        }
    }
    F.share = cover / (double) n; F.fades = plays > 0 && (double) at0 >= 0.9 * (double) plays; F.found = true;
    return F;
}

// ---------------------------------------------------------------- the faders
// The events with Amount at s: the stretch that the extra plays of each slice cover, times s (to whole plays), nothing
// beyond the sound's end. Slice: the slice length times that (the stretch stays).
inline std::vector<Event> scaled (const std::vector<Event>& ev, double s, long n, double slice = 1.0, long least = 96)
{
    std::vector<Event> out;
    for (const Event& e : ev)
    {
        long end = std::abs (s - 1.0) < 1e-12 ? e.end : e.s0 + e.L * (long) std::floor (1.0 + ((double) (e.end - e.s0) / (double) e.L - 1.0) * s + 0.5);
        end = std::min (end, n);
        const long L = std::abs (slice - 1.0) < 1e-12 ? e.L : std::max ((long) std::lround ((double) e.L * slice), least);
        if (end > e.s0 + L && e.s0 + L <= n) out.push_back ({ e.s0, L, end });
    }
    return out;
}
// z with every event played: the slice from s0 on, again and again over what follows it, up to the event's end. With
// fades each play comes up from nothing and goes back to it (2 ms).
inline Stereo play (const Stereo& z, const std::vector<Event>& ev, bool fades, long fade = 96)
{
    Stereo o = z; const long n = (long) z.size(); long done = 0;
    for (const Event& e : ev)
    {
        if (e.s0 < done || e.s0 + e.L > n) continue;                                 // an event that starts inside the one before (after a change of the fader): left out
        const long end = std::min (e.end, n); const bool f = fades && e.L >= 2 * fade;
        for (long t = e.s0; t < end; ++t)
        {
            const long i = (t - e.s0) % e.L; double g = 1.0;
            if (f) { if (i < fade) g = (double) i / (double) fade; else if (i >= e.L - fade) g = (double) (e.L - 1 - i) / (double) fade; }
            o.l[(size_t) t] = g * z.l[(size_t) (e.s0 + i)]; o.r[(size_t) t] = g * z.r[(size_t) (e.s0 + i)];
        }
        done = end;
    }
    return o;
}
// The found events on another sound. at: the other sound's rate against the rate of the sound they were found on.
inline Stereo onto (const Stereo& z, const Found& F, double s = 1.0, double slice = 1.0)
{
    if (! F.found || s < 1e-12) return z;
    return play (z, scaled (F.events, s, (long) z.size(), slice, sizes (F.rate).fade), F.fades, sizes (F.rate).fade);
}
// The found sound with Amount at s. Right of the middle: each slice is played more often (over what follows). Left of
// it: fewer times, and since what the repeats covered is gone, the sound closes up and ends early (silence at the end).
// Slice under 1: the first part of each slice is played instead of all of it (a longer slice than the found one is not
// in the sound: what came after it is gone).
inline Stereo own (const Stereo& y, const Found& F, double s = 1.0, double slice = 1.0)
{
    if (! F.found) return y;
    const long n = (long) y.size(), fade = sizes (F.rate).fade; slice = std::min (slice, 1.0);
    if (std::abs (s - 1.0) < 1e-12 && std::abs (slice - 1.0) < 1e-12) return y;
    const std::vector<Event> ev = scaled (F.events, s, n);
    Stereo o = y; std::vector<Event> now;                                            // the events as they lie in the result
    if (s > 1.0)
    {
        for (const Event& e : ev)
        {
            long e0 = e.end; for (const Event& w : F.events) if (w.s0 == e.s0 && w.L == e.L) e0 = w.end;
            for (long t = e0; t < e.end; ++t) { const long i = (t - e.s0) % e.L; o.l[(size_t) t] = y.l[(size_t) (e.s0 + i)]; o.r[(size_t) t] = y.r[(size_t) (e.s0 + i)]; }
            now.push_back (e);
        }
    }
    else if (std::abs (s - 1.0) >= 1e-12)
    {
        std::vector<char> keep ((size_t) n, 1);
        for (const Event& w : F.events)
        {
            long to = w.s0 + w.L; for (const Event& e : ev) if (e.s0 == w.s0 && e.L == w.L) to = e.end;
            for (long t = to; t < std::min (w.end, n); ++t) keep[(size_t) t] = 0;
        }
        std::vector<long> at ((size_t) n + 1, 0); long k = 0;                        // where each sample went
        for (long t = 0; t < n; ++t) { at[(size_t) t] = k; if (keep[(size_t) t]) { o.l[(size_t) k] = y.l[(size_t) t]; o.r[(size_t) k] = y.r[(size_t) t]; ++k; } }
        at[(size_t) n] = k; for (long t = k; t < n; ++t) { o.l[(size_t) t] = 0.0; o.r[(size_t) t] = 0.0; }
        for (const Event& e : ev) now.push_back ({ at[(size_t) e.s0], e.L, at[(size_t) e.s0] + (e.end - e.s0) });
    }
    else now = ev;
    if (std::abs (slice - 1.0) >= 1e-12)
    {
        std::vector<Event> fine; for (const Event& e : now) { const long L = std::max ((long) std::lround ((double) e.L * slice), fade); if (L < e.L) fine.push_back ({ e.s0, L, e.end }); }
        o = play (o, fine, F.fades, fade);
    }
    return o;
}

} // namespace stutter
} // namespace dr
