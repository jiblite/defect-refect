#pragma once
// The reader of pitch movement: what the PITCH fader of the MOVEMENT group works with.
//   track()   how a sound's pitch moves, in cents every 10 ms; where nothing can be followed the curve stands still
// Pitch movement is read off a sound without naming its pitch: how far the lines of the spectrum (the partials) shift from
// one moment to another, on a logarithmic frequency axis, measured between moments 10 ms to 1.3 s apart and put together
// into one curve by least squares. The notes of a tune are not followed (a step between two notes is no shift of lines
// that can be matched); bends, glides, vibrato, revs and Doppler are.
// Two looks: a short one (85 ms windows) that follows fast movement, and a long one (341 ms) that tells apart partials
// 12 Hz from each other, which an engine at low revs needs.
// Rounds: the sound is flattened with the whole curve read so far (tape speed) and read again, until nothing more is
// left to read; then once more, strictly. Where a round follows nothing, the curve the sound is flattened with keeps
// the movement it had. What the strict reading still finds on top says where the curve is not sure: there it is not
// kept.
// "Followed": the same lines are found again over a span of time, directly. A sound that is lost for a moment (up to
// 0.6 s) is looked for on the other side of the loss, from where it lies if it stood still to where it lies if it
// went on moving. Found, the curve goes through the loss in a straight line, if that is proven: the lines have to
// stand stiller with the glide than with the same step taken at once (a sound's lines fit those of its own harmonics
// in part, and are found where they are not). Not found but most alike near there, the sound goes on and the curve
// stands still across the loss; most alike far away, or found and not proven, another sound may start. A pause (the
// level 30 dB under what sounds before and after) separates two sounds for good.
// What the reader cannot read it leaves alone. The curve that is kept stands still wherever the strict reading
// follows nothing, and the next stretch goes on where the last one left it; it begins afresh only at a pause. A step
// anywhere else has to be proven: where no sound is sustained (zaps, chirps) a short event keeps its place only if
// the lines after the step are alike those before it once the sound is flattened with the curve, and beside a
// sustained sound a short event is not kept at all (see compose). No reading follows a step that an earlier round
// put into the curve the sound is flattened with. The curve's fast part (above about 3 Hz) is kept only where the
// lines stand stiller with it than without (see proven): two sounds close in pitch beat, and a curve that follows
// the flutter of their lines would add one. Movement that is not read stays in the sound.
// What it cannot read: noise (no lines), partials closer than about 25 Hz that move fast, falling or rising sounds
// that follow each other without a pause (read in part), short sounds beside a sustained one, two things moving
// differently at the same time (it follows one of them, or neither).
// A port of the Python prototype Research/movement/flow.py (Tools/move_parity.py compares the two). 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <future>
#include <string>
#include <vector>

#include "Room.h"      // dr::Vec (Dsp.h), room::Transform

namespace dr
{
namespace flow
{
using Stop = std::function<bool()>;                     // asked now and then: true = give up (the answer is then not used)
using Log = std::function<void (const std::string&)>;   // what the reader does, in words (for the tests)
constexpr int STEP = 480;                               // the curve: a value every 10 ms
constexpr int LOOKS[2][2] = { { 4096, 480 }, { 16384, 960 } };      // window, hop
constexpr double FMIN = 35.0, FMAX = 8000.0, RES = 5.0; // the log axis: 5 cents per step
constexpr int NB = 1882;                                // its steps: ceil (1200 log2 (FMAX / FMIN) / RES) + 1
constexpr double SPAN = 1.3;                            // s: the longest span measured directly
constexpr double RANGE = 70.0;                          // cents searched either way between neighbouring frames
constexpr double NEAR = 20.0;                           // cents searched around what the curve so far says, for frames further apart
constexpr double SIGMA = 1.2;                           // steps: the width a line is drawn with
constexpr double OVER = 10.0;                           // dB over the noise around it that makes a peak a line
constexpr double FLOOR_HZ = 375.0;                      // the noise around a bin: from the lower quarter of the bins in a block this wide
constexpr double AGREE = 4.0;                           // cents: a measurement agrees with the curve when it is this close (plus a tenth of its size)
constexpr double PK = 0.5;                              // how alike two frames' lines must be for a measurement to count at all
constexpr double TRUST = 0.75;                          // share of a moment's measurements that must agree for the moment to be held
constexpr double EDGE = 0.05;                           // the long look takes over when its measurements agree by this much more than the short look's
constexpr double TIE = 0.35;                            // how alike two moments' lines must be to count as the same lines (at the place the curve says)
constexpr double COVER = 0.5;                           // share of the pairs across a moment that must be tied for the moment to count as followed over that span
constexpr double LONG = 0.585;                          // s: lines followed over this span or more: the chain is trusted as far as it goes
constexpr double CUT = 0.25;                            // a place where pairs reach less than this share of what they reach on both sides of it is looked at again
constexpr double REACH = 2.0;                           // where lines are followed over short spans only, the chain counts against its average over this many times the span
constexpr double DONE_RMS = 1.0, DONE_MOST = 8.0;       // cents left to read at which the flattening has settled
constexpr int PASSES = 5;                               // rounds at most
constexpr double GAP = 0.25;                            // s: two stretches up to this far apart are joined by a straight line
constexpr double RAMP = 0.1;                            // s: over which the curve steps to an event that begins afresh, and back from one
constexpr double JOIN = 0.6;                            // s: a sound lost for at most this long is looked for again on the other side of the loss
constexpr double WIDE = 240.0;                          // cents: how far from what the movement on both sides suggests it is looked for at most
constexpr double FAR = 1300.0;                          // cents: how far it is looked to see whether what is most alike lies near or far
constexpr double WEAK = 0.25;                           // how alike the two sides of a loss must be at least for the sound to count as going on
constexpr double CRUMB = 0.15;                          // s: a stretch shorter than this next to a short loss hangs onto its neighbour
constexpr double SURE_CENTS = 3.0, SURE_SHARE = 0.5;    // where reading the flattened sound again still finds more than this (and more than this share of the curve's own movement), the curve is not kept
constexpr double PAUSE = 1e-3;                          // a pause: the level this far under what sounds before and after it
constexpr long LEAST = 2 * 4096;                        // a sound shorter than this (171 ms): nothing is read
constexpr double JUMP = 1e-4;                           // a cut: the level rises or falls by this much (40 dB) within two hundredths of a second; the ends of the file are cuts
constexpr long NEARCUT = 1632;                          // samples (3.4 frames): this near a cut the short window has a tenth of its weight or more at the cut
constexpr long REACHCUT = 8192;                         // samples: from this far outside a sound the long window still reaches it
constexpr double SHIFT[2] = { 2.0, 0.5 };               // Hz: how far a cut through its window moves a line on average, for the short look and the long one
constexpr double STAND = 7.0;                           // cents: beside a cut a look's reading counts only where that moves the sound's lines by less than this on average
constexpr double ALIKE = 0.75;                          // a step of the curve between two stretches is proven when, the sound flattened with the curve, the lines after the step are this alike those before it
constexpr double ROUND = 150.0;                         // cents: how far from where the step puts them the lines after it are looked for (they are then put where they are found)
constexpr double STEADY = 1.1;                          // a sound found again across a loss: the curve goes through the loss only if the lines then stand this many times stiller than with a curve that takes the same step at once
constexpr double AROUND = 0.6;                          // s: how much of the sound on both sides of the loss is flattened for that
constexpr double SLOW = 6.0;                            // moments: the curve smoothed this much is its slow part (below about 3 Hz); the rest is its fast part
constexpr double FASTKEEP_NONE = 1.0, FASTKEEP_ALL = 1.05;      // the fast part is dropped where the lines stand no stiller with it than with the slow part alone, and kept whole where they stand this many times stiller
constexpr int FINE = 2;                                 // the spectrum the stillness of lines is measured in: this many values per bin of the long window (the measure then does not depend on where between two bins a line lies)
constexpr double LINES = 3.0;                           // a block of the spectrum holds lines when its energy is gathered this many times closer than evenly spread energy is (noise: 2)
constexpr double DUST = 1e-6;                           // of a frame's mean energy per value: added to every value, so that what is left of nothing does not count as a line
constexpr int GHOP = 3840;                              // samples: how far apart the frames lie in which the stillness of lines is measured (80 ms; the long window is 341 ms long)
constexpr long TOGETHER = 9600000;                      // samples: up to this length (200 s) the two sounds a proof flattens are worked on at the same time
constexpr int GBLOCK = 256;                             // values per half block of that spectrum: FINE max (8, round (FLOOR_HZ LOOKS[1][0] / SR)); a block is two of them, fading in and out
constexpr int GFIRST = 24;                              // its first value: ceil (FMIN FINE LOOKS[1][0] / SR)
constexpr int GBLOCKS = 20;                             // blocks: floor ((FMAX FINE LOOKS[1][0] / SR - GFIRST) / GBLOCK) - 1

// ---------------------------------------------------------------- small things, as numpy and scipy have them
inline double sq (double v) { return v * v; }
inline long nearest (double v) { return (long) std::nearbyint (v); }                                            // numpy.round: halves go to the even number
inline double between (const Vec& v, double t)                                                                  // numpy.interp on 0, 1, 2 ...
{
    const long n = (long) v.size(); if (n == 0) return 0.0; if (t <= 0.0) return v[0]; if (t >= (double) (n - 1)) return v[(size_t) n - 1];
    const long i = (long) t; return (v[(size_t) i + 1] - v[(size_t) i]) * (t - (double) i) + v[(size_t) i];
}
inline double interpAt (double x, const Vec& xp, const Vec& fp)                                                 // numpy.interp for rising xp
{
    const size_t n = xp.size(); if (n == 0) return 0.0; if (x <= xp[0]) return fp[0]; if (x >= xp[n - 1]) return fp[n - 1];
    const size_t i = (size_t) (std::upper_bound (xp.begin(), xp.end(), x) - xp.begin()) - 1;                    // xp[i] <= x < xp[i + 1]
    return (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]) * (x - xp[i]) + fp[i];
}
inline Vec gradient (const Vec& v)                                                                              // numpy.gradient
{
    const long n = (long) v.size(); Vec g ((size_t) n, 0.0); if (n < 2) return g;
    g[0] = v[1] - v[0]; g[(size_t) n - 1] = v[(size_t) n - 1] - v[(size_t) n - 2];
    for (long i = 1; i < n - 1; ++i) g[(size_t) i] = (v[(size_t) i + 1] - v[(size_t) i - 1]) / 2.0;
    return g;
}
inline Vec largestWithin (const Vec& v, int half)                                                               // scipy.ndimage.maximum_filter1d, size 2 half + 1, of values that are not negative
{
    const long n = (long) v.size(); Vec o ((size_t) n);
    for (long i = 0; i < n; ++i) { double m = v[(size_t) i]; for (long j = std::max (0L, i - half); j <= std::min (n - 1, i + half); ++j) m = std::max (m, v[(size_t) j]); o[(size_t) i] = m; }
    return o;
}
inline Vec gaussN (const Vec& v, double sigma)                                                                  // scipy.ndimage.gaussian_filter1d (v, sigma, mode='nearest')
{
    const long r = (long) (4.0 * sigma + 0.5); Vec w ((size_t) (2 * r + 1)); double s = 0;
    for (long k = -r; k <= r; ++k) { w[(size_t) (k + r)] = std::exp (-0.5 * sq ((double) k / sigma)); s += w[(size_t) (k + r)]; }
    for (auto& a : w) a /= s;
    const long n = (long) v.size(); Vec o ((size_t) n);
    for (long i = 0; i < n; ++i) { double a = 0; for (long k = -r; k <= r; ++k) a += w[(size_t) (k + r)] * v[(size_t) std::clamp (i + k, 0L, n - 1)]; o[(size_t) i] = a; }
    return o;
}
// The slow part of a curve: smoothed over SLOW moments, the curve carried on past both ends as it arrives there
// (mirrored through its end points), so that a steady glide has no fast part at the ends of the file either.
inline Vec slowPart (const Vec& v)
{
    const long T = (long) v.size(), R = (long) (4.0 * SLOW + 0.5); if (T == 0) return {};
    Vec p ((size_t) (T + 2 * R));
    for (long i = 1; i <= R; ++i) { p[(size_t) (R - i)] = 2.0 * v[0] - v[(size_t) std::min (i, T - 1)]; p[(size_t) (R + T - 1 + i)] = 2.0 * v[(size_t) T - 1] - v[(size_t) std::max (T - 1 - i, 0L)]; }
    for (long i = 0; i < T; ++i) p[(size_t) (R + i)] = v[(size_t) i];
    const Vec s = gaussN (p, SLOW); return Vec (s.begin() + R, s.begin() + R + T);
}
inline double dot (const double* a, const double* b, long n)
{
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0; long i = 0;
    for (; i + 4 <= n; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// ---------------------------------------------------------------- the tape-speed reader
constexpr int RH = 16, RPH = 512;                       // a Hann-windowed sinc: zero crossings each side, table steps per crossing
inline const Vec& readerTable()
{
    static const Vec tab = []
    {
        Vec t ((size_t) (2 * RH * RPH + 1));
        for (int i = 0; i <= 2 * RH * RPH; ++i) { const double u = (double) (i - RH * RPH) / (double) RPH, s = u == 0.0 ? 1.0 : std::sin (PI * u) / (PI * u); t[(size_t) i] = s * (0.5 + 0.5 * std::cos (PI * u / (double) RH)); }
        return t;
    }();
    return tab;
}
// The weights with which the samples k0 .. k1 count for the place p read at the speed `step` (samples per sample).
// Where the places move faster than one sample per sample (pitch up) the reader's cutoff comes down with them, so
// nothing folds back. Calls use (k, weight) for every sample that counts.
template <typename Use> inline void readerTaps (double p, double step, long n, Use&& use)
{
    const Vec& tab = readerTable(); const double f = std::min (1.0, 1.0 / std::max (step, 1e-6)), reach = (double) RH / f;
    const long k0 = std::max (0L, (long) std::floor (p - reach)), k1 = std::min (n - 1, (long) std::ceil (p + reach));
    for (long k = k0; k <= k1; ++k)
    {
        const double u = ((double) k - p) * f; if (std::abs (u) >= (double) RH) continue;
        const double ui = u * (double) RPH + (double) (RH * RPH); const long i0 = std::min ((long) ui, (long) (2 * RH * RPH - 1)); const double fr = ui - (double) i0;
        use (k, (tab[(size_t) i0] * (1.0 - fr) + tab[(size_t) i0 + 1] * fr) * f);
    }
}
inline double stepAt (const Vec& pos, long j)           // numpy.gradient of the places
{
    const long m = (long) pos.size(); return m < 2 ? 1.0 : j == 0 ? pos[1] - pos[0] : j == m - 1 ? pos[(size_t) m - 1] - pos[(size_t) m - 2] : (pos[(size_t) j + 1] - pos[(size_t) j - 1]) / 2.0;
}

// ---------------------------------------------------------------- cuts
// Where a sound starts or stops abruptly, the ends of the file included. A window that lies across such a cut holds a
// tone that stops, and the frequency its phase says it has is then off by up to 0.4 bins (5 Hz with the short window,
// 1 Hz with the long one) whatever the tone's own frequency: nothing for a line at 4 kHz, a semitone and more for one
// at 60 Hz. Read like that, the curve of a low sound runs away at every cut. Recordings cut in two, against the same
// recordings read whole: off by 50 cents on average over the 40 ms next to the cut, up to 180, where the sound itself
// hardly moves; with silence around the piece the curve runs on into the silence. A sound whose lines lie high reads
// right up to the cut. What was tried and thrown out: only lines high enough to stand the cut count there (the reader
// then loses whole stretches of an engine at low revs).
// So the reading beside a cut counts only where the sound's lines lie high enough for the look that reads there (see
// lowness). Where they do not, the curve stands still beside the cut (it takes the reading of the nearest frame the
// cut leaves alone), which loses the movement of those 40 ms where it was read right and never adds one. On the quiet
// side of a cut nothing is followed.
struct Cut { long pos = 0; int dir = 0; };              // dir +1: the sound starts with sample pos; -1: it stopped with the sample before pos
// power (i): the sound's power at sample i of n (its square; of a sound with two sides the mean of theirs).
template <typename Power> inline std::vector<Cut> cutsOf (long n, Power&& power)
{
    const int B = (int) ((n + STEP - 1) / STEP); std::vector<Cut> all; if (B < 1) return all;
    Vec E ((size_t) B + 6, 0.0);                        // the level of every hundredth of a second: block b at b + 3, silence outside the file
    for (int b = 0; b < B; ++b) { const long a = (long) b * STEP, e = std::min (n, a + STEP); double s = 0; for (long i = a; i < e; ++i) s += power (i); E[(size_t) b + 3] = s / (double) (e - a); }
    auto lev = [&] (int b) { return E[(size_t) (b + 3)]; };
    auto quietAfter = [&] (int b) { return lev (b) < JUMP * std::max (lev (b + 1), lev (b + 2)); };         // against what comes after it
    auto quietBefore = [&] (int b) { return lev (b) < JUMP * std::max (lev (b - 1), lev (b - 2)); };        // against what came before it
    for (int b = 0; b <= B; ++b)
    {
        if (b < B && quietAfter (b - 1) && ! quietAfter (b))            // a sound starts in block b: at its first sample that is not quiet
        {
            const double ref = JUMP * std::max (lev (b), lev (b + 1)); long at = (long) b * STEP;
            for (long i = at; i < std::min (n, (long) (b + 2) * STEP); ++i) if (power (i) >= ref) { at = i; break; }
            all.push_back ({ at, 1 });
        }
        if (b >= 1 && quietBefore (b) && ! quietBefore (b - 1))         // a sound stops in block b - 1: after its last sample that is not quiet
        {
            const double ref = JUMP * std::max (lev (b - 1), lev (b - 2)); long at = std::min (n, (long) b * STEP);
            for (long i = at - 1; i >= std::max (0L, (long) (b - 2) * STEP); --i) if (power (i) >= ref) { at = i + 1; break; }
            all.push_back ({ at, -1 });
        }
    }
    return all;
}
inline std::vector<Cut> cuts (const Vec& x) { return cutsOf ((long) x.size(), [&x] (long i) { return sq (x[(size_t) i]); }); }
// Per frame: the frame whose reading counts for it (itself where no cut is near), -1 = not followed (on the quiet side
// of a cut, or beside a cut in a sound too short to have a frame that is left alone).
inline std::vector<int> besideCuts (const std::vector<Cut>& all, int T)
{
    std::vector<int> src ((size_t) T); for (int t = 0; t < T; ++t) src[(size_t) t] = t;
    std::vector<char> spoiled ((size_t) T, 0), dead ((size_t) T, 0);
    struct Zone { int a, b, hold; }; std::vector<Zone> zones;           // the frames a .. b beside a cut, and the frame they hold on to
    auto down = [] (long v) { return (int) (v >= 0 ? v / STEP : -((-v + STEP - 1) / STEP)); };          // the last frame at or before sample v
    for (size_t k = 0; k < all.size(); ++k)
    {
        const long p = all[k].pos;
        if (all[k].dir > 0)
        {
            const int a = down (p - 1) + 1, h = down (p + NEARCUT - 1) + 1; zones.push_back ({ a, h - 1, h });       // the frames from p to just under p + NEARCUT
            const long back = std::max (k > 0 ? all[k - 1].pos : p - REACHCUT, p - REACHCUT);
            for (int t = std::min (a - 1, T - 1); t >= 0 && (long) t * STEP >= back; --t) dead[(size_t) t] = 1;
        }
        else
        {
            const int b = down (p - 1), h = down (p - NEARCUT); zones.push_back ({ h + 1, b, h });                    // the frames from just over p - NEARCUT to just under p
            const long on = std::min (k + 1 < all.size() ? all[k + 1].pos - 1 : p + REACHCUT, p + REACHCUT);
            for (int t = std::max (b + 1, 0); t < T && (long) t * STEP <= on; ++t) dead[(size_t) t] = 1;
        }
    }
    for (const Zone& z : zones) for (int t = std::max (0, z.a); t <= std::min (T - 1, z.b); ++t) spoiled[(size_t) t] = 1;
    for (const Zone& z : zones)
    {
        const bool ok = z.hold >= 0 && z.hold < T && ! spoiled[(size_t) z.hold] && ! dead[(size_t) z.hold];
        for (int t = std::max (0, z.a); t <= std::min (T - 1, z.b); ++t) { if (ok) src[(size_t) t] = z.hold; else dead[(size_t) t] = 1; }
    }
    for (int t = 0; t < T; ++t) if (dead[(size_t) t]) src[(size_t) t] = -1;
    return src;
}

// ---------------------------------------------------------------- lines
// Every frame's lines on the log axis. A line is a peak of the spectrum that stands OVER dB over the noise around it;
// it is drawn at the frequency its phase says it has (not at its bin), the more over the noise the stronger.
// Kept as the cells that are not empty (row t: start[t] .. start[t + 1]).
struct Lines
{
    int T = 0; std::vector<int> start, bin, cnt; Vec val, e, nrm;       // cnt: lines per frame; e: the frame's mean power; nrm: the row's length
};
inline Lines lines (const Vec& x, int n, int hop, const Stop& stop = {})
{
    const long len = (long) x.size(); const int T = 1 + (int) (len / hop), K = n / 2 + 1;
    Lines L; L.T = T; L.start.assign ((size_t) T + 1, 0); L.cnt.assign ((size_t) T, 0); L.e.assign ((size_t) T, 0.0); L.nrm.assign ((size_t) T, 0.0);
    Vec h ((size_t) n), dh ((size_t) n); for (int i = 0; i < n; ++i) { h[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * PI * (double) i / (double) n); dh[(size_t) i] = (PI / (double) n) * std::sin (2.0 * PI * (double) i / (double) n); }
    const int blk = std::max (8, (int) nearest (FLOOR_HZ * (double) n / SR)), nblk = K / blk;
    std::vector<int> jb ((size_t) K); Vec ub ((size_t) K), w ((size_t) K);
    for (int k = 0; k < K; ++k)
    {
        int s = 0; while (s < nblk && ((double) s + 0.5) * (double) blk < (double) k) ++s;          // block centres under k
        const int j = std::clamp (s - 1, 0, nblk - 2); jb[(size_t) k] = j;
        ub[(size_t) k] = std::clamp (((double) k - ((double) j + 0.5) * (double) blk) / (double) blk, 0.0, 1.0); w[(size_t) k] = 2.0 * PI * (double) k / (double) n;
    }
    const room::Transform tr (n); Vec re ((size_t) n), im ((size_t) n), row ((size_t) NB, 0.0), Q ((size_t) nblk), part ((size_t) blk);
    constexpr int BATCH = 256; Vec P ((size_t) BATCH * (size_t) K), num ((size_t) BATCH * (size_t) K);
    const double near = 1.5 * 2.0 * PI / (double) n;
    for (int a = 0; a < T; a += BATCH)
    {
        if (stop && stop()) { L.T = -1; return L; }
        const int cnt = std::min (BATCH, T - a); double top = 0;
        for (int q = 0; q < cnt; ++q)
        {
            const long at = (long) (a + q) * hop - n / 2; double e = 0;
            for (int i = 0; i < n; ++i) { const long j = at + i; const double v = (j >= 0 && j < len) ? x[(size_t) j] : 0.0; re[(size_t) i] = v * h[(size_t) i]; im[(size_t) i] = v * dh[(size_t) i]; e += v * v; }
            L.e[(size_t) (a + q)] = e / (double) n; tr.run (re, im);                // the frame under the window as the real part, under the window's slope as the imaginary part
            double* p = P.data() + (size_t) q * (size_t) K; double* u = num.data() + (size_t) q * (size_t) K;
            for (int k = 0; k < K; ++k)
            {
                const size_t c = (size_t) k, m = (size_t) ((n - k) & (n - 1));
                const double xr = 0.5 * (re[c] + re[m]), xi = 0.5 * (im[c] - im[m]), dr_ = 0.5 * (im[c] + im[m]), di = 0.5 * (re[m] - re[c]);
                p[k] = xr * xr + xi * xi; u[k] = di * xr - dr_ * xi; top = std::max (top, p[k]);
            }
        }
        const double least = 1e-10 * top + 1e-30;
        for (int q = 0; q < cnt; ++q)
        {
            const int t = a + q; const double* p = P.data() + (size_t) q * (size_t) K; const double* u = num.data() + (size_t) q * (size_t) K;
            for (int b = 0; b < nblk; ++b)              // the noise: the lower quartile of the block (0.288 of the mean of noise)
            {
                std::copy (p + (size_t) b * (size_t) blk, p + (size_t) (b + 1) * (size_t) blk, part.begin());
                const double pos = 0.25 * (double) (blk - 1); const int i = (int) std::floor (pos); const double fr = pos - (double) i;
                std::nth_element (part.begin(), part.begin() + i, part.end()); const double lo = part[(size_t) i], hi = *std::min_element (part.begin() + i + 1, part.end());
                Q[(size_t) b] = (fr >= 0.5 ? hi - (hi - lo) * (1.0 - fr) : lo + (hi - lo) * fr) / 0.288;
            }
            int found = 0, first = NB, last = -1;
            for (int k = 1; k < K - 1; ++k)
            {
                if (! (p[k] > p[k - 1] && p[k] >= p[k + 1])) continue;
                const size_t c = (size_t) k; const double floor = std::max (Q[(size_t) jb[c]] * (1.0 - ub[c]) + Q[(size_t) jb[c] + 1] * ub[c], least);
                const double over = 10.0 * std::log10 (std::max (p[k], 1e-30) / floor); if (! (over > OVER)) continue;
                const double inst = w[c] - u[k] / std::max (p[k], 1e-30), f = inst * SR / (2.0 * PI);
                if (! (std::abs (inst - w[c]) < near && f > FMIN && f < FMAX)) continue;
                ++found; const double pos = 1200.0 * std::log2 (f / FMIN) / RES, wt = std::clamp ((over - OVER + 4.0) / 24.0, 0.0, 1.0); const long i0 = nearest (pos);
                for (long o = -4; o <= 4; ++o) { const long b = i0 + o; if (b >= 0 && b < NB) { row[(size_t) b] += wt * std::exp (-0.5 * sq (((double) b - pos) / SIGMA)); first = std::min (first, (int) b); last = std::max (last, (int) b); } }
            }
            L.cnt[(size_t) t] = found; double s2 = 0;
            for (int b = first; b <= last; ++b) if (row[(size_t) b] != 0.0) { L.bin.push_back (b); L.val.push_back (row[(size_t) b]); s2 += sq (row[(size_t) b]); row[(size_t) b] = 0.0; }
            L.nrm[(size_t) t] = std::sqrt (s2); L.start[(size_t) t + 1] = (int) L.bin.size();
        }
    }
    return L;
}

// How low the lines of the frame around sample c lie: the mean of 1 / frequency over them (per Hz), each line as
// strong as it is drawn. Times 1731 and a shift in Hz: how many cents that shift moves the lines on average.
// No lines: a large number.
inline double lowness (const Vec& x, long c)
{
    constexpr int lead = 10; Vec seg ((size_t) (2 * lead * STEP), 0.0); const long len = (long) x.size();
    for (long i = 0; i < (long) seg.size(); ++i) { const long j = c - (long) lead * STEP + i; if (j >= 0 && j < len) seg[(size_t) i] = x[(size_t) j]; }
    const Lines L = lines (seg, LOOKS[0][0], STEP); if (L.T <= lead) return 1e9;
    double num = 0, den = 0;
    for (int k = L.start[(size_t) lead]; k < L.start[(size_t) lead + 1]; ++k) { const double v = sq (L.val[(size_t) k]); num += v / (FMIN * std::pow (2.0, (double) L.bin[(size_t) k] * RES / 1200.0)); den += v; }
    return den > 0.0 ? num / den : 1e9;
}

// How alike two frames' lines are, the second's moved by a number of steps.
struct Peak { double shift = 0, pk = 0; bool inside = false; };
struct Matcher
{
    const Lines& A; Vec dense; int loaded = -1;
    explicit Matcher (const Lines& a) : A (a), dense ((size_t) NB, 0.0) {}
    void load (int j)
    {
        if (j == loaded) return;
        if (loaded >= 0) for (int q = A.start[(size_t) loaded]; q < A.start[(size_t) loaded + 1]; ++q) dense[(size_t) A.bin[(size_t) q]] = 0.0;
        for (int q = A.start[(size_t) j]; q < A.start[(size_t) j + 1]; ++q) dense[(size_t) A.bin[(size_t) q]] = A.val[(size_t) q];
        loaded = j;
    }
    // C[s + S], s = -S .. S: frame i's lines against frame j's moved by o + s steps (1 = the same picture)
    void alike (int i, int j, long o, int S, double* C)
    {
        load (j); std::fill (C, C + 2 * S + 1, 0.0);
        for (int q = A.start[(size_t) i]; q < A.start[(size_t) i + 1]; ++q)
        {
            const long k = (long) A.bin[(size_t) q] + o, lo = std::max ((long) -S, -k), hi = std::min ((long) S, (long) NB - 1 - k); const double v = A.val[(size_t) q];
            for (long s = lo; s <= hi; ++s) C[s + S] += v * dense[(size_t) (k + s)];
        }
        const double nn = A.nrm[(size_t) i] * A.nrm[(size_t) j] + 1e-12; for (int s = 0; s <= 2 * S; ++s) C[s] /= nn;
    }
};
// The best place among C[0 .. 2 S] (found between the steps from the three values around it), as a shift in cents.
inline Peak peakOf (const double* C, int S, long o)
{
    int j = 0; for (int s = 1; s <= 2 * S; ++s) if (C[s] > C[j]) j = s;
    Peak k; k.pk = C[j]; k.inside = j > 0 && j < 2 * S; const int jj = std::clamp (j, 1, 2 * S - 1);
    const double l0 = std::log (std::max (C[jj - 1], 1e-9)), l1 = std::log (std::max (C[jj], 1e-9)), l2 = std::log (std::max (C[jj + 1], 1e-9)), den = l0 - 2.0 * l1 + l2;
    const double off = den < -1e-9 ? 0.5 * (l0 - l2) / std::min (den, -1e-9) : 0.0;
    k.shift = ((double) (o + jj - S) + std::clamp (off, -1.0, 1.0)) * RES; return k;
}

// ---------------------------------------------------------------- the curve that fits the measurements
// p (cents) with the smallest sum of w (p[j] - p[i] - s)^2 + lam1 sum (p[t + 1] - p[t])^2 + lam0 sum p^2; measurements
// far off the curve count less (three rounds).
inline Vec solve (int T, const std::vector<int>& I, const std::vector<int>& J, const Vec& S, const Vec& W0, double lam0 = 1e-6, double lam1 = 1e-4, double huber = 6.0, int rounds = 3)
{
    Vec p ((size_t) T, 0.0); if (I.empty()) return p;
    int bw = 0; for (size_t k = 0; k < I.size(); ++k) bw = std::max (bw, J[k] - I[k]);
    const size_t B = (size_t) bw + 1; Vec W = W0, M ((size_t) T * B), z ((size_t) T);         // the matrix's lower half, row i holding the columns i - bw .. i
    for (int it = 0; it < rounds; ++it)
    {
        std::fill (M.begin(), M.end(), 0.0); std::fill (z.begin(), z.end(), 0.0);
        for (size_t k = 0; k < I.size(); ++k)
        {
            const size_t i = (size_t) I[k], j = (size_t) J[k]; const double w = W[k];
            M[i * B + (size_t) bw] += w; M[j * B + (size_t) bw] += w; M[j * B + (size_t) bw - (j - i)] -= w; z[j] += w * S[k]; z[i] -= w * S[k];
        }
        for (int t = 0; t < T; ++t)
        {
            double& d = M[(size_t) t * B + (size_t) bw]; d += lam0; if (t < T - 1) d += lam1; if (t > 0) { d += lam1; M[(size_t) t * B + (size_t) bw - 1] -= lam1; }
        }
        for (int i = 0; i < T; ++i)                     // M = L L': row by row
        {
            double* ri = M.data() + (size_t) i * B; const int j0 = std::max (0, i - bw);
            for (int j = j0; j <= i; ++j)
            {
                const double* rj = M.data() + (size_t) j * B;      // columns j - bw .. j; the shared columns j0 .. j - 1
                const double s = ri[j - (i - bw)] - dot (ri + (j0 - (i - bw)), rj + (j0 - (j - bw)), j - j0);
                ri[j - (i - bw)] = j == i ? std::sqrt (std::max (s, 1e-300)) : s / rj[bw];
            }
            z[(size_t) i] = (z[(size_t) i] - dot (ri + (j0 - (i - bw)), z.data() + j0, i - j0)) / ri[bw];
        }
        for (int i = T - 1; i >= 0; --i)
        {
            const double* ri = M.data() + (size_t) i * B; const double v = z[(size_t) i] / ri[bw]; p[(size_t) i] = v;
            for (int m = std::max (0, i - bw); m < i; ++m) z[(size_t) m] -= ri[m - (i - bw)] * v;
        }
        for (size_t k = 0; k < I.size(); ++k) { const double r = std::abs (p[(size_t) J[k]] - p[(size_t) I[k]] - S[k]); W[k] = W0[k] * std::min (1.0, huber / std::max (r, 1e-9)); }
    }
    return p;
}
inline bool agrees (const Vec& p, int i, int j, double s, double k = 1.0) { return std::abs (p[(size_t) j] - p[(size_t) i] - s) < k * (AGREE + 0.1 * std::abs (s)); }
// Per frame: the share (by weight) of its measurements that agree with the curve, evened over 100 ms, times how many there are (up to 1).
inline Vec trustOf (int T, const Vec& p, const std::vector<int>& I, const std::vector<int>& J, const Vec& S, const Vec& W, int hop, double k)
{
    Vec good ((size_t) T, 0.0), all ((size_t) T, 0.0);
    for (size_t q = 0; q < I.size(); ++q) { const double ok = agrees (p, I[q], J[q], S[q], k) ? W[q] : 0.0; good[(size_t) I[q]] += ok; good[(size_t) J[q]] += ok; all[(size_t) I[q]] += W[q]; all[(size_t) J[q]] += W[q]; }
    const int size = std::max (1, (int) nearest (0.1 * SR / (double) hop)) | 1; const Vec g = uniformFilter (good, size, 1), a = uniformFilter (all, size, 1);
    Vec tr ((size_t) T); for (int t = 0; t < T; ++t) tr[(size_t) t] = g[(size_t) t] / (a[(size_t) t] + 1e-9) * std::min (1.0, g[(size_t) t] / 0.5);
    return tr;
}

// ---------------------------------------------------------------- one look
struct LookOut { Vec p, tr, span, reach; std::vector<char> mended, goes, other; bool stopped = false; };
// speed: how fast the curve the sound was flattened with moves around each frame (cents per second); brk: the frames
// of a pause; cv: that curve itself. loose: how many times more loosely a measurement may agree with the curve.
inline LookOut look (const Vec& x, int n, int hop, double loose, double trust, const Vec& speed, const std::vector<char>& brk, const Vec& cv, const Stop& stop, const Log& log)
{
    LookOut O; const Lines A = lines (x, n, hop, stop); if (A.T < 0) { O.stopped = true; return O; }
    const int T = A.T; O.tr.assign ((size_t) T, 0.0); O.span.assign ((size_t) T, 0.0); O.reach.assign ((size_t) T, 0.0); O.mended.assign ((size_t) T, 0); O.goes.assign ((size_t) T, 0); O.other.assign ((size_t) T, 0);
    const double emax = *std::max_element (A.e.begin(), A.e.end()); std::vector<char> live ((size_t) T); for (int t = 0; t < T; ++t) live[(size_t) t] = A.cnt[(size_t) t] >= 1 && A.e[(size_t) t] > emax * 1e-6;
    std::vector<int> cb ((size_t) T + 1, 0); for (int t = 0; t < T; ++t) cb[(size_t) t + 1] = cb[(size_t) t] + (brk[(size_t) t] ? 1 : 0);
    auto free = [&] (int t, int d) { return cb[(size_t) (t + d) + 1] - cb[(size_t) t] == 0; };          // no break between two sounds from frame t to t + d
    std::vector<int> I, J; Vec S, W; Vec& p = O.p; p.assign ((size_t) T, 0.0);
    std::vector<int> tls; std::vector<std::vector<int>> tieI; std::vector<Vec> tieS;                    // the spans the same lines are looked for over (in frames), and the pairs found: first frame, shift
    Matcher M (A); Vec C ((size_t) (2 * nearest (FAR / RES) + 1));
    const int Sn = (int) nearest (NEAR / RES);
    for (int d = 1; d < T && (double) d * (double) hop <= SPAN * SR; d *= 2)
    {
        if (stop && stop()) { O.stopped = true; return O; }
        const bool first = (double) d * (double) hop <= 0.021 * SR, tie = (double) d * (double) hop >= 0.9 * (double) n; const int Sx = first ? (int) nearest (RANGE / RES) : Sn;
        if (tie) { tls.push_back (d); tieI.emplace_back(); tieS.emplace_back(); }
        for (int t = 0; t + d < T; ++t)
        {
            if (! (live[(size_t) t] && live[(size_t) (t + d)] && free (t, d))) continue;
            const long o = first ? 0 : nearest ((p[(size_t) (t + d)] - p[(size_t) t]) / RES); M.alike (t, t + d, o, Sx, C.data()); const Peak k = peakOf (C.data(), Sx, o);
            if (! k.inside) continue;
            if (k.pk > PK) { I.push_back (t); J.push_back (t + d); S.push_back (k.shift); W.push_back (sq ((k.pk - PK) / (1.0 - PK)) / std::sqrt ((double) d)); }
            if (tie && k.pk > TIE) { tieI.back().push_back (t); tieS.back().push_back (k.shift); }
        }
        p = solve (T, I, J, S, W);
    }
    Vec& tr = O.tr; tr = trustOf (T, p, I, J, S, W, hop, loose);
    for (int it = 0; it < 2; ++it)                      // the measurements of moments that are not held go, and the curve is fitted again
    {
        size_t kept = 0;
        for (size_t q = 0; q < I.size(); ++q) if (tr[(size_t) I[q]] >= trust && tr[(size_t) J[q]] >= trust) { I[kept] = I[q]; J[kept] = J[q]; S[kept] = S[q]; W[kept] = W[q]; ++kept; }
        if (kept == I.size()) break;
        I.resize (kept); J.resize (kept); S.resize (kept); W.resize (kept); p = solve (T, I, J, S, W);
        const Vec t2 = trustOf (T, p, I, J, S, W, hop, loose); for (int t = 0; t < T; ++t) tr[(size_t) t] = std::min (tr[(size_t) t], t2[(size_t) t]);
        if (stop && stop()) { O.stopped = true; return O; }
    }
    bool anyTie = false; for (auto& v : tieI) anyTie = anyTie || ! v.empty();
    if (! anyTie || tls.empty()) return O;

    const int d0 = tls[0]; Vec mlag ((size_t) T, 0.0); Vec& span = O.span; Vec& reach = O.reach; std::vector<char>& mended = O.mended; std::vector<char>& goes = O.goes; std::vector<char>& other = O.other;
    auto covering = [&] (const std::vector<int>& i, int dd)        // per frame: how many of the pairs (i, i + dd) lie across it
    {
        std::vector<int> c ((size_t) T + 1, 0); for (int a : i) { ++c[(size_t) a]; --c[(size_t) (a + dd) + 1]; }
        for (int t = 1; t <= T; ++t) c[(size_t) t] += c[(size_t) t - 1];
        c.resize ((size_t) T); return c;
    };
    std::vector<int> cand0; for (int t = 0; t + d0 < T; ++t) if (live[(size_t) t] && live[(size_t) (t + d0)]) cand0.push_back (t);
    const std::vector<int> nc0 = covering (cand0, d0);
    // Which moments are followed over the shortest span (base), the longest span over each moment, how far to both sides.
    auto spans = [&] (std::vector<char>& base)
    {
        std::vector<std::vector<int>> by (tls.size());
        for (size_t li = 0; li < tls.size(); ++li) for (size_t q = 0; q < tieI[li].size(); ++q) { const int i = tieI[li][q]; if (agrees (p, i, i + tls[li], tieS[li][q], 2.0)) by[li].push_back (i); }
        const std::vector<int> nt0 = covering (by[0], d0); base.assign ((size_t) T, 0);
        for (int t = 0; t < T; ++t) base[(size_t) t] = nc0[(size_t) t] > 0 && (double) nt0[(size_t) t] >= COVER * (double) nc0[(size_t) t];
        std::fill (span.begin(), span.end(), 0.0); std::fill (reach.begin(), reach.end(), 0.0);
        for (size_t li = 0; li < tls.size(); ++li)
        {
            const int dd = tls[li]; std::vector<int> i;
            if (li == 0) { i = by[0]; for (int t = 0; t < T; ++t) if (base[(size_t) t]) span[(size_t) t] = (double) dd * (double) hop / SR; }
            else
            {
                std::vector<int> cand; for (int a = 0; a + dd < T; ++a) if (base[(size_t) a] && base[(size_t) (a + dd)]) cand.push_back (a);       // longer spans: of the pairs of moments followed over the shortest
                for (int a : by[li]) if (base[(size_t) a] && base[(size_t) (a + dd)]) i.push_back (a);
                const std::vector<int> nc = covering (cand, dd), nt = covering (i, dd);
                for (int t = 0; t < T; ++t) if (nc[(size_t) t] > 0 && (double) nt[(size_t) t] >= COVER * (double) nc[(size_t) t] && (base[(size_t) t] || mended[(size_t) t])) span[(size_t) t] = (double) dd * (double) hop / SR;
            }
            if (! i.empty())                            // how far the pairs reach to both sides of each moment
            {
                std::vector<char> tie ((size_t) T, 0); for (int a : i) tie[(size_t) a] = 1;
                std::vector<int> dist ((size_t) T, 1 << 30); int lastTie = -(1 << 30);
                for (int t = 0; t < T; ++t) { if (tie[(size_t) t]) lastTie = t; if (lastTie > -(1 << 29)) dist[(size_t) t] = t - lastTie; }
                lastTie = 1 << 30; for (int t = T - 1; t >= 0; --t) { if (tie[(size_t) t]) lastTie = t; if (lastTie < (1 << 29)) dist[(size_t) t] = std::min (dist[(size_t) t], lastTie - t); }
                const int h = dd / 2;
                for (int t = h; t < T; ++t) reach[(size_t) t] = std::max (reach[(size_t) t], std::max (0.0, (double) dd / 2.0 - (double) dist[(size_t) (t - h)]) * (double) hop / SR);
            }
        }
        for (int t = 0; t < T; ++t) { span[(size_t) t] = std::max (span[(size_t) t], mlag[(size_t) t]); reach[(size_t) t] = std::max (reach[(size_t) t], 0.5 * mlag[(size_t) t]); }       // a loss the sound was found again across: followed, over the span it was found across
    };
    auto again = [&]                                    // the pairs once more, around the mended curve
    {
        for (size_t li = 0; li < tls.size(); ++li)
        {
            const int d = tls[li]; tieI[li].clear(); tieS[li].clear();
            for (int t = 0; t + d < T; ++t)
            {
                if (! (live[(size_t) t] && live[(size_t) (t + d)] && free (t, d))) continue;
                const long o = nearest ((p[(size_t) (t + d)] - p[(size_t) t]) / RES); M.alike (t, t + d, o, Sn, C.data()); const Peak k = peakOf (C.data(), Sn, o);
                if (k.inside && k.pk > TIE) { tieI[li].push_back (t); tieS[li].push_back (k.shift); }
            }
        }
    };
    // A sound lost or followed badly for a moment (the frames weak): is it the same sound on the other side? Measured
    // directly, from moments before to moments after: looked for from where it lies if the sound stood still across
    // the loss to where it lies if it went on moving as it does on both sides (a movement that stops inside the loss
    // ends in between), and as far beyond both as that movement is fast.
    auto mend = [&] (const std::vector<char>& weak, const std::vector<char>& base, bool lost)
    {
        bool any = false; const int Sf = (int) nearest (FAR / RES); Vec Cm ((size_t) (2 * Sf + 1)), br; char line[256];
        for (int at = 0; at < T; )
        {
            if (! weak[(size_t) at]) { ++at; continue; }
            const int a = at; int b = a; while (b < T && weak[(size_t) b]) ++b;
            at = b; const int L = b - a;
            if (a < 3 || b > T - 3 || (double) L * (double) hop > JOIN * SR || ! base[(size_t) a - 1] || ! base[(size_t) b]) continue;
            int ls = a - 1; while (ls > 0 && base[(size_t) ls - 1] && ! weak[(size_t) ls - 1]) --ls;
            int re = b + 1; while (re < T && base[(size_t) re] && ! weak[(size_t) re]) ++re;
            const int K = std::max (2, (int) nearest (0.2 * SR / (double) hop)), la = std::max (ls, a - 1 - K), rb = std::min (re, b + 1 + K);
            if (a - la < 3 || rb - b < 3 || cb[(size_t) rb] - cb[(size_t) la] > 0) continue;
            double emin = A.e[(size_t) a], ml = 0, mr = 0; for (int t = a; t < b; ++t) emin = std::min (emin, A.e[(size_t) t]);
            for (int t = la; t < a; ++t) ml += A.e[(size_t) t]; for (int t = b; t < rb; ++t) mr += A.e[(size_t) t];
            ml /= (double) (a - la); mr /= (double) (rb - b);
            if (emin < 1e-3 * std::min (ml, mr)) continue;                                              // a pause is no loss
            const double sl = (p[(size_t) a - 1] - p[(size_t) la]) / (double) (a - 1 - la), sr = (p[(size_t) rb - 1] - p[(size_t) b]) / (double) (rb - 1 - b);
            double vl = 0, vr = 0; for (int t = la; t < a - 1; ++t) vl += sq (p[(size_t) t + 1] - p[(size_t) t]); for (int t = b; t < rb - 1; ++t) vr += sq (p[(size_t) t + 1] - p[(size_t) t]);
            double V = std::max (std::sqrt (vl / (double) (a - 1 - la)), std::sqrt (vr / (double) (rb - 1 - b)));
            { double sp = 0; for (int t = std::max (0, la); t < rb; ++t) sp = std::max (sp, speed[(size_t) t]); V = std::max (V, sp * (double) hop / SR); }
            const double Dp = (double) (L + 1) * 0.5 * (sl + sr), Hm = std::min (WIDE, 2.0 * NEAR + (double) (L + 1) * V);
            // the sound was flattened: where the whole curve's movement on both sides points, less what the flattening took
            const double Dm = Dp + (double) (L + 1) * 0.5 * ((cv[(size_t) a - 1] - cv[(size_t) la]) / (double) (a - 1 - la) + (cv[(size_t) rb - 1] - cv[(size_t) b]) / (double) (rb - 1 - b)) - (cv[(size_t) b] - cv[(size_t) a - 1]);
            const double Ds = -(cv[(size_t) b] - cv[(size_t) a - 1]);                                     // and where it lies if the sound stood still across the loss
            const double lo = std::min (Dm, Ds) - Hm, hi = std::max (Dm, Ds) + Hm, D0 = 0.5 * (lo + hi), H = std::min (0.5 * (hi - lo), 0.75 * FAR);
            bool done = false, seen = false; double seenNear = 0, seenFar = 0; const int Sh = (int) nearest (H / RES);
            for (int d : tls)
            {
                const int i0 = std::max (ls, b + 1 - d), i1 = std::min (a - 2, re - 1 - d), cnt = i1 - i0 + 1; if (cnt < 3) continue;
                br.clear(); double pks = 0; std::fill (Cm.begin(), Cm.end(), 0.0);
                for (int i = i0; i <= i1; ++i)
                {
                    const double pre = p[(size_t) a - 1] - p[(size_t) i], post = p[(size_t) (i + d)] - p[(size_t) b]; const long o = nearest ((pre + D0 + post) / RES);
                    M.alike (i, i + d, o, Sf, C.data()); for (int s = 0; s <= 2 * Sf; ++s) Cm[(size_t) s] += C[(size_t) s];
                    const Peak k = peakOf (C.data() + (Sf - Sh), Sh, o);                                // the best place near where the movement points
                    if (k.inside && k.pk > PK) { br.push_back (k.shift - pre - post); pks += k.pk; }
                }
                double hereNear = -1e300, hereFar = -1e300;                                             // how alike the two sides are at best, near where the sound is looked for and far from there
                for (int s = 0; s <= 2 * Sf; ++s) { const double v = Cm[(size_t) s] / (double) cnt; if (std::abs (s - Sf) <= Sh) hereNear = std::max (hereNear, v); else hereFar = std::max (hereFar, v); }
                if (! seen) { seen = true; seenNear = hereNear; seenFar = hereFar; }
                const double g = (double) br.size(); if (g < std::max (3.0, 0.5 * (double) cnt) || hereNear < hereFar) continue;      // (what is more alike elsewhere is not this sound found again: a sound's lines fit those of its own harmonics in part)
                double Mv = median (br); double sum = 0; int nok = 0; for (double v : br) if (std::abs (v - Mv) < 2.0 * AGREE + 0.1 * std::abs (Mv)) { sum += v; ++nok; }
                if ((double) nok < std::max (3.0, 0.6 * g)) continue;
                Mv = sum / (double) nok; const double old = p[(size_t) b] - p[(size_t) a - 1]; done = true;
                if (! lost && std::abs (Mv - old) < 2.0 * AGREE) break;                                 // as the curve has it already
                for (int q = 0; q < L; ++q) p[(size_t) (a + q)] = p[(size_t) a - 1] + Mv * ((double) q + 1.0) / ((double) L + 1.0);
                for (int t = b; t < T; ++t) p[(size_t) t] += Mv - old;
                for (int t = a; t < b; ++t) { mended[(size_t) t] = 1; mlag[(size_t) t] = (double) d * (double) hop / SR; }
                any = true;
                if (log) { std::snprintf (line, sizeof line, "   %s %.2f to %.2f s, over %.2f s: %.0f cents instead of %.0f (searched %.0f +- %.0f, of which %.0f from what is left, alike %.2f)", lost ? "lost" : "weak", a * (double) hop / SR, b * (double) hop / SR, d * (double) hop / SR, Mv, old, D0, H, Dp, pks / g); log (line); }
                break;
            }
            // not found again for sure: does the sound go on all the same (what is most alike lies near where the movement points), or does another start?
            if (! done && lost && seen)
            {
                const bool on = seenNear >= WEAK && seenNear >= seenFar - 0.1; for (int t = a; t < b; ++t) (on ? goes : other)[(size_t) t] = 1;
                if (log) { std::snprintf (line, sizeof line, "   lost %.2f to %.2f s: %s (alike %.2f near, %.2f far)", a * (double) hop / SR, b * (double) hop / SR, on ? "goes on" : "another sound", seenNear, seenFar); log (line); }
            }
        }
        return any;
    };
    std::vector<char> base, weak ((size_t) T); spans (base);
    for (int t = 0; t < T; ++t) weak[(size_t) t] = ! base[(size_t) t];
    if (mend (weak, base, true)) { again(); spans (base); }
    if (stop && stop()) { O.stopped = true; return O; }
    // a place where pairs reach much less far than on both sides of it
    {
        Vec m ((size_t) T); double run = 0; for (int t = 0; t < T; ++t) { run = std::max (run, reach[(size_t) t]); m[(size_t) t] = run; }
        run = 0; bool any = false; for (int t = T - 1; t >= 0; --t) { run = std::max (run, reach[(size_t) t]); m[(size_t) t] = std::min (m[(size_t) t], run); weak[(size_t) t] = base[(size_t) t] && m[(size_t) t] > 0 && reach[(size_t) t] <= CUT * m[(size_t) t]; any = any || weak[(size_t) t]; }
        if (any && mend (weak, base, false)) { again(); spans (base); }
    }
    for (int t = 0; t < T; ++t) if (mended[(size_t) t]) tr[(size_t) t] = std::max (tr[(size_t) t], trust);
    return O;
}

// ---------------------------------------------------------------- both looks on the 10 ms grid
struct Both { Vec slope, span; std::vector<char> md, go, ot; std::vector<int> which; bool stopped = false; };
// slope: of the look that counts at each moment; span: over which the lines are followed (the better look's);
// md: a loss the sound was found again across; go: a loss the sound goes on across; ot: a loss after which another
// sound may start; which: 1 short, 2 long, 0 none.
inline Both both (const Vec& x, double loose, double trust, const Vec& speed, const std::vector<char>& brk, const Vec& cv, const Stop& stop, const Log& log)
{
    const long len = (long) x.size(); const int T = 1 + (int) (len / STEP);
    struct Side { Vec slope, trust, span; std::vector<char> md, go, ot; bool stopped = false; };
    auto side = [&] (int which) -> Side
    {
        const int n = LOOKS[which][0], hop = LOOKS[which][1], m = hop / STEP, Tl = 1 + (int) (len / hop);
        Vec sp ((size_t) Tl, 0.0), c ((size_t) Tl); std::vector<char> bk ((size_t) Tl, 0);
        for (int i = 0; i < Tl; ++i)
        {
            for (int q = 0; q < m; ++q) { const size_t ix = (size_t) std::min (i * m + q, T - 1); sp[(size_t) i] = q == 0 ? speed[ix] : std::max (sp[(size_t) i], speed[ix]); bk[(size_t) i] = bk[(size_t) i] || brk[ix]; }
            c[(size_t) i] = cv[(size_t) std::min (i * m, T - 1)];
        }
        const LookOut o = look (x, n, hop, loose, trust, sp, bk, c, stop, log); Side s; if (o.stopped) { s.stopped = true; return s; }
        Vec fp ((size_t) T); s.trust.resize ((size_t) T); s.span.resize ((size_t) T); s.md.resize ((size_t) T); s.go.resize ((size_t) T); s.ot.resize ((size_t) T);
        for (int t = 0; t < T; ++t)
        {
            const double u = (double) t * (double) STEP / (double) hop; const size_t near = (size_t) std::min (nearest (u), (long) Tl - 1);
            fp[(size_t) t] = between (o.p, u); s.trust[(size_t) t] = between (o.tr, u); s.span[(size_t) t] = o.span[near]; s.md[(size_t) t] = o.mended[near]; s.go[(size_t) t] = o.goes[near]; s.ot[(size_t) t] = o.other[near];
        }
        s.slope = gradient (fp); return s;
    };
    Both B; auto job = std::async (std::launch::async, [&] { return side (1); }); const Side s = side (0); const Side l = job.get();
    if (s.stopped || l.stopped) { B.stopped = true; return B; }
    std::vector<int> use ((size_t) T); Vec u1 ((size_t) T), u2 ((size_t) T);
    // the short look counts where its measurements agree, unless the long look's agree clearly better (partials too close for the short window)
    for (int t = 0; t < T; ++t) { const size_t i = (size_t) t; use[i] = (s.trust[i] >= trust && s.trust[i] >= l.trust[i] - EDGE) ? 1 : l.trust[i] >= trust ? 2 : 0; u1[i] = use[i] == 1; u2[i] = use[i] == 2; }
    const Vec v2 = uniformFilter (u2, 15, 1), v1 = uniformFilter (u1, 15, 1);
    B.slope.resize ((size_t) T); B.span.resize ((size_t) T); B.md.resize ((size_t) T); B.go.resize ((size_t) T); B.ot.resize ((size_t) T); B.which.resize ((size_t) T);
    for (int t = 0; t < T; ++t)
    {
        const size_t i = (size_t) t; const double vote = v2[i] - v1[i]; int u = use[i];
        u = (u > 0 && vote > 0 && l.trust[i] >= trust) ? 2 : (u > 0 && vote <= 0 && s.trust[i] >= trust) ? 1 : u;       // no flicker between the two
        u = (s.md[i] && ! l.md[i]) ? 1 : (l.md[i] && ! s.md[i]) ? 2 : u;                                            // a loss one look found the sound again across: that look speaks there
        B.slope[i] = u == 1 ? s.slope[i] : u == 2 ? l.slope[i] : 0.0; B.span[i] = std::max (s.span[i], l.span[i]);
        B.md[i] = s.md[i] || l.md[i]; B.go[i] = s.go[i] || l.go[i]; B.ot[i] = s.ot[i] || l.ot[i]; B.which[i] = (s.span[i] > 0 || l.span[i] > 0) ? (s.span[i] >= l.span[i] ? 1 : 2) : 0;
    }
    return B;
}

// ---------------------------------------------------------------- flattening
// The curve a sound is flattened with stays inside +-top cents (as it is up to +-lin).
inline Vec bound (Vec w, double lin = 1200.0, double top = 1800.0)
{
    for (auto& v : w) { const double a = std::abs (v); if (a > lin) v = (v < 0 ? -1.0 : 1.0) * (lin + (top - lin) * std::tanh ((a - lin) / (top - lin))); }
    return w;
}
// x played so that the movement p (cents per 10 ms) is gone: every sample takes 2^(c / 1200) of its time. ty: where
// x's frames are in the result, in the result's frames (inSamples: in its samples). The curve counts around its
// average (asItIs: as it is).
inline Vec flatten (const Vec& x, const Vec& p, Vec& ty, const Stop& stop = {}, bool asItIs = false, bool inSamples = false)
{
    const long n = (long) x.size(); const int T = (int) p.size(); double mean = 0; if (! asItIs) { for (double v : p) mean += v; mean /= (double) std::max (T, 1); }
    Vec pc (p); for (auto& v : pc) v -= mean;
    auto took = [&] (long i) { return std::pow (2.0, between (pc, (double) i / (double) STEP) / 1200.0); };
    // when each sample starts (tau), kept at the frames; then the places to read at for every moment of the result
    ty.assign ((size_t) T, 0.0); double tau = 0, last = 0;
    for (long i = 0; i < n; ++i) { if (i % STEP == 0 && i / STEP < T) ty[(size_t) (i / STEP)] = tau; if (i == n - 1) last = tau; tau += took (i); }
    for (int t = 0; t < T; ++t) if ((long) t * STEP > n - 1) ty[(size_t) t] = last;
    if (! inSamples) for (auto& v : ty) v /= (double) STEP;
    const long m = (long) last + 1; long i = 0; double a0 = 0.0, a1 = n > 1 ? took (0) : 0.0;                              // a0 = tau[i], a1 = tau[i + 1]
    auto place = [&] (long j)                           // where in x the result's moment j is read; asked for in rising order
    {
        const double want = (double) j;
        while (i < n - 2 && a1 <= want) { ++i; a0 = a1; a1 = a0 + took (i); }
        return (n < 2 || want >= a1) ? (double) (n - 1) : (1.0 / (a1 - a0)) * (want - a0) + (double) i;
    };
    // every place read at the speed the places around it give (stepAt), without holding the list of places (a number per sample)
    Vec o ((size_t) m, 0.0); double prev = 0.0, cur = place (0), next = m > 1 ? place (1) : cur;
    for (long j = 0; j < m; ++j)
    {
        if (stop && (j & 0xffff) == 0 && stop()) return {};
        const double step = m < 2 ? 1.0 : j == 0 ? next - cur : j == m - 1 ? cur - prev : (next - prev) / 2.0;
        double s = 0; readerTaps (cur, step, n, [&] (long k, double w) { s += x[(size_t) k] * w; }); o[(size_t) j] = s;
        prev = cur; cur = next; if (j + 2 < m) next = place (j + 2);
    }
    return o;
}

// ---------------------------------------------------------------- how still the lines stand
// How still the lines of x stand once x is flattened with the curve c (cents per 10 ms, taken as it is): for the
// moments from .. to - 1 of x, per block of the spectrum, how gathered the block's energy is. Seen through the long
// window: a line that stands still has its energy in two or three bins, one that still moves spreads it over more,
// and two sounds that do not lie at the same place hold twice the lines. The measure: the number of values in the
// block times the sum of the squared energies over the square of their sum (1: spread evenly, 2: noise, the number
// of values: all in one). Blocks FLOOR_HZ wide that fade in and out and overlap by half, so that a line moving from
// one to the next changes nothing.
struct Gathered
{
    Vec g; bool stopped = false;                        // row i (GBLOCKS values) belongs to the moment from + i
    const double* row (int i) const { return g.data() + (size_t) i * (size_t) GBLOCKS; }
};
inline Gathered gathered (const Vec& x, const Vec& c, int from, int to, const Stop& stop = {})
{
    Gathered O; const int T = (int) c.size(); from = std::clamp (from, 0, T); to = std::clamp (to, from, T); const int rows = to - from;
    Vec ty; const Vec y = flatten (x, c, ty, stop, true, true); if (y.empty()) { O.stopped = true; return O; }
    const int N = LOOKS[1][0], hop = GHOP, H = N / 2, NV = (GBLOCKS + 1) * GBLOCK; const long m = (long) y.size(); const int Tf = 1 + (int) (m / hop);
    std::vector<int> first ((size_t) rows), slot ((size_t) Tf, -1); Vec part ((size_t) rows); int frames = 0;       // the two long frames each moment lies between
    for (int r = 0; r < rows; ++r)
    {
        const double v = std::clamp (ty[(size_t) (from + r)] / (double) hop, 0.0, (double) (Tf - 1)); const int i0 = std::min ((int) v, std::max (Tf - 2, 0));
        first[(size_t) r] = i0; part[(size_t) r] = v - (double) i0;
        for (int f : { i0, std::min (i0 + 1, Tf - 1) }) if (slot[(size_t) f] < 0) slot[(size_t) f] = frames++;
    }
    Vec h ((size_t) N), w ((size_t) (2 * GBLOCK)), cs ((size_t) NV), sn ((size_t) NV), re ((size_t) N), im ((size_t) N), P ((size_t) NV), G ((size_t) frames * (size_t) GBLOCKS, 1.0);
    for (int i = 0; i < N; ++i) h[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * PI * (double) i / (double) N);
    double sw = 0; for (int i = 0; i < 2 * GBLOCK; ++i) { w[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * PI * ((double) i + 0.5) / (double) (2 * GBLOCK)); sw += w[(size_t) i]; }
    for (int k = 0; k < NV; ++k) { const double th = 2.0 * PI * (double) (k + GFIRST) / (double) (FINE * N); cs[(size_t) k] = std::cos (th); sn[(size_t) k] = std::sin (th); }
    // A frame under the window with as many zeros after it (FINE = 2) has its even samples as the real part and its odd
    // ones as the imaginary part of a transform of N points; the two halves come apart again value by value.
    const room::Transform tr (N); int done = 0;
    for (int f = 0; f < Tf; ++f)
    {
        if (slot[(size_t) f] < 0) continue;
        if (stop && (done++ & 15) == 0 && stop()) { O.stopped = true; return O; }
        const long at = (long) f * hop - H;
        for (int i = 0; i < H; ++i)
        {
            const long j = at + 2 * (long) i;
            re[(size_t) i] = (j >= 0 && j < m) ? y[(size_t) j] * h[(size_t) (2 * i)] : 0.0; im[(size_t) i] = (j + 1 >= 0 && j + 1 < m) ? y[(size_t) (j + 1)] * h[(size_t) (2 * i + 1)] : 0.0;
        }
        std::fill (re.begin() + H, re.end(), 0.0); std::fill (im.begin() + H, im.end(), 0.0); tr.run (re, im);
        double mean = 0;
        for (int k = 0; k < NV; ++k)
        {
            const size_t a = (size_t) (k + GFIRST), b = (size_t) (N - (k + GFIRST));
            const double er = 0.5 * (re[a] + re[b]), ei = 0.5 * (im[a] - im[b]), odr = 0.5 * (im[a] + im[b]), odi = 0.5 * (re[b] - re[a]);
            const double xr = er + cs[(size_t) k] * odr + sn[(size_t) k] * odi, xi = ei + cs[(size_t) k] * odi - sn[(size_t) k] * odr;
            P[(size_t) k] = xr * xr + xi * xi; mean += P[(size_t) k];
        }
        const double dust = DUST * mean / (double) NV; double* g = G.data() + (size_t) slot[(size_t) f] * (size_t) GBLOCKS;
        for (int j = 0; j < GBLOCKS; ++j)
        {
            const double* Q = P.data() + (size_t) j * (size_t) GBLOCK; double s1 = 0, s2 = 0;
            for (int i = 0; i < 2 * GBLOCK; ++i) { const double v = Q[i] + dust; s1 += v * w[(size_t) i]; s2 += v * v * w[(size_t) i]; }
            g[j] = s1 > 0.0 ? sw * s2 / (s1 * s1) : 1.0;
        }
    }
    O.g.resize ((size_t) rows * (size_t) GBLOCKS);
    for (int r = 0; r < rows; ++r)
    {
        const double* a = G.data() + (size_t) slot[(size_t) first[(size_t) r]] * (size_t) GBLOCKS; const double* b = G.data() + (size_t) slot[(size_t) std::min (first[(size_t) r] + 1, Tf - 1)] * (size_t) GBLOCKS;
        const double fr = part[(size_t) r]; for (int j = 0; j < GBLOCKS; ++j) O.g[(size_t) r * (size_t) GBLOCKS + (size_t) j] = a[j] * (1.0 - fr) + b[j] * fr;
    }
    return O;
}
// How many times stiller the lines stand in a than in b (rows of gathered): block by block the ratio, over the blocks
// that hold lines in either, averaged as ratios are (every such block has one voice, the strongest lines no more than
// the weakest); said: how many blocks spoke. 1 where no block holds lines: nothing can be said.
inline double stiller (const double* a, const double* b, int* said = nullptr)
{
    double lr = 0; int m = 0;
    for (int j = 0; j < GBLOCKS; ++j) if (std::max (a[j], b[j]) > LINES) { lr += std::log (std::max (a[j], 1.0) / std::max (b[j], 1.0)); ++m; }
    if (said != nullptr) *said = m;
    return std::exp (lr / (double) std::max (m, 1));
}
// A loss the sound was found again across (md): how far the sound moved in all is known, the path is not, so the
// path is a straight line (a moment more on each side: the two looks' moments lie that far apart).
inline void straighten (Vec& q, const std::vector<char>& md)
{
    const int T = (int) q.size();
    for (int t = 0; t < T; )
    {
        if (! md[(size_t) t]) { ++t; continue; }
        int e = t; while (e < T && md[(size_t) e]) ++e;
        const int a = std::max (t - 1, 0), b = std::min (e + 1, T); t = e;
        if (a == 0 && b == T) continue;
        const double va = a > 0 ? q[(size_t) a - 1] : q[(size_t) b], vb = b < T ? q[(size_t) b] : va;
        for (int i = a; i < b; ++i) q[(size_t) i] = va + (vb - va) * ((double) (i - a) + 1.0) / ((double) (b - a) + 1.0);
    }
}
// The curve with its fast part only where that part is proven: where the lines of x stand stiller once x is flattened
// with the whole curve than with the curve's slow part alone. Two sounds close to each other in pitch beat, their
// lines flutter, and a curve that follows the flutter adds one; a reading that jitters adds its jitter.
// False: stopped.
inline bool proven (const Vec& x, Vec& out, const Stop& stop, const Log& log)
{
    const int T = (int) out.size(); const double fs = SR / (double) STEP; const Vec slow = slowPart (out);
    Vec fast ((size_t) T); double most = 0, mean = 0; for (int t = 0; t < T; ++t) { fast[(size_t) t] = out[(size_t) t] - slow[(size_t) t]; most = std::max (most, std::abs (fast[(size_t) t])); mean += out[(size_t) t]; }
    if (! (most > 1.0)) return true;
    mean /= (double) T; const int w = 2 * (int) nearest (0.15 * fs) + 1;
    // the values of each of the two curves, evened over w moments block by block (a long sound: one curve after the other,
    // so that there are never two flattened sounds at a time)
    auto evened = [&] (const Vec& curve, Vec& into)
    {
        Vec c ((size_t) T); for (int t = 0; t < T; ++t) c[(size_t) t] = curve[(size_t) t] - mean;
        const Gathered g = gathered (x, c, 0, T, stop); if (g.stopped) return false;
        into.resize ((size_t) T * (size_t) GBLOCKS); Vec col ((size_t) T);
        for (int j = 0; j < GBLOCKS; ++j)
        {
            for (int t = 0; t < T; ++t) col[(size_t) t] = g.row (t)[j];
            const Vec e = uniformFilter (col, w, 1); for (int t = 0; t < T; ++t) into[(size_t) t * (size_t) GBLOCKS + (size_t) j] = e[(size_t) t];
        }
        return true;
    };
    Vec ga, gb;
    if ((long) x.size() <= TOGETHER) { auto job = std::async (std::launch::async, [&] { return evened (slow, gb); }); const bool a = evened (out, ga), b = job.get(); if (! (a && b)) return false; }
    else if (! evened (out, ga) || ! evened (slow, gb)) return false;
    Vec keep ((size_t) T);
    for (int t = 0; t < T; ++t)
    {
        const double ratio = stiller (ga.data() + (size_t) t * (size_t) GBLOCKS, gb.data() + (size_t) t * (size_t) GBLOCKS);
        keep[(size_t) t] = std::clamp ((ratio - FASTKEEP_NONE) / (FASTKEEP_ALL - FASTKEEP_NONE), 0.0, 1.0);
    }
    keep = gaussN (keep, 5.0); double km = 0, kl = 1e300, f2 = 0, f1 = 0;
    for (int t = 0; t < T; ++t) { out[(size_t) t] = slow[(size_t) t] + keep[(size_t) t] * fast[(size_t) t]; km += keep[(size_t) t]; kl = std::min (kl, keep[(size_t) t]); f1 += fast[(size_t) t]; f2 += sq (fast[(size_t) t]); }
    if (log) { char line[200]; f1 /= (double) T; std::snprintf (line, sizeof line, "the fast part of the curve (rms %.1f cents): kept %.2f on average, least %.2f", std::sqrt (std::max (f2 / (double) T - f1 * f1, 0.0)), km / (double) T, kl); log (line); }
    return true;
}

// ---------------------------------------------------------------- a step between two stretches, proven or not
// A passage a .. b - 1 between two stretches (the one before begins at ea, the one after ends before eb) inside which
// the curve out takes a step: how alike the sound's lines just after the passage are those just before it once the
// sound is flattened with out (0 .. 1), and how much further the curve would have to step for them to lie at the
// same place (cents, looked for ROUND either way). The short look's lines, of up to six moments on each side that
// lie two moments clear of the passage, each side's taken together. Inside the passage the curve stands still (the
// step at its end), and outside the two stretches as well.
struct Alike { double like = 0, more = 0; bool stopped = false; };
inline Alike stepAlike (const Vec& x, const Vec& out, int ea, int a, int b, int eb, const Stop& stop = {})
{
    Alike R; const int T = (int) out.size(); const long len = (long) x.size();
    const int fb0 = std::max (ea, a - 8), fb1 = std::max (ea + 1, a - 2), fa0 = std::min (eb - 1, b + 2), fa1 = std::min (eb, b + 8);      // the moments before: fb0 .. fb1 - 1, after: fa0 .. fa1 - 1
    const int lo = std::max (0, fb0 - 6), hi = std::min (T - 1, fa1 - 1 + 6); const long s0 = (long) lo * STEP, s1 = std::min (len, (long) hi * STEP + 1), n = s1 - s0;
    if (n < 2) return R;
    Vec cur ((size_t) (hi - lo + 1)); for (int t = lo; t <= hi; ++t) cur[(size_t) (t - lo)] = out[(size_t) t] - out[(size_t) a - 1];
    for (int t = a; t < b; ++t) cur[(size_t) (t - lo)] = 0.0;
    if (ea > lo) for (int t = lo; t < ea; ++t) cur[(size_t) (t - lo)] = cur[(size_t) (ea - lo)];
    if (eb <= hi) for (int t = eb; t <= hi; ++t) cur[(size_t) (t - lo)] = cur[(size_t) (eb - 1 - lo)];
    const Vec seg (x.begin() + s0, x.begin() + s1); Vec ty; const Vec y = flatten (seg, cur, ty, stop, true, true); if (y.empty()) { R.stopped = true; return R; }
    const Lines L = lines (y, LOOKS[0][0], STEP, stop); if (L.T < 0) { R.stopped = true; return R; }
    Vec Pb ((size_t) NB, 0.0), Pa ((size_t) NB, 0.0);
    auto add = [&] (int t, Vec& P)
    {
        const size_t r = (size_t) std::clamp (nearest (ty[(size_t) (t - lo)] / (double) STEP), 0L, (long) L.T - 1);
        for (int k = L.start[r]; k < L.start[r + 1]; ++k) P[(size_t) L.bin[(size_t) k]] += L.val[(size_t) k];
    };
    for (int t = fb0; t < fb1; ++t) add (t, Pb);
    for (int t = fa0; t < fa1; ++t) add (t, Pa);
    double nb = 0, na = 0; for (int i = 0; i < NB; ++i) { nb += sq (Pb[(size_t) i]); na += sq (Pa[(size_t) i]); }
    const double den = std::sqrt (nb * na); if (! (den > 0.0)) return R;
    const int S = (int) nearest (ROUND / RES); Vec C ((size_t) (2 * S + 1), 0.0);
    for (int s = -S; s <= S; ++s) { double v = 0; for (int i = std::max (0, -s); i < NB - std::max (0, s); ++i) v += Pb[(size_t) i] * Pa[(size_t) (i + s)]; C[(size_t) (s + S)] = v / den; }
    const Peak k = peakOf (C.data(), S, 0); if (! k.inside) return R;          // most alike at the edge of where it is looked for: not found
    R.like = k.pk; R.more = k.shift; return R;
}

// ---------------------------------------------------------------- the chain as far as it is kept
// q: the chain (cents, 10 ms grid); span: over which each moment's lines are followed (0: not); runs: the moments
// that hang together (followed loosely or strictly); raw: the spans in the read sound's own time; apart: pauses;
// goes: losses the sound goes on across; md: losses it was found again across (a straight line, see straighten);
// other: losses after which another sound may start (what follows is most alike what came before somewhere else,
// or was found there and not proven).
// A stretch: moments in a row that hang together. Inside a stretch the chain counts as it is where the lines are
// followed over a long span; where they are followed over short spans only (events handing over to each other can
// make such a chain run away) it counts against its own average over REACH times that span (not when plain).
// Where nothing is read nothing may be made up: the curve stands still, and the next stretch goes on where the last
// one left it. It begins afresh only where a new sound starts. That is at a pause: the sounds on both sides stand
// around their own averages, and the step between them lies inside the pause. And it is at a place where another
// sound may start, for events only: an event (a stretch shorter than LONG, or without lines followed over a long
// span) stands around its own average there, at the level the sustained sounds hold around it (zero if there are
// none), and the curve steps to it just before it begins, over RAMP, or in a straight line where the two stretches
// are less than GAP apart. A sustained sound does not begin afresh there (an engine that drops fast looks like
// another sound starting): it goes on where the last sustained sound left the curve.
// prior: the curve the read sound was flattened with. Given (the rounds), the curve moves as prior does wherever it
// would stand still: what an earlier round read stays in the curve the sound is flattened with next, so a passage
// that is lost in one round can be read again in the next. Not given: it stands still (the curve that is kept).
// stepped: the passages (first moment, the moment after the last) in which the curve takes a step of NEAR or more
// outside a pause are added to it.
// The curve that is kept (the sound x is given) holds less, so that nothing in it is made up. Beside a sustained
// sound an event is not kept: short stretches next to an engine or a glide are where a reading goes wrong (a line
// that beats, a thump, a few moments in noise), and a step to an event and back is made up wherever the sound in
// truth runs on (it moved faster than it was looked for, or its strongest line handed over to another). Where no
// sound is sustained the events are the sound (zaps, chirps): the longest part stands, and every other part keeps
// its place only if the step to it is proven: with the sound flattened by the curve, the lines after the step are
// ALIKE those before it (see stepAlike), and the part is put where they are found, one part after the other going
// out from the longest. A step under NEAR needs no proof. A part whose place is not proven is not kept: the curve
// stands still across it. stopped: set when stop said to give up (the answer is then not used).
inline Vec compose (const Vec& chain, const Vec& span, bool plain, const std::vector<char>* runs, const Vec* raw, const std::vector<char>& apart, const std::vector<char>& goes,
                    const std::vector<char>& md, const std::vector<char>& other, const Vec* prior, std::vector<char>& held,
                    std::vector<std::pair<int, int>>* stepped = nullptr, const Vec* x = nullptr, const Stop& stop = {}, const Log& log = {}, bool* stopped = nullptr)
{
    const int T = (int) chain.size(); const double fs = SR / (double) STEP; held.assign ((size_t) T, 0); std::vector<char> run ((size_t) T); Vec out ((size_t) T, 0.0);
    for (int t = 0; t < T; ++t) { held[(size_t) t] = span[(size_t) t] > 0; run[(size_t) t] = held[(size_t) t] || (runs != nullptr && (*runs)[(size_t) t]); }
    Vec q (chain); straighten (q, md);
    auto pr = [&] (int t) { return prior != nullptr ? (*prior)[(size_t) t] : 0.0; };
    // a crumb (a stretch shorter than CRUMB next to a short loss) is no sound of its own: it hangs onto its neighbour, the curve standing still across the loss
    {
        std::vector<std::pair<int, int>> rs; for (int t = 0; t < T; ) { if (! run[(size_t) t]) { ++t; continue; } int b = t; while (b < T && run[(size_t) b]) ++b; rs.push_back ({ t, b }); t = b; }
        for (size_t k = 0; k + 1 < rs.size(); ++k)
        {
            const int a = rs[k].second, b = rs[k + 1].first; bool not_ = false; for (int t = a; t < b; ++t) not_ = not_ || apart[(size_t) t] || other[(size_t) t];
            if ((double) (b - a) <= GAP * fs && (double) std::min (rs[k].second - rs[k].first, rs[k + 1].second - rs[k + 1].first) < CRUMB * fs && ! not_) for (int t = a; t < b; ++t) run[(size_t) t] = 1;
        }
    }
    for (int t = 0; t < T; ++t) run[(size_t) t] = (run[(size_t) t] || goes[(size_t) t]) && ! (other[(size_t) t] && ! held[(size_t) t]);        // where another sound may start and nothing is followed strictly, nothing hangs together
    for (int t = 0; t < T; )                            // moments that hang together without one that is followed strictly: nothing is read there, it is no stretch
    {
        if (! run[(size_t) t]) { ++t; continue; }
        int b = t; bool any = false; while (b < T && run[(size_t) b]) { any = any || held[(size_t) b]; ++b; }
        if (! any) for (int i = t; i < b; ++i) run[(size_t) i] = 0;
        t = b;
    }
    std::vector<std::pair<int, int>> ends; std::vector<char> sus, evt; std::vector<int> kinds; Vec inc, h, cs; char line[360]; int K = 0;
    const int gap = (int) nearest (GAP * fs), ramp = (int) nearest (RAMP * fs);
    auto first = [&] (int i) -> double& { return out[(size_t) ends[(size_t) i].first]; };
    auto last = [&] (int i) -> double& { return out[(size_t) ends[(size_t) i].second - 1]; };
    auto shift = [&] (int i, double d) { for (int t = ends[(size_t) i].first; t < ends[(size_t) i].second; ++t) out[(size_t) t] += d; };
    auto sumOf = [&] (int i) { double v = 0; for (int t = ends[(size_t) i].first; t < ends[(size_t) i].second; ++t) v += out[(size_t) t]; return v; };
    auto moved = [&] (int i, int j) { return pr (ends[(size_t) j].first) - pr (ends[(size_t) i].second - 1); };        // how far prior moves from the end of stretch i to the start of stretch j
    for (;;)                                            // once, or (the curve that is kept) until no stretch is taken out any more
    {
        std::fill (out.begin(), out.end(), 0.0); ends.clear(); sus.clear();
        for (int at = 0; at < T; )
        {
            if (! run[(size_t) at]) { ++at; continue; }
            const int a = at; int b = a; while (b < T && run[(size_t) b]) ++b;
            at = b; const int n = b - a; inc.assign ((size_t) std::max (n - 1, 0), 0.0);
            for (int i = 0; i + 1 < n; ++i) inc[(size_t) i] = q[(size_t) (a + i) + 1] - q[(size_t) (a + i)];
            auto longly = [&] (int i) { return (raw != nullptr ? (*raw)[(size_t) (a + i)] : span[(size_t) (a + i)]) >= LONG; };
            auto shortly = [&] (int i) { return held[(size_t) (a + i)] && ! longly (i); };
            if (! plain)
                for (int c = 0; c < n; )
                {
                    if (! shortly (c)) { ++c; continue; }
                    int d = c; while (d < n && shortly (d)) ++d;
                    const int m = d - c;
                    if (m >= 2)
                    {
                        cs.assign ((size_t) m + 1, 0.0); for (int i = 0; i < m; ++i) cs[(size_t) i + 1] = cs[(size_t) i] + q[(size_t) (a + c + i)];
                        h.resize ((size_t) m);
                        for (int i = 0; i < m; ++i)
                        {
                            const long rr = std::max (1L, nearest (REACH * span[(size_t) (a + c + i)] * fs)), lo = std::max (0L, (long) i - rr), hi = std::min ((long) m, (long) i + rr + 1);
                            h[(size_t) i] = q[(size_t) (a + c + i)] - (cs[(size_t) hi] - cs[(size_t) lo]) / (double) (hi - lo);
                        }
                        for (int i = 0; i + 1 < m; ++i) inc[(size_t) (c + i)] = h[(size_t) i + 1] - h[(size_t) i];
                    }
                    c = d;
                }
            for (int i = 0; i + 1 < n; ++i) if (! (held[(size_t) (a + i)] && held[(size_t) (a + i) + 1])) inc[(size_t) i] = pr (a + i + 1) - pr (a + i);      // followed loosely only: nothing is read there
            double acc = 0, mean = 0; out[(size_t) a] = 0.0; for (int i = 0; i + 1 < n; ++i) { acc += inc[(size_t) i]; out[(size_t) (a + i) + 1] = acc; mean += acc; }
            mean /= (double) n; for (int i = 0; i < n; ++i) out[(size_t) (a + i)] -= mean;
            bool anyLong = false; for (int i = 0; i < n; ++i) anyLong = anyLong || (held[(size_t) (a + i)] && longly (i));
            ends.push_back ({ a, b }); sus.push_back (anyLong && (double) n >= LONG * fs);
        }
        if (ends.empty()) return out;
        // between the stretches
        K = (int) ends.size(); kinds.assign ((size_t) std::max (K - 1, 0), 0);          // what lies between stretch k and the next: 2 a pause, 1 a place where another sound may start, 0 nothing known
        for (int k = 0; k + 1 < K; ++k)
        {
            bool pause = false, oth = false; for (int t = ends[(size_t) k].second; t < ends[(size_t) k + 1].first; ++t) { pause = pause || apart[(size_t) t]; oth = oth || other[(size_t) t]; }
            kinds[(size_t) k] = pause ? 2 : oth ? 1 : 0;
        }
        evt.assign ((size_t) K, 0);                     // evt: the stretch belongs to a part that holds no sustained sound
        std::vector<int> drop; bool beside = false; double dStep = 0; int dA = 0, dB = 0; Alike why;       // the stretches to take out, and why
        for (int k = 0; k < K && drop.empty(); )
        {
            int j = k; while (j < K - 1 && kinds[(size_t) j] != 2) ++j;                                     // the stretches k .. j: no pause between them
            std::vector<std::vector<int>> parts (1, std::vector<int> (1, k));                               // a part: stretches with nothing known between them, each going on where the one before left the curve
            for (int i = k + 1; i <= j; ++i)
            {
                if (kinds[(size_t) i - 1] == 0) { shift (i, last (i - 1) + moved (i - 1, i) - first (i)); parts.back().push_back (i); }
                else parts.push_back (std::vector<int> (1, i));
            }
            std::vector<int> S;                             // the parts that hold a sustained sound
            for (size_t u = 0; u < parts.size(); ++u) { bool any = false; for (int i : parts[u]) any = any || sus[(size_t) i]; if (any) S.push_back ((int) u); }
            for (size_t u = 1; u < S.size(); ++u)           // each goes on where the one before left the curve
            {
                const int e = parts[(size_t) S[u - 1]].back(), f = parts[(size_t) S[u]].front(); const double d = last (e) + moved (e, f) - first (f);
                for (int i : parts[(size_t) S[u]]) shift (i, d);
            }
            if (! S.empty())                                // and together they stand around the average of their sustained sounds
            {
                double sum = 0; long cnt = 0; for (int u : S) for (int i : parts[(size_t) u]) if (sus[(size_t) i]) { sum += sumOf (i); cnt += ends[(size_t) i].second - ends[(size_t) i].first; }
                const double m = sum / (double) cnt; for (int u : S) for (int i : parts[(size_t) u]) shift (i, -m);
            }
            for (size_t u = 0; u < parts.size(); ++u)       // a part of events only: around its own average, at the level the sustained sounds hold around it
            {
                if (std::find (S.begin(), S.end(), (int) u) != S.end()) continue;
                const std::vector<int>& P = parts[u]; double sum = 0; long cnt = 0; for (int i : P) { sum += sumOf (i); cnt += ends[(size_t) i].second - ends[(size_t) i].first; }
                int bf = -1, af = -1; for (int v : S) { if (v < (int) u) bf = v; else if (af < 0) af = v; }
                const double B = bf >= 0 ? last (parts[(size_t) bf].back()) : af >= 0 ? first (parts[(size_t) af].front()) : 0.0, m = sum / (double) cnt;
                for (int i : P) { shift (i, B - m); evt[(size_t) i] = 1; }
            }
            if (x != nullptr && ! S.empty())                // beside a sustained sound an event is not kept
            {
                for (int i = k; i <= j; ++i) if (! sus[(size_t) i]) drop.push_back (i);
                beside = ! drop.empty();
            }
            else if (x != nullptr && parts.size() > 1)      // events only: placed against the longest part, each by the step to it (those ahead of it by the step from them)
            {
                auto lengthOf = [&] (size_t u) { long v = 0; for (int i : parts[u]) v += ends[(size_t) i].second - ends[(size_t) i].first; return v; };
                size_t f0 = 0; for (size_t u = 1; u < parts.size(); ++u) if (lengthOf (u) > lengthOf (f0)) f0 = u;
                std::vector<size_t> order; for (size_t u = f0; u-- > 0; ) order.push_back (u);
                for (size_t u = f0 + 1; u < parts.size(); ++u) order.push_back (u);
                for (size_t u : order)
                {
                    const std::vector<int>& P = parts[u]; const bool back = u < f0; const int i0 = back ? P.back() : P.front() - 1, i1 = i0 + 1;
                    const int a = ends[(size_t) i0].second, b = ends[(size_t) i1].first; const double d = out[(size_t) b] - out[(size_t) a - 1];
                    if (std::abs (d - (pr (b) - pr (a - 1))) < NEAR) continue;
                    const Alike A = stepAlike (*x, out, ends[(size_t) i0].first, a, b, ends[(size_t) i1].second, stop);
                    if (A.stopped) { if (stopped != nullptr) *stopped = true; return out; }
                    if (A.like < ALIKE) { drop = P; dStep = d; dA = a; dB = b; why = A; break; }
                    if (log) { std::snprintf (line, sizeof line, "   a step of %+.0f cents between %.2f and %.2f s: the two sides are alike %.2f there, %+.0f cents off: placed", d, (double) a / fs, (double) b / fs, A.like, A.more); log (line); }
                    for (int i : P) shift (i, back ? -A.more : A.more);
                }
            }
            k = j + 1;
        }
        if (drop.empty()) break;
        for (int i : drop)
        {
            const int a = ends[(size_t) i].first, b = ends[(size_t) i].second;
            if (log)
            {
                if (beside) std::snprintf (line, sizeof line, "   stretch %.2f to %.2f s: an event beside a sustained sound, not kept", (double) a / fs, (double) b / fs);
                else std::snprintf (line, sizeof line, "   stretch %.2f to %.2f s: an event whose place is not proven, not kept (a step of %+.0f cents between %.2f and %.2f s: the two sides are alike %.2f there, %+.0f cents off)", (double) a / fs, (double) b / fs, dStep, (double) dA / fs, (double) dB / fs, why.like, why.more);
                log (line);
            }
            for (int t = a; t < b; ++t) { run[(size_t) t] = 0; held[(size_t) t] = 0; }
        }
    }
    if (log)
        for (int i = 0; i < K; ++i)
        {
            static const char* const after[3] = { "nothing known", "another sound may start", "a pause" };
            const int n = std::snprintf (line, sizeof line, "   stretch %.2f to %.2f s, %s, %+.0f to %+.0f cents", (double) ends[(size_t) i].first / fs, (double) ends[(size_t) i].second / fs,
                                         sus[(size_t) i] ? "sustained" : evt[(size_t) i] ? "an event of its own" : "an event", first (i), last (i));
            if (i < K - 1 && n > 0 && n < (int) sizeof line) std::snprintf (line + n, sizeof line - (size_t) n, "; then %s", after[kinds[(size_t) i]]);
            log (line);
        }
    for (int p = 0; p <= K; ++p)                        // the passage a .. b - 1 before stretch p (after the last one: p = K)
    {
        const int a = p > 0 ? ends[(size_t) p - 1].second : 0, b = p < K ? ends[(size_t) p].first : T; if (b <= a) continue;
        const int n = b - a;
        if (p == 0) { const double vb = out[(size_t) b]; for (int t = a; t < b; ++t) out[(size_t) t] = vb + pr (t) - pr (b); continue; }
        if (p == K) { const double va = out[(size_t) a - 1]; for (int t = a; t < b; ++t) out[(size_t) t] = va + pr (t) - pr (a - 1); continue; }
        // the passage moves as prior does; what that leaves of the way from va to vb is a step, taken where `step` goes from 0 to 1
        const double va = out[(size_t) a - 1], vb = out[(size_t) b], left = (vb - va) - (pr (b) - pr (a - 1));
        int p0 = 0, p1 = 0; if (kinds[(size_t) p - 1] == 2) { p0 = -1; for (int t = a; t < b; ++t) if (apart[(size_t) t]) { if (p0 < 0) p0 = t - a; p1 = t - a + 1; } }
        for (int i = 0; i < n; ++i)
        {
            double step;
            if (kinds[(size_t) p - 1] == 2) step = i < p0 ? 0.0 : i >= p1 ? 1.0 : ((double) (i - p0) + 1.0) / ((double) (p1 - p0) + 1.0);                   // inside the pause
            else if (n <= gap) step = ((double) i + 1.0) / ((double) n + 1.0);
            else if (evt[(size_t) p]) step = i < n - ramp ? 0.0 : ((double) (i - (n - ramp)) + 1.0) / ((double) ramp + 1.0);                        // events begin afresh there: the curve steps to them just before they begin
            else step = i < ramp ? ((double) i + 1.0) / ((double) ramp + 1.0) : 1.0;                                                                // after events, at once back to where the sustained sound goes on
            out[(size_t) (a + i)] = va + (pr (a + i) - pr (a - 1)) + left * step;
        }
        if (stepped != nullptr && kinds[(size_t) p - 1] != 2 && std::abs (left) >= NEAR) stepped->push_back ({ a, b });
    }
    return out;
}

// ---------------------------------------------------------------- the reader
struct Track { Vec cents; std::vector<char> held; bool stopped = false; };      // held: which look follows each moment (1 short, 2 long, 0 none)
inline Track track (const Vec& x, const Stop& stop = {}, const Log& log = {})
{
    const long n = (long) x.size(); const int T = 1 + (int) (n / STEP); const double fs = SR / (double) STEP;
    Track R; R.cents.assign ((size_t) T, 0.0); R.held.assign ((size_t) T, 0); if (n < LEAST) return R;
    char line[256];
    // pauses: far under what sounds within half a second before and after
    std::vector<char> brk ((size_t) T, 0);
    {
        Vec e ((size_t) T, 0.0);
        for (int t = 0; t < T; ++t) { const long a = (long) t * STEP - 2048; double s = 0; for (long i = std::max (0L, -a); i < 4096 && a + i < n; ++i) s += sq (x[(size_t) (a + i)]); e[(size_t) t] = s / 4096.0; }
        const int k = (int) nearest (0.5 * fs) / 2; const Vec cm = largestWithin (e, k);
        for (int t = 0; t < T; ++t) brk[(size_t) t] = e[(size_t) t] < PAUSE * std::min (cm[(size_t) std::clamp (t - k, 0, T - 1)], cm[(size_t) std::clamp (t + k, 0, T - 1)]);
    }
    struct Read { Vec r, span, raw; std::vector<char> md, md0, go, ot; std::vector<int> which; bool stopped = false; };       // md0: md before the proofs
    // made: the moments at which the curve the sound is flattened with takes a step that nothing proves (see compose,
    // stepped). No reading follows them: a reading that did would hold on to the step, and the curve that is kept with it.
    std::vector<char> made ((size_t) T, 0);
    auto mark = [&] (const std::vector<std::pair<int, int>>& st) { std::fill (made.begin(), made.end(), (char) 0); for (const auto& p : st) for (int t = p.first; t < p.second; ++t) made[(size_t) t] = 1; };
    // y: the sound flattened with the curve c; ty: where x's frames are in y, in y's frames
    auto read = [&] (const Vec& y, const Vec& ty, const Vec& c, double loose, double trust) -> Read
    {
        double mean = 0; for (double v : c) mean += v; mean /= (double) T;
        Vec k ((size_t) T); for (int t = 0; t < T; ++t) k[(size_t) t] = std::pow (2.0, -(c[(size_t) t] - mean) / 1200.0);       // a span of the flattened sound's time, in the sound's own time
        Vec sp = gradient (uniformFilter (c, 5, 1)); for (auto& v : sp) v = std::abs (v) * fs;
        sp = largestWithin (sp, 15); for (int t = 0; t < T; ++t) sp[(size_t) t] *= k[(size_t) t];                               // how fast the curve moves around each moment, cents per second of the flattened sound
        const int Ty = 1 + (int) ((long) y.size() / STEP); Vec frames ((size_t) T); for (int t = 0; t < T; ++t) frames[(size_t) t] = (double) t;
        Vec spy ((size_t) Ty), cvy ((size_t) Ty); std::vector<char> by ((size_t) Ty, 0);
        for (int j = 0; j < Ty; ++j)
        {
            const double own = interpAt ((double) j, ty, frames); spy[(size_t) j] = between (sp, own); cvy[(size_t) j] = between (c, own);
            by[(size_t) j] = brk[(size_t) std::clamp (nearest (own), 0L, (long) T - 1)];
        }
        for (int t = 0; t < T; ++t) if (brk[(size_t) t]) by[(size_t) std::clamp (nearest (ty[(size_t) t]), 0L, (long) Ty - 1)] = 1;
        const Both B = both (y, loose, trust, spy, by, cvy, stop, log); Read F; if (B.stopped) { F.stopped = true; return F; }
        Vec chain ((size_t) Ty); double acc = 0; for (int j = 0; j < Ty; ++j) { acc += B.slope[(size_t) j]; chain[(size_t) j] = acc; }
        F.r.resize ((size_t) T); F.span.resize ((size_t) T); F.raw.resize ((size_t) T); F.md.resize ((size_t) T); F.go.resize ((size_t) T); F.ot.resize ((size_t) T); F.which.resize ((size_t) T);
        for (int t = 0; t < T; ++t)
        {
            const size_t i = (size_t) t, near = (size_t) std::clamp (nearest (ty[i]), 0L, (long) Ty - 1);
            const bool none = brk[i] || made[i];
            F.r[i] = between (chain, ty[i]); F.raw[i] = none ? 0.0 : B.span[near]; F.span[i] = none ? 0.0 : B.span[near] * k[i];
            F.md[i] = B.md[near]; F.go[i] = B.go[near] && ! brk[i]; F.ot[i] = B.ot[near]; F.which[i] = B.which[near];
        }
        // A sound found again across a loss (a sound's lines fit those of its own harmonics in part, and can be found where
        // they are not): the curve goes through the loss only if the sound is heard to move like that inside the loss, that
        // is if its lines stand stiller there with the curve gliding through than with a curve that takes the same step at
        // once in the middle of the loss (the two curves are the same outside the loss). Not proven: nothing is known of the
        // passage. A finding that moves the curve by less than NEAR cents in all needs no proof: it cannot add more than
        // that, and a curve that stood still at every such place would read too little every time (a glide under a row of
        // clicks).
        F.md0 = F.md; Vec qs ((size_t) T); for (int t = 0; t < T; ++t) qs[(size_t) t] = c[(size_t) t] + F.r[(size_t) t];
        straighten (qs, F.md); const int W = (int) nearest (AROUND * fs);
        for (int at = 0; at < T; )
        {
            if (! F.md0[(size_t) at]) { ++at; continue; }
            const int r0 = at; int r1 = r0; while (r1 < T && F.md0[(size_t) r1]) ++r1;
            at = r1; const int a = std::max (r0 - 1, 0), b = std::min (r1 + 1, T); if (a == 0 || b == T) continue;
            const int lo = std::max (0, a - W), hi = std::min (T, b + W), h = (a + b) / 2; const double va = qs[(size_t) a - 1], vb = qs[(size_t) b], mid = 0.5 * (va + vb);
            if (std::abs (vb - va) < NEAR)
            {
                if (log) { std::snprintf (line, sizeof line, "   found again %.2f to %.2f s, %.0f cents: kept as it is", (double) r0 / fs, (double) r1 / fs, vb - va); log (line); }
                continue;
            }
            Vec qa ((size_t) (hi - lo)); for (int t = lo; t < hi; ++t) qa[(size_t) (t - lo)] = qs[(size_t) t] - mid;
            Vec qb (qa); for (int t = a; t < b; ++t) qb[(size_t) (t - lo)] = (t < h ? va : vb) - mid;
            const Vec seg (x.begin() + (long) lo * STEP, x.begin() + std::min (n, (long) hi * STEP)); const int k0 = std::max (a - lo - 2, 0), k1 = std::min (b - lo + 2, hi - lo);
            double ga[GBLOCKS], gb[GBLOCKS];
            auto average = [&] (const Vec& curve, double* into)       // over the moments of the loss
            {
                const Gathered g = gathered (seg, curve, k0, k1, stop); if (g.stopped) return false;
                for (int j = 0; j < GBLOCKS; ++j) { double v = 0; for (int i = 0; i < k1 - k0; ++i) v += g.row (i)[j]; into[j] = v / (double) (k1 - k0); }
                return true;
            };
            if (! average (qa, ga) || ! average (qb, gb)) { F.stopped = true; return F; }
            int said = 0; const double ratio = stiller (ga, gb, &said); const bool stays = ratio >= STEADY;
            if (log) { std::snprintf (line, sizeof line, "   found again %.2f to %.2f s, %.0f cents: the lines stand %.2f times stiller with the glide than with the step (%d blocks): %s", (double) r0 / fs, (double) r1 / fs, vb - va, ratio, said, stays ? "kept" : "not proven, nothing is known of this passage"); log (line); }
            if (! stays) for (int t = r0; t < r1; ++t) { F.span[(size_t) t] = 0.0; F.raw[(size_t) t] = 0.0; F.md[(size_t) t] = 0; F.ot[(size_t) t] = 1; }       // found somewhere else and not proven: another sound may start there
        }
        return F;
    };
    // every stretch of moments in a row for which on (t) holds
    auto stretches = [&] (const std::function<bool (int)>& on, const std::function<void (int, int)>& use)
    {
        for (int t = 0; t < T; ) { if (! on (t)) { ++t; continue; } int b = t; while (b < T && on (b)) ++b; use (t, b); t = b; }
    };
    struct Round { double left, share; Vec q; Read F; std::vector<char> got; Vec c; };          // c: the curve the sound was flattened with for this reading
    std::vector<Round> kept; Vec c ((size_t) T, 0.0), y, ty ((size_t) T), q; Read F; std::vector<char> got ((size_t) T, 0), held;
    const Vec* now = &x;                                // the sound as it is read: x itself until it is flattened (then y)
    for (int t = 0; t < T; ++t) ty[(size_t) t] = (double) std::min ((long) t * STEP, n - 1) / (double) STEP;
    bool haveC = true; int it = 0, stall = 0; double before = -1.0; bool any = false; Vec prior ((size_t) T, 0.0);
    // The curve to flatten with next: the chain as far as it hangs together, moving as the curve before did where nothing
    // is read, its fast part only where that is proven. The sound flattened before is let go first: never two at a time.
    auto next = [&] (const Vec& before_) -> bool
    {
        y = Vec(); now = &x; std::vector<std::pair<int, int>> st; Vec g = compose (q, F.span, true, nullptr, nullptr, brk, F.go, F.md, F.ot, &before_, held, &st);
        if (! proven (x, g, stop, log)) return false;
        c = bound (std::move (g)); mark (st); y = flatten (x, c, ty, stop); now = &y; return ! (y.empty() || (stop && stop()));
    };
    for (;;)
    {
        F = read (*now, ty, c, 3.0, 0.6); if (F.stopped) { R.stopped = true; return R; }
        q.resize ((size_t) T); any = false; long ng = 0;
        for (int t = 0; t < T; ++t) { q[(size_t) t] = c[(size_t) t] + F.r[(size_t) t]; got[(size_t) t] = F.span[(size_t) t] > 0; any = any || got[(size_t) t]; ng += got[(size_t) t]; }
        if (! any) { if (log) { std::snprintf (line, sizeof line, "round %d: nothing followed", it); log (line); } break; }
        double s2 = 0, most = 0;                        // what was read on top of the curve, every stretch around its own average
        stretches ([&] (int t) { return (bool) got[(size_t) t]; }, [&] (int a, int b)
        {
            double m = 0; for (int t = a; t < b; ++t) m += F.r[(size_t) t]; m /= (double) (b - a);
            for (int t = a; t < b; ++t) { const double v = F.r[(size_t) t] - m; s2 += v * v; most = std::max (most, std::abs (v)); }
        });
        const double left = std::sqrt (s2 / (double) ng), share = (double) ng / (double) T;
        if (log) { std::snprintf (line, sizeof line, "round %d: followed %.2f, left to read rms %.1f, at most %.0f", it, share, left, most); log (line); }
        ++it; stall = (before >= 0.0 && left > 0.7 * before) ? stall + 1 : 0; kept.push_back ({ left, share, q, F, got, c });
        if ((left < DONE_RMS && most < DONE_MOST) || stall >= 2 || it >= PASSES) break;
        before = left; prior = c; if (! next (prior)) { R.stopped = true; return R; }
    }
    if (kept.size() > 1)                                // the reading that left the least to read (of those that follow about as much as the best) is the one to go on from
    {
        double top = 0; for (auto& k : kept) top = std::max (top, k.share);
        size_t best = kept.size() - 1; bool have = false;
        for (size_t k = 1; k < kept.size(); ++k) if (kept[k].share >= 0.8 * top && (! have || kept[k].left < kept[best].left)) { best = k; have = true; }
        if (best != kept.size() - 1 && kept[best].left < 0.7 * kept.back().left)
        {
            if (log) { std::snprintf (line, sizeof line, "going on from the round that left %.1f", kept[best].left); log (line); }
            q = kept[best].q; F = kept[best].F; got = kept[best].got; prior = kept[best].c; haveC = false; any = false; for (char g : got) any = any || g;
        }
    }
    std::vector<char> runs; const std::vector<char>* runsP = nullptr;
    if (any)
    {
        // the curve for the last, strict reading. The sound is flattened again even where the curve stays as it was: the
        // proof needs the room of a flattened sound, and there are never two at a time.
        runs.resize ((size_t) T); for (int t = 0; t < T; ++t) runs[(size_t) t] = got[(size_t) t] || F.go[(size_t) t];
        runsP = &runs; const Vec was = haveC ? c : prior; const Vec* pr = haveC ? &was : &prior;
        y = Vec(); now = &x; std::vector<std::pair<int, int>> st; Vec g = compose (q, F.span, true, nullptr, nullptr, brk, F.go, F.md, F.ot, pr, held, &st);
        if (! proven (x, g, stop, log)) { R.stopped = true; return R; }
        g = bound (std::move (g)); double far = 0; for (int t = 0; t < T; ++t) if (got[(size_t) t]) far = std::max (far, std::abs (g[(size_t) t] - c[(size_t) t]));
        if (! haveC || far > 2.0) { c = g; mark (st); }
        y = flatten (x, c, ty, stop); now = &y; if (y.empty() || (stop && stop())) { R.stopped = true; return R; }
        F = read (*now, ty, c, 1.0, TRUST); if (F.stopped) { R.stopped = true; return R; }
        for (int t = 0; t < T; ++t) q[(size_t) t] = c[(size_t) t] + F.r[(size_t) t];
        // what reading the flattened sound again still finds: where that is no less than half of the curve's own movement, the curve is not kept
        const int w = 2 * (int) nearest (0.5 * fs) + 1;
        auto loc = [&] (const Vec& v) { Vec v2 (v.size()); for (size_t i = 0; i < v.size(); ++i) v2[i] = v[i] * v[i]; const Vec m = uniformFilter (v, w, 1), m2 = uniformFilter (v2, w, 1); Vec o (v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = std::sqrt (std::max (m2[i] - m[i] * m[i], 0.0)); return o; };
        // (by its slow part: whether its fast part counts is for the proof to say)
        Vec rg ((size_t) T, 0.0); for (int t = 1; t < T; ++t) rg[(size_t) t] = rg[(size_t) t - 1] + ((F.md0[(size_t) t] || F.md0[(size_t) t - 1]) ? 0.0 : F.r[(size_t) t] - F.r[(size_t) t - 1]);       // without the steps where a loss was mended in this reading
        rg = slowPart (rg); const Vec lr = loc (rg), lq = loc (q); std::vector<char> unsure ((size_t) T), hd ((size_t) T); long nh = 0;
        for (int t = 0; t < T; ++t) { hd[(size_t) t] = F.span[(size_t) t] > 0; nh += hd[(size_t) t]; unsure[(size_t) t] = lr[(size_t) t] > SURE_CENTS && lr[(size_t) t] > SURE_SHARE * lq[(size_t) t] && hd[(size_t) t]; }
        if (nh > 0)                                     // and as a whole: a reading that leaves half of its own movement to be read again is none
        {
            Vec ra = rg, qa = q;
            stretches ([&] (int t) { return (bool) hd[(size_t) t]; }, [&] (int a, int b)
            {
                double m1 = 0, m2 = 0; for (int t = a; t < b; ++t) { m1 += ra[(size_t) t]; m2 += qa[(size_t) t]; } m1 /= (double) (b - a); m2 /= (double) (b - a);
                for (int t = a; t < b; ++t) { ra[(size_t) t] -= m1; qa[(size_t) t] -= m2; }
            });
            auto spread = [&] (const Vec& v) { double m = 0; for (int t = 0; t < T; ++t) if (hd[(size_t) t]) m += v[(size_t) t]; m /= (double) nh; double s = 0; for (int t = 0; t < T; ++t) if (hd[(size_t) t]) s += sq (v[(size_t) t] - m); return std::sqrt (s / (double) nh); };
            const double sr = spread (ra), sq_ = spread (qa); if (sr > 10.0 && sr > SURE_SHARE * sq_) unsure = hd;
            if (log) { long nu = 0; for (char u : unsure) nu += u; std::snprintf (line, sizeof line, "last: followed %.2f, read on top rms %.1f, not kept %.2f", (double) nh / (double) T, spread (F.r), (double) nu / (double) T); log (line); }
        }
        for (int t = 0; t < T; ++t) if (unsure[(size_t) t]) { F.span[(size_t) t] = 0.0; F.raw[(size_t) t] = 0.0; }
    }
    if (q.size() != (size_t) T) q.assign ((size_t) T, 0.0);
    if (F.span.size() == (size_t) T)                    // beside a cut (see cuts): the curve stands still where the reading does not count, and nothing is followed on the quiet side
    {
        const std::vector<int> src = besideCuts (cuts (x), T);
        for (int t = 0; t < T; )
        {
            const int s = src[(size_t) t];
            if (s < 0) { const size_t i = (size_t) t; F.span[i] = 0.0; F.raw[i] = 0.0; F.go[i] = 0; if (runsP != nullptr) runs[i] = 0; ++t; continue; }
            if (s == t) { ++t; continue; }
            // the frames t .. e - 1 hold on to frame s. Going out from s their own reading counts as far as the look that follows s
            // follows them too, if the sound's lines lie high enough for that look; from there on out the curve stands still
            int e = t; while (e < T && src[(size_t) e] == s && e != s) ++e;
            const int look = F.span[(size_t) s] > 0 ? F.which[(size_t) s] : 0;
            const bool high = (look == 1 || look == 2) && (1200.0 / std::log (2.0)) * SHIFT[look - 1] * lowness (x, (long) s * STEP) < STAND;
            int from = s;
            auto one = [&] (int u, int before)
            {
                const size_t i = (size_t) u;
                if (high && from == before && F.which[i] == look && F.span[i] > 0) { from = u; return; }
                const size_t j = (size_t) from; q[i] = q[j]; F.span[i] = F.span[j]; F.raw[i] = F.raw[j]; F.go[i] = F.go[j]; F.md[i] = F.md[j]; F.ot[i] = F.ot[j]; F.which[i] = F.which[j]; if (runsP != nullptr) runs[i] = runs[j];
            };
            if (t < s) for (int u = e - 1; u >= t; --u) one (u, u + 1); else for (int u = t; u < e; ++u) one (u, u - 1);
            t = e;
        }
    }
    if (F.span.size() != (size_t) T) return R;          // nothing was read
    y = Vec(); now = &x; bool gaveUp = false;           // the curve that is kept: nothing in it that is not proven (see compose)
    R.cents = gaussN (compose (q, F.span, false, runsP, &F.raw, brk, F.go, F.md, F.ot, nullptr, held, nullptr, &x, stop, log, &gaveUp), 1.0);
    if (gaveUp) { R.stopped = true; return R; }
    if (! proven (x, R.cents, stop, log)) { R.stopped = true; return R; }                              // the fast part of the curve has to earn its place (see proven)
    for (int t = 0; t < T; ++t) R.held[(size_t) t] = held[(size_t) t] ? (char) F.which[(size_t) t] : (char) 0;
    return R;
}

} // namespace flow
} // namespace dr
