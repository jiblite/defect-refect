#pragma once
// The MOVEMENT faders: what moves in a sound by itself, with no effect on it, and what can be done with that.
//   find()    how a sound's loudness, stereo image, tone and pitch move over time, and its overall colour
//   apply()   a sound with the five faders set
// Every fader runs 0 .. 2: 0 = flat, 1 = the sound as it is (nothing is done, bit for bit), 2 = the other sound's
// movement (two sounds) or the sound's own movement doubled (one sound). In between: straight lines in dB and cents.
//   Pitch      tape speed: the sound is read at a moving place, nothing is stretched or built anew, so nothing
//              warbles and nothing gets dull. The length stays, so the sound keeps its average pitch.
//   Loudness   the level every 5 ms (dips shorter than 80 ms bridged, so a low engine's single pulses read as one
//              even sound while a hit keeps its edges), against the sound's own level.
//   Stereo     where the sound sits (right against left) and how wide it is (side against mid), every 10 ms,
//              around the place it stands at on average. Every moment keeps its power.
//   Tone       the balance of 18 bands (an octave wide, every half octave from 44 Hz) against the moment's average
//              and against its own average over time. Every moment keeps its power.
//   Spectral   the long-term colour in broad strokes (a third of an octave); flat = the same power in every octave.
// The other sound's movement goes on at the same times; past its end its last value holds.
// Order: Pitch first, then Spectral, Tone, Stereo, Loudness. After a pitch change the sound's own curves are carried
// along with it (each moment of the sound keeps its own values), the other sound's stay on the clock.
// A port of the Python prototype Research/movement (move.py, warp.py, flow.py; Tools/move_parity.py compares the
// two). 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "Tone.h"      // dr::Vec, dr::Stereo (Dsp.h), room::Transform, tone::spectra, tone::gainFilter
#include "Flow.h"      // the reader of pitch movement, and the tape-speed reader

namespace dr
{
namespace move
{
using Stop = std::function<bool()>;                     // asked now and then: true = give up (the answer is then not used)
constexpr int STEP = 480;                               // the grid of pitch, stereo and tone: 10 ms
constexpr int LSTEP = 240;                              // the grid of loudness: 5 ms

// ---------------------------------------------------------------- small things, as the prototype has them
inline Vec hann (int n) { Vec w ((size_t) n); for (int i = 0; i < n; ++i) w[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * PI * ((double) i + 0.5) / (double) n); return w; }
inline Vec hannUnit (int n) { Vec w = hann (n); double s = 0; for (double v : w) s += v; for (auto& v : w) v /= s; return w; }      // adding up to one

// A signal of len samples, given by at (i), under the window w every hop samples, the first window centred on
// sample 0 (nothing before the start and after the end): 1 + len / hop sums.
template <typename At> inline Vec windowed (long len, const Vec& w, int hop, At&& at)
{
    const long n = (long) w.size(), T = 1 + len / hop; Vec o ((size_t) T, 0.0);
    for (long t = 0; t < T; ++t) { const long a = t * hop - n / 2; double s = 0; for (long i = std::max (0L, -a); i < n && a + i < len; ++i) s += at (a + i) * w[(size_t) i]; o[(size_t) t] = s; }
    return o;
}
// Evened over 7 values (a Hann bell), the ends held.
inline Vec smooth7 (const Vec& v)
{
    static const Vec w = hannUnit (7); const long n = (long) v.size(); Vec o ((size_t) n);
    for (long i = 0; i < n; ++i) { double s = 0; for (long k = -3; k <= 3; ++k) s += v[(size_t) std::clamp (i + k, 0L, n - 1)] * w[(size_t) (k + 3)]; o[(size_t) i] = s; }
    return o;
}
// The largest (or smallest) of the values up to `half` either side.
inline Vec extreme (const Vec& v, int half, bool largest)
{
    const long n = (long) v.size(); Vec o ((size_t) n);
    for (long i = 0; i < n; ++i)
    {
        double m = v[(size_t) i];
        for (long j = std::max (0L, i - half); j <= std::min (n - 1, i + half); ++j) m = largest ? std::max (m, v[(size_t) j]) : std::min (m, v[(size_t) j]);
        o[(size_t) i] = m;
    }
    return o;
}
// A curve with one value per step, read between its values (numpy.interp on 0, 1, 2 ...; held past the ends).
inline double between (const Vec& v, double t)
{
    const long n = (long) v.size(); if (n == 0) return 0.0; if (t <= 0.0) return v[0]; if (t >= (double) (n - 1)) return v[(size_t) n - 1];
    const long i = (long) t; return (v[(size_t) i + 1] - v[(size_t) i]) * (t - (double) i) + v[(size_t) i];
}
// A curve of another sound at this sound's times: cut, or its last value held.
inline double held (const Vec& v, size_t t) { return v.empty() ? 0.0 : v[std::min (t, v.size() - 1)]; }

// ---------------------------------------------------------------- Loudness
constexpr int BRIDGE = 8;                               // frames either side (40 ms): a dip shorter than 80 ms between two peaks is bridged
constexpr double LIFT_FULL = -24.0, LIFT_NONE = -50.0;  // flat: parts this far under the sound's level are lifted all the way / not at all
struct Loud { Vec db; double ref = -200.0; };           // the level every 5 ms, and the sound's own level: the power average of all moments within 40 dB of the loudest
inline Loud loud (const Stereo& x)
{
    static const Vec w = hannUnit (2 * LSTEP); const long n = (long) x.size();
    const Vec P = windowed (n, w, LSTEP, [&] (long i) { return 0.5 * (x.l[(size_t) i] * x.l[(size_t) i] + x.r[(size_t) i] * x.r[(size_t) i]); });
    const Vec E = smooth7 (extreme (extreme (P, BRIDGE, true), BRIDGE, false)); const double top = *std::max_element (E.begin(), E.end());
    Loud o; o.db.assign (E.size(), -200.0); if (top <= 1e-20) return o;
    double s = 0; long c = 0; for (double e : E) if (e >= top * 1e-4) { s += e; ++c; }
    for (size_t i = 0; i < E.size(); ++i) o.db[i] = 10.0 * std::log10 (std::max (E[i], 1e-20));
    o.ref = 10.0 * std::log10 (s / (double) c); return o;
}
// The gain (dB) that evens a moment o dB off the sound's level; far under the level less and less (the floor is not pulled up).
inline double lift (double o) { return o >= LIFT_FULL ? -o : -LIFT_FULL * std::clamp ((o - LIFT_NONE) / (LIFT_FULL - LIFT_NONE), 0.0, 1.0); }
// Gain in dB. o: the sound's own level against its own average; xo: the other sound's (two: whether there is one).
inline double loudGain (double o, double xo, bool two, double v)
{
    if (v <= 1.0) return (1.0 - v) * lift (o);
    if (! two) return (v - 1.0) * std::clamp (o, -40.0, 12.0);
    return (v - 1.0) * (lift (o) + std::max (xo, -80.0));
}

// ---------------------------------------------------------------- Stereo
constexpr double PAN_MAX = 24.0, WIDTH_MIN = -40.0, WIDTH_MAX = 6.0;
// Where the sound sits and how wide it is, every 10 ms. pan: right against left in dB (+ = right). width: side against
// mid in dB after the two channels were brought to the same level (a source panned to one side has no width of its
// own). pan0, width0: the averages (by power), the place the image stands still at.
struct Space { bool mono = true; Vec pan, width, power; double pan0 = 0.0, width0 = WIDTH_MIN; };
inline Space space (const Stereo& x)
{
    const long n = (long) x.size(), T = 1 + n / STEP; Space S; double apart = 0, pk = 0;
    for (long i = 0; i < n; ++i) { apart = std::max (apart, std::abs (x.l[(size_t) i] - x.r[(size_t) i])); pk = std::max ({ pk, std::abs (x.l[(size_t) i]), std::abs (x.r[(size_t) i]) }); }
    S.pan.assign ((size_t) T, 0.0); S.width.assign ((size_t) T, WIDTH_MIN); S.power.assign ((size_t) T, 0.0);
    if (apart <= 1e-9 * std::max (pk, 1e-30)) return S;
    S.mono = false; static const Vec w = hannUnit (5 * STEP);
    const Vec pl = smooth7 (windowed (n, w, STEP, [&] (long i) { return x.l[(size_t) i] * x.l[(size_t) i]; })), pr = smooth7 (windowed (n, w, STEP, [&] (long i) { return x.r[(size_t) i] * x.r[(size_t) i]; }));
    const Vec c = smooth7 (windowed (n, w, STEP, [&] (long i) { return x.l[(size_t) i] * x.r[(size_t) i]; }));
    double top = 0; for (long t = 0; t < T; ++t) { S.power[(size_t) t] = pl[(size_t) t] + pr[(size_t) t]; top = std::max (top, S.power[(size_t) t]); }
    const double eps = 1e-10 * std::max (top, 1e-30); double sw = 0, sp = 0, swd = 0; std::vector<char> on ((size_t) T, 0); bool any = false;
    for (long t = 0; t < T; ++t)
    {
        const size_t u = (size_t) t; on[u] = S.power[u] > 1e-4 * top; any = any || on[u];
        const double pan = std::clamp (10.0 * std::log10 ((pr[u] + eps) / (pl[u] + eps)), -PAN_MAX, PAN_MAX), a2 = std::pow (10.0, pan / 20.0);       // left times a2, right over a2: both the same
        const double pm = (pl[u] * a2 + pr[u] / a2 + 2.0 * c[u]) / 4.0, ps = (pl[u] * a2 + pr[u] / a2 - 2.0 * c[u]) / 4.0;
        S.pan[u] = pan; S.width[u] = std::clamp (10.0 * std::log10 ((std::max (ps, 0.0) + eps) / (std::max (pm, 0.0) + eps)), WIDTH_MIN, WIDTH_MAX);
        if (on[u]) { sw += S.power[u]; sp += S.power[u] * S.pan[u]; swd += S.power[u] * S.width[u]; }
    }
    if (! any) { std::fill (S.pan.begin(), S.pan.end(), 0.0); std::fill (S.width.begin(), S.width.end(), WIDTH_MIN); return S; }
    S.pan0 = sp / sw; S.width0 = swd / sw;
    for (long t = 0; t < T; ++t) if (! on[(size_t) t]) { S.pan[(size_t) t] = S.pan0; S.width[(size_t) t] = S.width0; }      // a silent moment: the average
    return S;
}
// The place and width the image is to have, from the sound's own (pan, width: at the moments of the sound now sounding).
inline void spaceTargets (const Vec& pan, const Vec& width, const Space& own, const Space* other, double v, Vec& tp, Vec& tw)
{
    const size_t T = pan.size(); tp.resize (T); tw.resize (T);
    for (size_t t = 0; t < T; ++t)
    {
        const double dp = pan[t] - own.pan0, dw = width[t] - own.width0; double p, w;
        if (v <= 1.0 || other == nullptr) { p = own.pan0 + v * dp; w = own.width0 + v * dw; }
        else
        {
            const double xp = held (other->pan, t) - other->pan0, xw = other->mono ? 0.0 : held (other->width, t) - other->width0;
            p = own.pan0 + (2.0 - v) * dp + (v - 1.0) * xp; w = own.width0 + (2.0 - v) * dw + (v - 1.0) * xw;
        }
        tp[t] = std::clamp (p, -PAN_MAX, PAN_MAX); tw[t] = std::clamp (w, WIDTH_MIN, WIDTH_MAX);
    }
}
// y with the image moved from (pan, width) to (tp, tw), all per 10 ms; the power of every moment stays.
inline Stereo spaceApply (const Stereo& y, const Vec& pan, const Vec& width, const Vec& tp, const Vec& tw, const Stop& stop = {})
{
    const long n = (long) y.size(); Stereo o; o.l.resize ((size_t) n); o.r.resize ((size_t) n);
    Vec k (width.size()); for (size_t t = 0; t < k.size(); ++t) k[t] = width[t] > WIDTH_MIN + 5.0 ? std::clamp (tw[t] - width[t], -24.0, 24.0) : 0.0;        // nothing to widen in a sound without sides
    for (long i = 0; i < n; ++i)
    {
        if (stop && (i & 0xffff) == 0 && stop()) return {};
        const double t = (double) i / (double) STEP, a = std::pow (10.0, between (pan, t) / 40.0), l = y.l[(size_t) i] * a, r = y.r[(size_t) i] / a;
        const double m = 0.5 * (l + r), s = 0.5 * (l - r) * std::pow (10.0, between (k, t) / 20.0), b = std::pow (10.0, between (tp, t) / 40.0);
        o.l[(size_t) i] = (m + s) / b; o.r[(size_t) i] = (m - s) * b;
    }
    static const Vec w = hannUnit (5 * STEP);
    const Vec p0 = windowed (n, w, STEP, [&] (long i) { return y.l[(size_t) i] * y.l[(size_t) i] + y.r[(size_t) i] * y.r[(size_t) i]; });
    const Vec p1 = windowed (n, w, STEP, [&] (long i) { return o.l[(size_t) i] * o.l[(size_t) i] + o.r[(size_t) i] * o.r[(size_t) i]; });
    Vec g (p0.size()); for (size_t t = 0; t < g.size(); ++t) g[t] = std::sqrt ((p0[t] + 1e-20) / (p1[t] + 1e-20));
    for (long i = 0; i < n; ++i) { const double gg = std::clamp (between (g, (double) i / (double) STEP), 0.25, 4.0); o.l[(size_t) i] *= gg; o.r[(size_t) i] *= gg; }
    return o;
}

// ---------------------------------------------------------------- Tone
constexpr int NT = 2048, NTB = NT / 2 + 1, NBAND = 18; constexpr double TONE_MAX = 18.0;
// The bands on the bins of a frame of NT: centres 44 Hz x 2^(b/2), up to 15.9 kHz, each an octave wide with a raised
// cosine's flanks; the ends run on flat; no DC. [band][bin]
inline const Vec& bandWeights()
{
    static const Vec W = []
    {
        Vec w ((size_t) NBAND * (size_t) NTB);
        for (int k = 0; k < NTB; ++k)
        {
            const double u = std::log2 (std::max ((double) k * SR / (double) NT, 1.0) / 44.0) * 2.0;
            for (int b = 0; b < NBAND; ++b)
            {
                const double c = std::cos (0.5 * PI * std::clamp (u - (double) b, -1.0, 1.0)); double v = c * c;
                if (b == 0 && u < 0.0) v = 1.0; if (b == NBAND - 1 && u > (double) (NBAND - 1)) v = 1.0; if (k == 0) v = 0.0;
                w[(size_t) b * (size_t) NTB + (size_t) k] = v;
            }
        }
        return w;
    }();
    return W;
}
// For each of T frames (NT long, every STEP, the first centred on sample 0) the frame whose reading counts. A frame
// that lies across a cut (all: flow::cutsOf, the file's ends among them) holds a sound that stops inside it, and that
// reads as a click: 18 dB and more in the bands where the sound has least, over the 30 ms either side of every cut,
// however steady the sound is. Such a frame takes the reading of the nearest frame that lies whole on the sounding
// side of the cut, so the balance stands still there; -1 where there is no such frame (a piece of sound shorter than
// a frame).
inline std::vector<long> wholeFrames (const std::vector<flow::Cut>& all, long T)
{
    std::vector<long> src ((size_t) T); for (long t = 0; t < T; ++t) src[(size_t) t] = t;
    auto down = [] (long v) { return v >= 0 ? v / STEP : -((-v + STEP - 1) / STEP); };       // v / STEP rounded down
    struct Zone { long a, b, hold; }; std::vector<Zone> zones; std::vector<char> spoiled ((size_t) T, 0);
    for (const flow::Cut& c : all)
    {
        const long lo = down (c.pos - NT / 2), hi = -down (-(c.pos + NT / 2));         // the last frame that ends at or before the cut, the first that starts at or after it
        zones.push_back ({ lo + 1, hi - 1, c.dir > 0 ? hi : lo });
    }
    for (const Zone& z : zones) for (long t = std::max (0L, z.a); t <= std::min (T - 1, z.b); ++t) spoiled[(size_t) t] = 1;
    for (const Zone& z : zones)
    {
        const long from = z.hold >= 0 && z.hold < T && ! spoiled[(size_t) z.hold] ? z.hold : -1;
        for (long t = std::max (0L, z.a); t <= std::min (T - 1, z.b); ++t) src[(size_t) t] = from;
    }
    return src;
}
// How the balance of the bands moves, dB per band every 10 ms: each band's level against the moment's average over
// the bands that carry the sound (weights w: 1 for a band within 30 dB of the sound's strongest band, 0 from 50 dB
// under), and that against its own average over time. Level-free, and it adds up: a gain per band changes it by
// exactly that gain (less the gains' weighted average). A quiet moment (60 to 40 dB under the loudest) counts less.
// Beside a hard cut the balance stands still (wholeFrames).
struct ToneMove
{
    long T = 0; Vec m, w;                               // [T][NBAND], and the bands' weights
    const double* row (long t) const { return m.data() + (size_t) t * (size_t) NBAND; }
};
inline ToneMove toneMove (const Stereo& x, const Stop& stop = {})
{
    const long n = (long) x.size(), T = 1 + n / STEP; static const Vec win = hann (NT); const Vec& W = bandWeights();
    ToneMove o; o.T = T; o.m.assign ((size_t) T * (size_t) NBAND, 0.0); o.w.assign ((size_t) NBAND, 0.0);
    const std::vector<long> src = wholeFrames (flow::cutsOf (n, [&x] (long i) { return 0.5 * (x.l[(size_t) i] * x.l[(size_t) i] + x.r[(size_t) i] * x.r[(size_t) i]); }), T);
    const room::Transform tr (NT); Vec re ((size_t) NT), im ((size_t) NT), P ((size_t) NTB), E ((size_t) T * (size_t) NBAND, 0.0), tot ((size_t) T, 0.0);
    for (long t = 0; t < T; ++t)
    {
        if (stop && (t & 255) == 0 && stop()) { o.T = -1; return o; }
        if (src[(size_t) t] != t) continue;             // a frame across a cut is not read
        const long a = t * STEP - NT / 2;
        for (long i = 0; i < NT; ++i) { const long j = a + i; const bool in = j >= 0 && j < n; re[(size_t) i] = in ? x.l[(size_t) j] * win[(size_t) i] : 0.0; im[(size_t) i] = in ? x.r[(size_t) j] * win[(size_t) i] : 0.0; }
        tr.run (re, im);        // left as the real part and right as the imaginary part: one transform for both
        for (int k = 0; k < NTB; ++k) { const size_t u = (size_t) k, m = (size_t) ((NT - k) & (NT - 1)); P[u] = 0.5 * (re[u] * re[u] + im[u] * im[u] + re[m] * re[m] + im[m] * im[m]); }
        double s = 0;
        for (int b = 0; b < NBAND; ++b) { const double* wb = W.data() + (size_t) b * (size_t) NTB; double e = 0; for (int k = 0; k < NTB; ++k) e += P[(size_t) k] * wb[k]; E[(size_t) t * (size_t) NBAND + (size_t) b] = e; s += e; }
        tot[(size_t) t] = s;
    }
    const double top = *std::max_element (tot.begin(), tot.end()); if (top <= 1e-20) return o;
    Vec lt ((size_t) NBAND, 0.0); long cnt = 0;
    for (long t = 0; t < T; ++t) if (tot[(size_t) t] > 1e-4 * top) { ++cnt; for (int b = 0; b < NBAND; ++b) lt[(size_t) b] += E[(size_t) t * (size_t) NBAND + (size_t) b]; }
    double ltop = -1e300; for (auto& v : lt) { v = 10.0 * std::log10 (std::max (v / (double) cnt, 1e-30)); ltop = std::max (ltop, v); }
    double sw = 0; for (int b = 0; b < NBAND; ++b) { o.w[(size_t) b] = std::clamp ((lt[(size_t) b] - (ltop - 50.0)) / 20.0, 0.0, 1.0); sw += o.w[(size_t) b]; }
    Vec m ((size_t) T * (size_t) NBAND), avg ((size_t) NBAND, 0.0); double swt = 0;
    for (long t = 0; t < T; ++t)
    {
        double* r = m.data() + (size_t) t * (size_t) NBAND; double a = 0;
        for (int b = 0; b < NBAND; ++b) { r[b] = 10.0 * std::log10 (std::max (E[(size_t) t * (size_t) NBAND + (size_t) b], 1e-9 * top)); a += r[b] * o.w[(size_t) b]; }
        a /= sw; for (int b = 0; b < NBAND; ++b) r[b] -= a;
        if (tot[(size_t) t] > 1e-4 * top) { swt += tot[(size_t) t]; for (int b = 0; b < NBAND; ++b) avg[(size_t) b] += tot[(size_t) t] * r[b]; }
    }
    for (auto& v : avg) v /= swt;
    for (long t = 0; t < T; ++t)        // a silent frame says nothing, also not to its neighbours
    {
        const double quiet = std::clamp ((10.0 * std::log10 (std::max (tot[(size_t) t], 1e-30) / top) + 60.0) / 20.0, 0.0, 1.0);
        for (int b = 0; b < NBAND; ++b) { double& v = m[(size_t) t * (size_t) NBAND + (size_t) b]; v = (v - avg[(size_t) b]) * quiet; }
    }
    for (long t = 0; t < T; ++t)        // across a cut: the reading of the whole frame beside it
    {
        const long from = src[(size_t) t]; if (from == t) continue;
        for (int b = 0; b < NBAND; ++b) m[(size_t) t * (size_t) NBAND + (size_t) b] = from >= 0 ? m[(size_t) from * (size_t) NBAND + (size_t) b] : 0.0;
    }
    for (long t = 0; t < T; ++t)
    {
        const double* p = m.data() + (size_t) std::max (t - 1, 0L) * (size_t) NBAND; const double* c = m.data() + (size_t) t * (size_t) NBAND; const double* q = m.data() + (size_t) std::min (t + 1, T - 1) * (size_t) NBAND;
        for (int b = 0; b < NBAND; ++b) o.m[(size_t) t * (size_t) NBAND + (size_t) b] = std::clamp (0.25 * p[b] + 0.5 * c[b] + 0.25 * q[b], -TONE_MAX, TONE_MAX);
    }
    return o;
}
// Gain in dB per band and 10 ms. m: the sound's own movement (at the moments now sounding), w: its bands' weights;
// other: the other sound's (null: one sound). A band the sound hardly has is left alone; a band the other sound hardly
// has counts as standing still.
inline Vec toneGain (const Vec& m, const Vec& w, const ToneMove* other, double v)
{
    const size_t T = m.size() / (size_t) NBAND; Vec g (m.size()); const bool two = v > 1.0 && other != nullptr && other->T > 0;
    for (size_t t = 0; t < T; ++t)
    {
        const double* xr = two ? other->row ((long) std::min (t, (size_t) other->T - 1)) : nullptr;
        for (size_t b = 0; b < (size_t) NBAND; ++b)
        {
            const double own = m[t * (size_t) NBAND + b], d = two ? (v - 1.0) * (xr[b] * other->w[b] - own) : (v - 1.0) * own;
            g[t * (size_t) NBAND + b] = std::clamp (d, -TONE_MAX, TONE_MAX) * w[b];
        }
    }
    return g;
}
// y with the band gains g ([T][NBAND], dB per 10 ms); every moment keeps its power.
inline Stereo toneApply (const Stereo& y, const Vec& g, const Stop& stop = {})
{
    const long n = (long) y.size(), hop = NT / 4, J = 1 + (n + NT) / hop, T = (long) (g.size() / (size_t) NBAND); if (T < 1) return y;
    static const Vec win = [] { Vec w = hann (NT); for (auto& v : w) v = std::sqrt (v); return w; }();
    static const Vec Wn = [] { Vec w = bandWeights(); for (int k = 0; k < NTB; ++k) { double s = 0; for (int b = 0; b < NBAND; ++b) s += w[(size_t) b * (size_t) NTB + (size_t) k]; s = std::max (s, 1e-9); for (int b = 0; b < NBAND; ++b) w[(size_t) b * (size_t) NTB + (size_t) k] /= s; } return w; }();
    Stereo o; o.l.assign ((size_t) n, 0.0); o.r.assign ((size_t) n, 0.0);
    const room::Transform tr (NT); Vec re ((size_t) NT), im ((size_t) NT), gb ((size_t) NBAND), G ((size_t) NTB);
    for (long j = 0; j < J; ++j)
    {
        if (stop && (j & 255) == 0 && stop()) return {};
        const long a = j * hop - NT;                    // the frame's first sample; its middle sits at a + NT / 2
        for (long i = 0; i < NT; ++i) { const long q = a + i; const bool in = q >= 0 && q < n; re[(size_t) i] = in ? y.l[(size_t) q] * win[(size_t) i] : 0.0; im[(size_t) i] = in ? y.r[(size_t) q] * win[(size_t) i] : 0.0; }
        tr.run (re, im);
        const double tt = std::clamp ((double) (a + NT / 2) / (double) STEP, 0.0, (double) (T - 1)); const long i0 = (long) std::floor (tt), i1 = std::min (i0 + 1, T - 1); const double fr = tt - (double) i0;
        for (int b = 0; b < NBAND; ++b) gb[(size_t) b] = g[(size_t) i0 * (size_t) NBAND + (size_t) b] * (1.0 - fr) + g[(size_t) i1 * (size_t) NBAND + (size_t) b] * fr;
        for (int k = 0; k < NTB; ++k) { double s = 0; for (int b = 0; b < NBAND; ++b) s += gb[(size_t) b] * Wn[(size_t) b * (size_t) NTB + (size_t) k]; G[(size_t) k] = std::pow (10.0, s / 20.0); }
        double p0 = 0, p1 = 0;                          // the power on the bins 0 .. NT / 2 of both channels, before and after
        for (int k = 0; k < NTB; ++k)
        {
            const size_t u = (size_t) k, m = (size_t) ((NT - k) & (NT - 1)); const double e = 0.5 * (re[u] * re[u] + im[u] * im[u] + re[m] * re[m] + im[m] * im[m]);
            p0 += e; p1 += e * G[u] * G[u];
        }
        const double keep = std::sqrt ((p0 + 1e-30) / (p1 + 1e-30));
        for (int k = 0; k < NTB; ++k) { const size_t u = (size_t) k, m = (size_t) ((NT - k) & (NT - 1)); const double f = G[u] * keep; re[u] *= f; im[u] *= f; if (m != u) { re[m] *= f; im[m] *= f; } }
        tr.back (re, im);
        for (long i = 0; i < NT; ++i) { const long q = a + i; if (q >= 0 && q < n) { o.l[(size_t) q] += re[(size_t) i] * win[(size_t) i] * 0.5; o.r[(size_t) q] += im[(size_t) i] * win[(size_t) i] * 0.5; } }
    }
    return o;
}

// ---------------------------------------------------------------- Spectral
constexpr double COLOUR_OCT = 1.0 / 6.0, COLOUR_HZ = 40.0;      // a colour: the long-term power evened over a third of an octave, and over no less than about 160 Hz
// The neutral colour on the tone engine's bins: the same power in every octave.
inline const Vec& neutral()
{
    static const Vec N = [] { Vec v ((size_t) tone::NB); for (int k = 0; k < tone::NB; ++k) v[(size_t) k] = -10.0 * std::log10 (std::clamp (std::max ((double) k, 1.0) * tone::binHz, 20.0, 20000.0)); return v; }();
    return N;
}
// A curve on the bins evened by a bell sigmaOct octaves wide (at least sigmaHz): three passes of a running mean.
inline Vec even (Vec v, double sigmaOct = COLOUR_OCT, double sigmaHz = COLOUR_HZ)
{
    const int n = (int) v.size(); if (n == 0) return v; Vec cs ((size_t) n + 1), o ((size_t) n);
    auto at = [&] (double q) { const double u = q + 0.5; const int i = std::clamp ((int) std::floor (u), 0, n - 1); return cs[(size_t) i] + (cs[(size_t) i + 1] - cs[(size_t) i]) * (u - (double) i); };
    for (int pass = 0; pass < 3; ++pass)
    {
        cs[0] = 0.0; for (int i = 0; i < n; ++i) cs[(size_t) i + 1] = cs[(size_t) i] + v[(size_t) i];
        for (int k = 0; k < n; ++k)
        {
            const double sig = std::max (sigmaOct * std::log (2.0) * (double) k, sigmaHz / tone::binHz), h = 0.5 * (std::sqrt (4.0 * sig * sig + 1.0) - 1.0);
            const double a = std::clamp ((double) k - h - 0.5, -0.5, (double) n - 0.5), b = std::clamp ((double) k + h + 0.5, -0.5, (double) n - 0.5);
            o[(size_t) k] = (at (b) - at (a)) / std::max (b - a, 1e-9);
        }
        v = o;
    }
    return v;
}
// A long-term spectrum (dB on the bins) in broad strokes: its power evened (not its dB: the gaps between the partials
// of a sound with few of them do not count as dark).
inline Vec colour (const Vec& S)
{
    const double mx = *std::max_element (S.begin(), S.end()); Vec p (S.size()); for (size_t k = 0; k < S.size(); ++k) p[k] = std::pow (10.0, (S[k] - mx) / 10.0);
    p = even (std::move (p)); for (auto& v : p) v = 10.0 * std::log10 (std::max (v, 1e-30)) + mx;
    return p;
}
// The fixed curve (dB on the bins). own, other: long-term spectra (other null: one sound). 0: the sound's colour brought
// to the neutral one; 2: to the other sound's colour, or (one sound) its own colour once more. Taken where the sound
// has something (within 60 dB of its strongest region; outside, the nearest value holds); at most +-30 dB, bent in
// softly as every tone curve of the app; the sound keeps its long-term power.
inline Vec specCurve (const Vec& own, const Vec* other, double v)
{
    const int NB = tone::NB; Vec c ((size_t) NB, 0.0); if (own.size() != (size_t) NB || v == 1.0 || *std::max_element (own.begin(), own.end()) < -250.0) return c;
    const bool two = v > 1.0 && other != nullptr; if (two && (other->size() != (size_t) NB || *std::max_element (other->begin(), other->end()) < -250.0)) return c;
    const Vec co = colour (own), ct = two ? colour (*other) : neutral(); const double mo = *std::max_element (co.begin(), co.end()), mt = *std::max_element (ct.begin(), ct.end());
    std::vector<int> ok; for (int k = 0; k < NB; ++k) if (co[(size_t) k] > mo - 60.0 && (double) k * tone::binHz >= 20.0 && (! two || ct[(size_t) k] > mt - 60.0)) ok.push_back (k);
    if (ok.empty()) return c;
    auto d = [&] (int k) { return ct[(size_t) k] - co[(size_t) k]; }; size_t j = 0;
    for (int k = 0; k < NB; ++k)
    {
        if (k <= ok.front()) c[(size_t) k] = d (ok.front());
        else if (k >= ok.back()) c[(size_t) k] = d (ok.back());
        else { while (ok[j + 1] < k) ++j; const int a = ok[j], b = ok[j + 1]; c[(size_t) k] = d (a) + (d (b) - d (a)) / (double) (b - a) * (double) (k - a); }
    }
    c = even (std::move (c)); Vec vals; vals.reserve (ok.size()); for (int k : ok) vals.push_back (c[(size_t) k]);
    const double med = median (vals), f = v <= 1.0 ? 1.0 - v : two ? v - 1.0 : -(v - 1.0);
    for (auto& g : c) g = std::clamp (g - med, -30.0, 30.0) * f;
    const double mx = *std::max_element (own.begin(), own.end()); Vec p ((size_t) NB); double den = 0;
    for (size_t k = 0; k < (size_t) NB; ++k) { p[k] = std::pow (10.0, (own[k] - mx) / 10.0); den += p[k]; }
    for (int it = 0; it < 3; ++it)
    {
        double num = 0; for (size_t k = 0; k < (size_t) NB; ++k) num += p[k] * std::pow (10.0, tone::softCap (c[k]) / 10.0);
        const double off = 10.0 * std::log10 (std::max (num / den, 1e-12)); for (auto& g : c) g -= off;
    }
    return c;
}

// ---------------------------------------------------------------- Pitch: the tape-speed change
// x read at the places pos (in samples, rising). Where the places move faster than one sample per sample (pitch up)
// the reader's cutoff comes down with them, so nothing folds back. Empty when given up.
inline Stereo readAt (const Stereo& x, const Vec& pos, const Stop& stop = {})
{
    const long n = (long) x.size(), m = (long) pos.size(); Stereo o; o.l.assign ((size_t) m, 0.0); o.r.assign ((size_t) m, 0.0);
    for (long j = 0; j < m; ++j)
    {
        if (stop && (j & 0xffff) == 0 && stop()) return {};
        double sl = 0, sr = 0; flow::readerTaps (pos[(size_t) j], flow::stepAt (pos, j), n, [&] (long k, double w) { sl += x.l[(size_t) k] * w; sr += x.r[(size_t) k] * w; });
        o.l[(size_t) j] = sl; o.r[(size_t) j] = sr;
    }
    return o;
}
// Where in the sound each moment of the result is read. o: the sound's own pitch movement, cents per 10 ms (0 where
// none is followed); xo: the other sound's or null. The change is a times the sound's own movement (at the sound's
// moments) plus b times the other's (on the clock). The length stays: the sound keeps its average pitch.
inline Vec pitchPositions (const Vec& o, const Vec* xo, double v, long n)
{
    const bool two = v > 1.0 && xo != nullptr; const double a = two ? 1.0 - v : v - 1.0, b = two ? v - 1.0 : 0.0; Vec pos ((size_t) std::max (n, 0L)); if (n < 1) return pos;
    Vec xc; if (two) { xc.resize (o.size()); for (size_t t = 0; t < xc.size(); ++t) xc[t] = held (*xo, t); }                         // the other's curve at this sound's frames: cut, or its last value held
    auto ownStep = [&] (long i) { return std::pow (2.0, -a * between (o, (double) i / (double) STEP) / 1200.0); };                  // the time that sample i of the sound takes
    auto clock = [&] (long i) { return std::pow (2.0, b * between (xc, (double) i / (double) STEP) / 1200.0); };                    // how fast the clock's moment i runs
    double A = 0, B = 0; for (long i = 0; i < n; ++i) { A += ownStep (i); B += two ? clock (i) : 1.0; }
    const double scale = A / B;
    // both running sums once more, side by side: the place in the sound whose time is the clock's
    long i = 0; double a0 = 0.0, a1 = ownStep (0), bj = 0.0;
    for (long j = 0; j < n; ++j)
    {
        const double want = (two ? bj : (double) j) * scale;
        while (i < n - 1 && a1 <= want) { ++i; a0 = a1; a1 = a0 + ownStep (i); }
        pos[(size_t) j] = want >= a1 ? (double) n : (1.0 / (a1 - a0)) * (want - a0) + (double) i;
        if (two) bj += clock (j);
    }
    return pos;
}

// ---------------------------------------------------------------- all of it
struct Pitch { Vec cents; std::vector<char> held; bool stopped = false; };       // every 10 ms: the movement in cents around the sound's own average (0 where nothing is followed), and which look follows it there (0: none)
constexpr long PITCH_MOST = 600L * 48000L;              // a sound longer than this (ten minutes) is not read: nothing is followed (the reader takes about a fifth of the sound's length and holds the sound twice more)
inline Pitch track (const Stereo& x, const Stop& stop = {}, const flow::Log& log = {})
{
    Pitch P;
    if ((long) x.size() > PITCH_MOST) { P.cents.assign (1 + x.size() / (size_t) STEP, 0.0); P.held.assign (P.cents.size(), 0); return P; }
    flow::Track t = flow::track (x.mid(), stop, log); P.stopped = t.stopped; P.cents = std::move (t.cents); P.held = std::move (t.held); return P;
}
struct Found { long n = 0; Loud loud; Space space; ToneMove tone; Vec spec; Pitch pitch; bool stopped = false; };       // spec: the long-term spectrum (dB on the tone engine's bins)
struct Faders
{
    double pitch = 1.0, loudness = 1.0, tone = 1.0, stereo = 1.0, spectral = 1.0;
    bool any() const { return pitch != 1.0 || loudness != 1.0 || tone != 1.0 || stereo != 1.0 || spectral != 1.0; }
};

// What moves in a sound, all but its pitch (see track() for that).
inline Found findPlain (const Stereo& x, const Stop& stop = {})
{
    Found F; F.n = (long) x.size(); F.loud = loud (x); F.space = space (x);
    F.tone = toneMove (x, stop); if (F.tone.T < 0) { F.stopped = true; return F; }
    const tone::Spectra S = tone::spectra (x, stop); if (stop && stop()) { F.stopped = true; return F; }
    F.spec = S.silent ? Vec ((size_t) tone::NB, -300.0) : S.both;
    return F;
}

// The sound's own curve (rows of `cols` values every `step` samples) at the moments now sounding, after its pitch was changed.
inline Vec mine (const Vec& curve, int cols, int step, const Vec& pos, long n)
{
    if (pos.empty() || curve.empty() || n < 1) return curve; const long T = (long) (curve.size() / (size_t) cols); Vec o (curve.size());
    for (long t = 0; t < T; ++t)
    {
        const long k = std::min (t * step, n - 1), j = std::clamp ((long) std::nearbyint (pos[(size_t) k] / (double) step), 0L, T - 1);
        std::copy (curve.begin() + j * cols, curve.begin() + (j + 1) * cols, o.begin() + t * cols);
    }
    return o;
}
inline bool moves (const Vec& v) { for (double a : v) if (a != 0.0) return true; return false; }

// x with the faders set. own: what was found on x; other: on the other sound (null: one sound). Empty when given up.
inline Stereo apply (const Stereo& x, const Found& own, const Found* other, const Faders& f, const Stop& stop = {})
{
    Stereo y = x; const long n = (long) y.size(); if (n == 0 || ! f.any()) return y; Vec pos;
    if (f.pitch != 1.0)
    {
        const Vec* xo = (other != nullptr && f.pitch > 1.0) ? &other->pitch.cents : nullptr;
        if (! own.pitch.cents.empty() && (moves (own.pitch.cents) || (xo != nullptr && moves (*xo))))
        {
            pos = pitchPositions (own.pitch.cents, xo, f.pitch, n); y = readAt (y, pos, stop); if (y.size() == 0) return {};
        }
    }
    if (f.spectral != 1.0)
    {
        const Vec c = specCurve (own.spec, other != nullptr ? &other->spec : nullptr, f.spectral);
        if (! tone::nothing (c)) { y = tone::gainFilter (y, c, c, stop); if (y.size() == 0) return {}; }
    }
    if (f.tone != 1.0 && own.tone.T > 0)
    {
        const Vec g = toneGain (mine (own.tone.m, NBAND, STEP, pos, n), own.tone.w, other != nullptr ? &other->tone : nullptr, f.tone);
        if (moves (g)) { y = toneApply (y, g, stop); if (y.size() == 0) return {}; }
    }
    if (f.stereo != 1.0 && ! (own.space.mono && (other == nullptr || f.stereo <= 1.0 || other->space.mono)))
    {
        const Vec pan = mine (own.space.pan, 1, STEP, pos, n), width = mine (own.space.width, 1, STEP, pos, n); Vec tp, tw;
        spaceTargets (pan, width, own.space, other != nullptr ? &other->space : nullptr, f.stereo, tp, tw);
        y = spaceApply (y, pan, width, tp, tw, stop); if (y.size() == 0) return {};
    }
    if (f.loudness != 1.0)
    {
        Vec g = mine (own.loud.db, 1, LSTEP, pos, n);
        for (size_t t = 0; t < g.size(); ++t) g[t] = loudGain (g[t] - own.loud.ref, other != nullptr ? held (other->loud.db, t) - other->loud.ref : 0.0, other != nullptr, f.loudness);
        for (long i = 0; i < n; ++i) { const double gg = std::pow (10.0, between (g, (double) i / (double) LSTEP) / 20.0); y.l[(size_t) i] *= gg; y.r[(size_t) i] *= gg; }
    }
    return y;
}

} // namespace move
} // namespace dr
