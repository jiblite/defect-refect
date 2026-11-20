#pragma once
// The bend engine (DISTORTION, SATURATION): a fixed curve that a sound's values were pushed through.
//   features()  what the meter sees of a sound, one half-wave at a time (the values above zero, or those under):
//               frames of 2048 samples sorted into ten classes by their peak against the half-wave's own; per class
//               how the frame's values lie under its peak (eleven places in their order), how steep the wave is at six
//               heights against the frame's own steepness, and how many frames the class has; and how much of the
//               half-wave sits within 0.1, 0.5, 1 and 2 dB of its peak
//   Meter       a few small nets taught with sounds through curves of many shapes (Research/bend): per half-wave, by
//               how many dB the curve squeezed each of eleven levels of the sound as it was (30 dB under its peak up
//               to the peak) against its plain gain for small values; and whether a bend is there to be heard. The
//               nets are taught alike from different starts and their answers averaged: on a real bend they agree, on
//               a chance find they do not
//   find()      the meter's answer for one sound
//   addFlat()   a sound whose tops were cut flat: the cut-off tops are rebuilt from the sound around them (the
//               declipper: "flat tops" below, and Declip.h). A plain cut is found by this alone, whatever the meter says
//   own()       the sound itself with the group's Amount at s: 0 = the bend walked back (three quarters of what was
//               read, less 1 dB: careful(); flat tops as rebuilt), 1 = untouched (bit for bit), 2 = pushed as far again
//   onto()      another sound through the found curve: 0 = none, 1 = its peak pushed as far into the curve as the
//               first sound's was (a plain cut whose tops were rebuilt: as far past the cut as the rebuilt peak
//               stands), 2 = as far again
// The curve is not one of a few textbook shapes: it is the eleven numbers per half-wave, joined smoothly, so a tube's
// uneven halves, a tape's soft shoulder and a clipper's corner are each what they were read as.
// The way back spreads values apart again. Where the curve pressed many levels into one (flat tops), nothing in a
// value says how high it stood: along the meter's curve alone such values are spread only as far as spreadMost allows.
// The sound around a flat top does say it, and from that the declipper rebuilds the tops (flat tops, below).
// What it does not do: a box whose bend is hidden behind a strong filter or a room (the wave no longer shows it); a
// limiter or a compressor (a level held down by a moving gain leaves the wave's shape alone: the level engine's work).
#include "Tone.h"
#include "Declip.h"
#include "Resample.h"
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace dr
{
namespace bend
{
using Stop = room::Stop;
constexpr int FR = 2048, HOPF = 1024, NAL = 12, NLEV = 10, NVAL = 5, NG = 11;
constexpr int NF = (NAL - 1) * NLEV + NLEV + (NVAL + 1) * NLEV + 4;         // 184 numbers per half-wave
constexpr int PLACE[NAL] = { 1535, 1638, 1740, 1842, 1904, 1945, 1975, 2006, 2027, 2037, 2045, 2047 };     // places in a frame's values in rising order (75 % .. the largest)
constexpr double LEV[NLEV] = { -48.0, -36.0, -27.0, -20.0, -14.0, -9.0, -5.0, -2.5, -1.0, -0.3 };            // the frame classes: a frame's peak against the half-wave's (dB), from
constexpr double VAL[NVAL] = { 0.2, 0.4, 0.6, 0.8, 0.9 };                                                    // the heights: a value against its frame's peak
constexpr double GRID[NG] = { -30.0, -26.0, -22.0, -18.0, -15.0, -12.0, -9.0, -6.0, -4.0, -2.0, 0.0 };        // where the answer is given: levels of the sound as it was against its peak (dB)
constexpr double SMALL = -34.0;                 // values this far under the peak and further pass with the curve's plain gain
constexpr double maxSqueeze = 40.0;             // no answer is used beyond this many dB
constexpr double spreadMost = 6.0;              // the way back spreads values at most this many times as far apart as they lie in the sound (in dB)
constexpr size_t minLen = 24000;                // half a second: under that nothing is read
constexpr double backPart = 0.75, backLess = 1.0;        // the way back takes this part of the read squeeze, less this many dB (see careful())
constexpr double sureAt = -2.0, sureTop = 5.0;  // the meter's verdict (a logit) and the squeeze it reads at the peak (dB) from which a bend is shown with the Sureness slider where it stands at first: both must be there
constexpr double lowAt = -6.0, topAt = 3.0;     // ... and from which it is kept at all: such a bend shows when the slider is turned down
constexpr double hardAt = 10.0;                 // squeezed by this much at the peak the group is called DISTORTION, under it SATURATION
constexpr double shownSure = 0.65;              // what the Sureness slider cuts at by default: a bend just found shows this

// A sound's peak without its few largest values (1 in 10000). pol 0: of both half-waves; +1 or -1: of that half-wave
// alone (the other half counts as zeros).
inline double peakOf (const Stereo& y, double pol)
{
    const size_t n = y.size(); if (n == 0) return 0.0;
    std::vector<double> a; a.reserve (2 * n);
    for (const Vec* v : { &y.l, &y.r }) for (size_t i = 0; i < n; ++i) a.push_back (pol == 0.0 ? std::abs ((*v)[i]) : std::max (pol * (*v)[i], 0.0));
    const size_t k = std::max<size_t> (1, (size_t) std::floor ((double) a.size() * 1e-4 + 0.5)), at = a.size() - std::min (k, a.size());
    std::nth_element (a.begin(), a.begin() + (std::ptrdiff_t) at, a.end()); return a[at];
}

// The level of a result o made from the sound y. The sound's peak (peakOf: without its few largest values) stays
// where it was, and no value comes out higher than the sound's highest was: rebuilt tops and values that were spread
// apart stand above that peak, and a file that was cut at full scale would come out over full scale with them, where
// the app's limiter then presses the very tops down that were rebuilt.
inline void sameLevel (Stereo& o, const Stereo& y)
{
    const double k = std::min (peakOf (y, 0.0) / std::max (peakOf (o, 0.0), 1e-30), y.peak() / std::max (o.peak(), 1e-30));
    for (Vec* ch : { &o.l, &o.r }) for (double& v : *ch) v *= k;
}

// What the meter sees of one half-wave: NF numbers, and the half-wave's peak. False when given up.
inline bool features (const Stereo& y, double pol, float* out, double& peak, const Stop& stop = {})
{
    const double vref = std::max (peakOf (y, pol), 1e-9); peak = vref;
    double qAcc[NLEV][NAL - 1] = {}, qN[NLEV] = {}, sAcc[NLEV][NVAL + 1] = {}, sN[NLEV][NVAL + 1] = {}, near[4] = {}; double npos = 0;
    const double nearAt[4] = { vref * std::pow (10.0, -0.1 / 20.0), vref * std::pow (10.0, -0.5 / 20.0), vref * std::pow (10.0, -1.0 / 20.0), vref * std::pow (10.0, -2.0 / 20.0) };
    const long n = (long) y.size(); std::vector<double> z ((size_t) std::max (n, 0L)), d2 (z.size()), srt ((size_t) FR);
    for (const Vec* ch : { &y.l, &y.r })
    {
        for (long i = 0; i < n; ++i) z[(size_t) i] = std::max (pol * (*ch)[(size_t) i], 0.0);
        for (long i = 1; i + 1 < n; ++i) { const double d = 0.5 * pol * ((*ch)[(size_t) (i + 1)] - (*ch)[(size_t) (i - 1)]); d2[(size_t) i] = d * d; }       // how steep the wave is at each value
        if (n > 0) d2[0] = d2[(size_t) (n - 1)] = 0.0;
        const long frames = n >= FR ? (n - FR) / HOPF + 1 : 0;
        for (long t = 0; t < frames; ++t)
        {
            if (stop && (t & 63) == 0 && stop()) return false;
            const double* f = z.data() + (size_t) (t * HOPF); const double* fd = d2.data() + (size_t) (t * HOPF);
            std::copy (f, f + FR, srt.begin()); std::nth_element (srt.begin(), srt.begin() + PLACE[0], srt.end()); std::sort (srt.begin() + PLACE[0], srt.end());
            const double p = srt[(size_t) PLACE[NAL - 1]], lev = 20.0 * std::log10 (std::max (p, 1e-12) / vref); if (! (lev > LEV[0])) continue;
            int b = 0; while (b + 1 < NLEV && lev >= LEV[b + 1]) ++b;
            for (int a = 0; a + 1 < NAL; ++a) qAcc[b][a] += std::max (20.0 * std::log10 (std::max (srt[(size_t) PLACE[a]], 1e-12) / std::max (p, 1e-12)), -60.0);
            qN[b] += 1.0;
            double sum[NVAL + 1] = {}, tot = 0; long cnt[NVAL + 1] = {}, pos = 0; const double pp = std::max (p, 1e-12);
            for (int i = 0; i < FR; ++i)
            {
                if (! (f[i] > 0.0)) continue;
                const double r = f[i] / pp; int k = 0; while (k < NVAL && r >= VAL[k]) ++k;
                sum[k] += fd[i]; ++cnt[k]; tot += fd[i]; ++pos;
            }
            tot /= (double) std::max (pos, 1L);
            if (tot > 0.0) for (int k = 0; k <= NVAL; ++k) if (cnt[k] >= 8)
            {
                sAcc[b][k] += std::clamp (10.0 * std::log10 (std::max (sum[k] / (double) cnt[k], 1e-30) / std::max (tot, 1e-30)), -40.0, 20.0); sN[b][k] += 1.0;
            }
        }
        for (long i = 0; i < n; ++i) { const double v = z[(size_t) i]; if (v > 0.0) { npos += 1.0; for (int k = 0; k < 4; ++k) if (v > nearAt[k]) near[k] += 1.0; } }
    }
    double all = 0; for (int b = 0; b < NLEV; ++b) all += qN[b];
    int o = 0;
    for (int b = 0; b < NLEV; ++b) for (int a = 0; a + 1 < NAL; ++a) out[o++] = (float) (qAcc[b][a] / std::max (qN[b], 1.0) / 20.0);
    for (int b = 0; b < NLEV; ++b) out[o++] = (float) (qN[b] / std::max (all, 1.0));
    for (int b = 0; b < NLEV; ++b) for (int k = 0; k <= NVAL; ++k) out[o++] = (float) (sAcc[b][k] / std::max (sN[b][k], 1.0) / 10.0);
    for (int k = 0; k < 4; ++k) out[o++] = (float) (std::log10 (near[k] / std::max (npos, 1.0) + 1e-6) / 3.0);
    return true;
}

// ---------------------------------------------------------------- the meter
struct Meter
{
    struct Lin { int in = 0, out = 0; std::vector<float> w, b; };       // w[out][in]
    struct Net { std::vector<float> mean, sd; std::vector<Lin> lin; };
    std::vector<Net> nets;              // several nets, taught alike from different starts: the meter's answer is the mean of theirs
    bool ready() const { return ! nets.empty(); }
    [[noreturn]] static void fail (const char* why) { throw std::runtime_error (std::string ("bend meter model: ") + why); }

    // Reads the file written by Tools/export_bend_model.py from memory. Throws std::runtime_error, and stays as it
    // was, when the block is not a complete model of the sizes its own header gives.
    void load (const void* data, size_t size)
    {
        const uint8_t* p = (const uint8_t*) data; size_t left = data != nullptr ? size : 0;
        auto take = [&] (void* dst, size_t bytes) { if (bytes > left) fail ("the file is cut short"); std::memcpy (dst, p, bytes); p += bytes; left -= bytes; };
        auto u32 = [&] { uint32_t v = 0; take (&v, 4); return v; };
        auto floats = [&] (std::vector<float>& v, size_t count)
        {
            if ((size_t) u32() != count) fail ("a layer does not have the size the header gives");
            if (count > left / 4) fail ("the file is cut short");
            v.resize (count); take (v.data(), count * 4);
            for (float f : v) if (! std::isfinite (f)) fail ("a number is not finite");
        };
        char magic[4]; take (magic, 4); if (std::memcmp (magic, "DRBM", 4) != 0) fail ("not a bend meter model file");
        if (u32() != 2) fail ("unknown format version");
        if (u32() != (uint32_t) NF) fail ("made for a different number of inputs");
        if (u32() != (uint32_t) NG) fail ("made for a different number of levels");
        const uint32_t count = u32(); if (count < 1 || count > 16) fail ("bad number of nets");
        Meter m;
        for (uint32_t k = 0; k < count; ++k)
        {
            const uint32_t layers = u32(); if (layers < 1 || layers > 16) fail ("bad number of layers");
            Net net; floats (net.mean, (size_t) NF); floats (net.sd, (size_t) NF);
            for (float s : net.sd) if (! (s > 0.0f)) fail ("a spread is not above zero");
            int in = 2 * NF + 1;
            for (uint32_t i = 0; i < layers; ++i)
            {
                const uint32_t out = u32(); if (out < 1 || out > 4096) fail ("bad layer width");
                Lin l; l.in = in; l.out = (int) out; floats (l.w, (size_t) out * (size_t) in); floats (l.b, (size_t) out); net.lin.push_back (std::move (l)); in = (int) out;
            }
            if (in != NG + 1) fail ("the last layer does not give the levels and the verdict");
            m.nets.push_back (std::move (net));
        }
        if (left != 0) fail ("more data than the header gives");
        *this = std::move (m);
    }

    // X: both half-waves' numbers ([2][NF], the upper half first); ratioDb: the upper half's peak against the lower
    // half's. Out: the squeeze per half-wave and level (dB), and the verdict (a logit: above 0 = more likely than not).
    void run (const float* X, double ratioDb, double S[2][NG], double& logit) const
    {
        logit = -99.0; for (int h = 0; h < 2; ++h) for (int g = 0; g < NG; ++g) S[h][g] = 0.0;
        if (! ready()) return;
        std::vector<float> Z ((size_t) 2 * (size_t) NF); double sum = 0;
        for (const Net& net : nets)
        {
            for (int h = 0; h < 2; ++h) for (int i = 0; i < NF; ++i) Z[(size_t) (h * NF + i)] = (X[h * NF + i] - net.mean[(size_t) i]) / net.sd[(size_t) i];
            for (int h = 0; h < 2; ++h)        // each half-wave in turn: its own numbers first, the other half's after, and how their peaks compare
            {
                std::vector<float> a ((size_t) (2 * NF + 1)), b;
                for (int i = 0; i < NF; ++i) { a[(size_t) i] = Z[(size_t) (h * NF + i)]; a[(size_t) (NF + i)] = Z[(size_t) ((1 - h) * NF + i)]; }
                a[(size_t) (2 * NF)] = (float) ((h == 0 ? ratioDb : -ratioDb) / 20.0);
                for (size_t li = 0; li < net.lin.size(); ++li)
                {
                    const Lin& L = net.lin[li]; b.assign ((size_t) L.out, 0.0f);
                    for (int o = 0; o < L.out; ++o)
                    {
                        const float* w = L.w.data() + (size_t) o * (size_t) L.in; double s = L.b[(size_t) o];
                        for (int i = 0; i < L.in; ++i) s += (double) w[i] * (double) a[(size_t) i];
                        b[(size_t) o] = li + 1 < net.lin.size() ? (float) std::max (s, 0.0) : (float) s;
                    }
                    a.swap (b);
                }
                for (int g = 0; g < NG; ++g) S[h][g] += 10.0 * (double) a[(size_t) g];
                sum += (double) a[(size_t) NG];
            }
        }
        const double k = 1.0 / (double) nets.size();
        for (int h = 0; h < 2; ++h) for (int g = 0; g < NG; ++g) S[h][g] *= k;
        logit = sum * k;
    }
};

// ---------------------------------------------------------------- the curve
// An answer made fit for use: no squeeze under zero, never falling towards the peak, none growing faster than the
// level itself allows (1 - 1/K dB per dB: the way back then spreads a value at most K times as far apart), held at
// maxSqueeze.
inline void tidy (const double* S, double K, double* r)
{
    const double most = 1.0 - 1.0 / K; double run = 0.0;
    for (int i = 0; i < NG; ++i) { run = std::max (run, std::min (std::max (S[i], 0.0), maxSqueeze)); r[i] = run; }
    r[0] = std::min (r[0], most * (GRID[0] - (SMALL - 3.0)));
    for (int i = 1; i < NG; ++i) r[i] = std::min (r[i], r[i - 1] + most * (GRID[i] - GRID[i - 1]));
}
// A smooth curve through rising points that never turns back between them (cubic pieces; the steepness at a point
// is the harmonic mean of the two straight pieces that meet there, at the two ends that of the one piece).
struct Smooth
{
    std::vector<double> x, y, m;
    Smooth (std::vector<double> xs, std::vector<double> ys) : x (std::move (xs)), y (std::move (ys)), m (x.size(), 0.0)
    {
        const size_t n = x.size(); std::vector<double> h (n - 1), d (n - 1);
        for (size_t i = 0; i + 1 < n; ++i) { h[i] = x[i + 1] - x[i]; d[i] = (y[i + 1] - y[i]) / h[i]; }
        m[0] = d[0]; m[n - 1] = d[n - 2];
        for (size_t i = 1; i + 1 < n; ++i) if (d[i - 1] * d[i] > 0.0) { const double w1 = 2.0 * h[i] + h[i - 1], w2 = h[i] + 2.0 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); }
    }
    double at (double v) const
    {
        size_t i = (size_t) (std::upper_bound (x.begin(), x.end(), v) - x.begin()); i = i == 0 ? 0 : i - 1; i = std::min (i, x.size() - 2);
        const double h = x[i + 1] - x[i], t = (v - x[i]) / h, t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h * m[i + 1];
    }
};
// One half-wave's curve both ways, on a table: inn (a value before the bend, in the scale where small values pass
// as they are) <-> out (the bent value, as in the sound). 16 steps per dB from 96 dB under the unbent peak to 12 over.
struct Curve
{
    static constexpr int N = 1729;
    double vp = 0, xp = 0, top = 0; std::vector<double> inn, out;
    Curve() = default;
    // A curve from nodes (a value before the bend -> the bent value, both rising): the line out = in under the first
    // node, straight pieces between the nodes, flat above the last. The table lies around ref. xpeak: the loudest value
    // before the bend, peak: the loudest bent value (a rebuilt half-wave's curve: flat tops, below).
    static Curve nodes (double ref, const Vec& xn, const Vec& yn, double xpeak, double peak)
    {
        Curve c; c.vp = peak; c.xp = xpeak; c.top = std::max (20.0 * std::log10 (std::max (xpeak, 1e-30) / std::max (peak, 1e-30)), 0.0); c.inn.resize ((size_t) N); c.out.resize ((size_t) N);
        for (int i = 0; i < N; ++i)
        {
            const double v = ref * std::pow (10.0, (-96.0 + (double) i / 16.0) / 20.0); c.inn[(size_t) i] = v; c.out[(size_t) i] = v < xn.front() ? v : declip::lin (xn, yn, v);
            if (i > 0) c.out[(size_t) i] = std::max (c.out[(size_t) i], c.out[(size_t) (i - 1)]);
        }
        return c;
    }
    Curve (const double* S, double peak, double K = spreadMost) : vp (peak), inn ((size_t) N), out ((size_t) N)
    {
        double r[NG]; tidy (S, K, r); top = r[NG - 1]; xp = vp * std::pow (10.0, top / 20.0);
        std::vector<double> lev { -60.0, SMALL - 3.0 }, sq { 0.0, 0.0 };
        for (int i = 0; i < NG; ++i) { lev.push_back (GRID[i]); sq.push_back (r[i]); }
        lev.push_back (12.0); sq.push_back (r[NG - 1] + (1.0 - 1.0 / K) * 12.0);         // above the peak: as flat as the cap allows
        const Smooth sm (lev, sq);
        for (int i = 0; i < N; ++i)
        {
            const double l = -96.0 + (double) i / 16.0; inn[(size_t) i] = xp * std::pow (10.0, l / 20.0); out[(size_t) i] = inn[(size_t) i] * std::pow (10.0, -sm.at (l) / 20.0);
            if (i > 0) out[(size_t) i] = std::max (out[(size_t) i], out[(size_t) (i - 1)] * (1.0 + 1e-9));       // the bent value never falls as the value before the bend rises
        }
    }
    static double look (const std::vector<double>& from, const std::vector<double>& to, double v)
    {
        if (v < from.front()) return v;         // under the table everything is a line
        if (v >= from.back()) return to.back();
        const size_t i = (size_t) (std::upper_bound (from.begin(), from.end(), v) - from.begin()) - 1;
        return to[i] + (to[i + 1] - to[i]) * (v - from[i]) / (from[i + 1] - from[i]);
    }
    double back (double v) const { return look (out, inn, v); }     // a bent value (>= 0) -> the value before the bend
    double fwd (double x) const { return look (inn, out, x); }      // a value before the bend (>= 0) -> bent
};

// ---------------------------------------------------------------- one sound's bend
// What the declipper made of a sound whose tops were cut flat (flat tops, below).
struct Flat
{
    bool did = false;                   // false: nothing of the declipper, the meter's curve alone
    bool rebuilt = false, second = false, tested = false;       // tops were rebuilt; a second pass was made and kept; the self-test was run
    bool whole = true;                  // a soft top's first pass worked on the whole sound (false: on a part of a long one)
    int kind[2] = { 0, 0 };             // per half-wave (the upper first): 0 no flat top, 1 a plain cut, 2 a soft flat top
    double c[2] = { 0, 0 };             // each half-wave's top value in the sound
    double test = 0;                    // the self-test's score
    double spread = 0, rho = 0, rough = 0;      // the first pass's own measures (flatSecond); how much rougher the second pass left the tops (flatRough)
    double share = 0, cut = 0;          // the share of all values that were unknown in the first pass; the share that sit at a plain cut
    std::shared_ptr<const Curve> tab[2];        // a half-wave's own curve (a plain cut, or a soft flat top), none where kind is 0
    std::vector<float> dl, dr;          // the sound before the bend less the sound in hand, at the app's rate, where this engine made it (0 elsewhere)
};
struct Found
{
    bool found = false, looked = false, stopped = false;
    size_t n = 0;                       // the sound it belongs to: its length
    double logit = -99.0, sure = 0.0;   // the meter's verdict, and the sureness 0 .. 1 made of it and of the squeeze at the peak
    double S[2][NG] = {};               // the squeeze per half-wave (the upper first) and level, as the meter read it (dB)
    double vp[2] = { 0, 0 };            // each half-wave's peak in the sound
    double top = 0;                     // how far the loudest values were squeezed, the stronger half-wave's (dB, made fit for use)
    std::shared_ptr<const Flat> flat;   // the declipper's part (none: not looked at)
    bool cut = false;                   // a plain cut whose cut-off peaks were rebuilt: DISTORTION whatever the squeeze
    bool hard() const { return cut || top >= hardAt; }
};
// How sure a bend is: the weaker of two witnesses, the meter's verdict and the squeeze it reads at the peak. The
// verdict alone is cautious (it was taught to say no where a bend cannot be heard, and says no to many a soft one
// that can); the squeeze alone takes any flat-topped sound. Together, on 96 sounds through outside boxes and 198
// clean ones: 45 of 48 distorted and 21 of 25 saturated sounds (6 dB and more at the peak) are shown where the slider
// stands at first, and of the clean ones only the four that have flat tops themselves (Research/bend).
inline double sureness (double logit, double top)
{
    const double e = std::min (logit - sureAt, top - sureTop);
    return 1.0 / (1.0 + std::exp (-e * 0.5 - std::log (shownSure / (1.0 - shownSure))));
}

inline Found find (const Stereo& x, const Meter& m, const Stop& stop = {})
{
    Found F; F.n = x.size(); if (! m.ready() || x.size() < minLen) return F;
    std::vector<float> X ((size_t) 2 * (size_t) NF);
    if (! features (x, 1.0, X.data(), F.vp[0], stop) || ! features (x, -1.0, X.data() + NF, F.vp[1], stop)) { F.stopped = true; return F; }
    F.looked = true;
    m.run (X.data(), 20.0 * std::log10 (F.vp[0] / F.vp[1]), F.S, F.logit);
    double r[NG]; for (int h = 0; h < 2; ++h) { tidy (F.S[h], spreadMost, r); F.top = std::max (F.top, r[NG - 1]); }
    F.sure = sureness (F.logit, F.top); F.found = F.logit >= lowAt && F.top >= topAt;
    return F;
}

// A file at another rate than the app's: the bend is read off the file's own samples, where every value still lies on
// the curve (bringing a sound to 48 kHz turns flat tops into ripples). From about twice the app's rate on, every
// second (third ...) sample is looked at: which samples are looked at does not matter to a curve, and the frames keep
// their length in time. n: the length of the sound at the app's rate, which the answer then belongs to.
inline Found findAt (const Stereo& raw, double rate, size_t n, const Meter& m, const Stop& stop = {})
{
    const size_t step = (size_t) std::max (1.0, std::floor (rate / SR + 0.25)); Found F;
    if (step <= 1) F = find (raw, m, stop);
    else
    {
        Stereo d; d.l.reserve (raw.size() / step + 1); d.r.reserve (raw.size() / step + 1);
        for (size_t i = 0; i < raw.size(); i += step) { d.l.push_back (raw.l[i]); d.r.push_back (raw.r[i]); }
        F = find (d, m, stop);
    }
    F.n = n; return F;
}

// The Amount as a drive: 0 = none of the found drive, 1 = as found, 2 = as far again in dB as the loudest values were squeezed.
inline double drive (double s, double top) { return s <= 1.0 ? s * s : std::pow (10.0, (s - 1.0) * top / 20.0); }

// The read squeeze as the way back uses it: three quarters of it, less 1 dB. The meter is off by a dB or two, and the
// two ways of being off are not alike: read too low, some of the bend stays in; read too high, values are spread apart
// that were never squeezed, which is a new fault and a worse one than the bend. On 48 outside cases this way back
// scores more than twice what the plain reading does and harms fewer of them (Research/bend).
inline void careful (const double* S, double* r) { for (int i = 0; i < NG; ++i) r[i] = std::max (backPart * std::min (std::max (S[i], 0.0), maxSqueeze) - backLess, 0.0); }

// ---------------------------------------------------------------- flat tops: the declipper
// A sound whose tops were cut flat. Where the curve pressed many levels into one, nothing in a value says how high it
// stood, but the sound around it does: a real sound is made of few partials that last a while, and the cut-off tops
// are what is missing for them to go on smoothly (the rebuilding itself and its tools: Declip.h; the prototype and the
// numbers: Research/declip). Each half-wave is looked at by itself:
//   a plain cut      many values sit exactly at its top, the values just under them are no denser than further
//                    down, and the next value down holds far fewer (a sound cut down to few steps has many values on
//                    every step: its top step is no cut). Nothing under the cut is touched; the values at the cut are
//                    rebuilt above it. This is found whatever the meter says: the meter reads a soft curve where there
//                    is a sharp cut, and walking that curve back spreads values apart that were never squeezed (build
//                    0.5.1 did so, and did harm).
//   a soft flat top  a pile of values under the top with a knee below it, and the meter has found a bend. The meter's
//                    reading is walked back up to a first height (where it turns flat), the rest is rebuilt; the
//                    rebuilt values then tell how the curve runs above that height (per level of the bent sound their
//                    middle value); the lower part of the zone is walked back along this curve and only the very top
//                    is rebuilt once more.
//   neither          the curve as the meter read it, walked back the careful way: nothing of this section.
// What keeps the rebuilding from doing harm, all of it read off the sound itself:
//   - a half-wave with three quarters of its values at the very top is left alone (there is nothing to go on), and both
//     sides of a wave are taken the same way where one of them fits both kinds (declip::regimes);
//   - the rebuilding is tried one step down first (the known values cut again 3 dB lower, rebuilt, and held against
//     what was there): a dense, noise-like sound fails, and is left as it is;
//   - a first pass whose values at one level of the bent sound lie far apart is refused: such tops follow no curve;
//   - a second pass that leaves the tops much rougher than the first did, or whose curve the first pass does not bear
//     out (higher bent values did not get higher values), is thrown away: such flat tops were not a curve's work (a
//     clipper that works at a higher rate and comes back down, for one).
constexpr double flatSlope = 0.5;               // the first height: where the bent value moves less than this many dB per dB of the value before the bend
constexpr double flatKeep = 0.8;                // the part of the zone above the first height that the second pass walks back along the curve the first one told
constexpr size_t zoneMin = 256, lowMin = 128;   // values a zone and its lower part must have for a curve to be read from them
constexpr double testMin = 20.0;                // the self-test's score under which nothing is rebuilt
constexpr double spreadMax = 3.5;               // dB: a first pass whose values lie further than this from their curve (the mean) is refused
constexpr double roughMax = 3.0; constexpr size_t topRun = 5;      // a second pass is kept if it leaves the tops (runs of topRun values at least) at most this many times as rough as the first
constexpr double sureCut = 0.9, topCut = 0.5;   // a plain cut shows by itself: how sure, and from how many dB of rebuilt peak above the cut
constexpr double flatLongest = 1200.0;          // seconds: a longer sound is not rebuilt (on four threads a plain cut takes about a tenth of the sound's length and a soft flat top about a quarter; the result takes as much memory again as half the sound)
constexpr int budHard[2] = { 8, 12 }, budFirst[2] = { 16, 25 }, budSecond[2] = { 8, 12 };      // rounds (steps of the schedule, rounds per step) for a plain cut and for a soft top's two passes: the first must settle, it tells the curve
constexpr int NO = NG + 2;

// One half-wave's reading over the bent sound's own levels: o = the level of a bent value in dB under the half-wave's
// peak (rising), q = by how many dB it was squeezed. The reading may not rise faster than the level.
inline void outSide (const double* S, double* o, double* q)
{
    double r[NG], lev[NO], run = 0.0;
    for (int i = 0; i < NG; ++i) { run = std::max (run, std::min (std::max (S[i], 0.0), maxSqueeze)); r[i] = run; }
    for (int i = 1; i < NG; ++i) r[i] = std::min (r[i], r[i - 1] + (GRID[i] - GRID[i - 1]));
    lev[0] = -90.0; lev[1] = SMALL - 3.0; q[0] = q[1] = 0.0;
    for (int i = 0; i < NG; ++i) { lev[i + 2] = GRID[i]; q[i + 2] = r[i]; }
    for (int i = 0; i < NO; ++i) o[i] = lev[i] - q[i] + q[NO - 1];
    for (int i = 1; i < NO; ++i) o[i] = std::max (o[i], o[i - 1] + 1e-6);
}
// The height (dB under the peak) up to which the reading is walked back. False: all the way up.
inline bool trustAt (const double* o, const double* q, double& at)
{
    int i = NO - 1;
    while (i > 0 && (o[i] - o[i - 1]) / std::max ((o[i] + q[i]) - (o[i - 1] + q[i - 1]), 1e-9) < flatSlope) --i;
    if (i == NO - 1) return false;
    at = o[i]; return true;
}
// Where a pass starts from.
struct FlatStart
{
    Stereo x;                           // the values walked back where they are trusted; at the others their bound (its sign: the side)
    declip::Flags u[2];                 // where the value is not known
    double bp = std::numeric_limits<double>::infinity(), bn = -std::numeric_limits<double>::infinity();     // the bounds above and below zero (infinite: that side has none)
    struct Half { double os = 0, o[NO] = {}, q[NO] = {}; } half[2];      // a soft half-wave: the first height (dB under the peak) and the reading's curve
    int kind[2] = { 0, 0 }; size_t unknown = 0;
};
inline FlatStart flatStart (const Stereo& y, const Found& F, const declip::Stat* st, const int* kinds)
{
    FlatStart W; W.kind[0] = kinds[0]; W.kind[1] = kinds[1]; W.x = y; const size_t n = y.size(); W.u[0].assign (n, 0); W.u[1].assign (n, 0);
    double b[2] = { 0, 0 }, Sc[2][NG];
    for (int h = 0; h < 2; ++h)
    {
        if (W.kind[h] == 2)
        {
            FlatStart::Half& H = W.half[h]; outSide (F.S[h], H.o, H.q); double os = 0.0; bool has = trustAt (H.o, H.q, os);
            if (declip::tied (st[h])) { os = std::min (20.0 * std::log10 (st[h].c / F.vp[h]) - declip::TIE_DB, has ? os : 0.0); has = true; }
            if (! has) W.kind[h] = 0;      // no height from which the reading is flat: nothing to rebuild
            else { H.os = os; b[h] = F.vp[h] * declip::db (os + declip::lin (H.o, H.q, (size_t) NO, os)); }
        }
        if (W.kind[h] == 1) b[h] = st[h].c * declip::db (-declip::TIE_DB);
        careful (F.S[h], Sc[h]);
    }
    const Curve cp (Sc[0], F.vp[0]), cn (Sc[1], F.vp[1]); const Curve* cv[2] = { &cp, &cn }; Vec* xs[2] = { &W.x.l, &W.x.r }; bool any[2] = { false, false };
    for (int c = 0; c < 2; ++c)
        for (size_t i = 0; i < n; ++i)
        {
            const double v = (*xs[c])[i]; if (v == 0.0) continue;
            const int h = v > 0.0 ? 0 : 1; const double pol = h == 0 ? 1.0 : -1.0, a = pol * v; double x = a; bool unk = false;
            if (W.kind[h] == 1) unk = a >= b[h];
            else if (W.kind[h] == 2)
            {
                const double lv = 20.0 * std::log10 (std::max (a, 1e-30) / F.vp[h]);
                if (lv >= W.half[h].os) unk = true; else x = a * declip::db (declip::lin (W.half[h].o, W.half[h].q, (size_t) NO, lv));
            }
            else x = cv[h]->back (a);
            if (unk) { x = b[h]; W.u[c][i] = 1; any[h] = true; ++W.unknown; }
            (*xs[c])[i] = pol * x;
        }
    if (any[0]) W.bp = b[0];
    if (any[1]) W.bn = -b[1];
    return W;
}
// A half-wave's unknown values as pairs (bent value, rebuilt value), in the order of the samples. seen: only where
// these flags are set (null: everywhere).
inline std::vector<std::pair<double, double>> flatPairs (const Stereo& y, const Stereo& out, const declip::Flags* u, double pol, const declip::Flags* seen = nullptr)
{
    std::vector<std::pair<double, double>> pr; const size_t n = y.size();
    for (size_t i = 0; i < n; ++i)
    {
        if (u[0][i] && pol * y.l[i] > 0.0 && (seen == nullptr || seen[0][i])) pr.push_back ({ pol * y.l[i], pol * out.l[i] });
        if (u[1][i] && pol * y.r[i] > 0.0 && (seen == nullptr || seen[1][i])) pr.push_back ({ pol * y.r[i], pol * out.r[i] });
    }
    return pr;
}
// The second pass's start, from the first rebuild. For the soft half-waves the curve above the first height is read
// from the rebuilt values (from the zone's lower part, never from the ties); those values are walked back along it and
// count as known, the rest of the zone stays unknown above the curve's last value. cy, cx: the curves' nodes (bent
// value, value before the bend). seen: where the first pass has worked (null: everywhere); the curve is read there
// and serves the whole sound. False: no soft half-wave had enough of a zone.
// What the first pass's values say of themselves (over the half-waves, by their numbers of pairs): spread, how far they
// lie from the curve at their level of the bent sound (the mean, dB); rho, whether higher bent values got higher
// values (the rank correlation in the upper half of the pairs).
inline bool flatSecond (const Stereo& y, const FlatStart& W, const Stereo& out, const declip::Stat* st, FlatStart& W2, Vec* cy, Vec* cx, double& spread, double& rho, const declip::Flags* seen = nullptr)
{
    W2 = W; bool did = false; const size_t n = y.size(); const Vec* ys[2] = { &y.l, &y.r }; Vec* xs[2] = { &W2.x.l, &W2.x.r }; double sp = 0.0, rh = 0.0, wt = 0.0;
    for (int h = 0; h < 2; ++h)
    {
        if (W.kind[h] != 2) continue;
        const double pol = h == 0 ? 1.0 : -1.0; std::vector<std::pair<double, double>> pr = flatPairs (y, out, W.u, pol, seen), low;
        if (pr.size() < zoneMin) continue;
        Vec yy; yy.reserve (pr.size()); double y0 = pr[0].first, x0 = pr[0].second;
        for (const auto& p : pr) { yy.push_back (p.first); y0 = std::min (y0, p.first); x0 = std::min (x0, p.second); }
        double th = declip::quantile (yy, flatKeep); if (declip::tied (st[h])) th = std::min (th, st[h].c * declip::db (-declip::TIE_DB));
        for (const auto& p : pr) if (p.first < th) low.push_back (p);
        if (low.size() < lowMin) continue;
        declip::curveFrom (low, y0, x0, cy[h], cx[h]); did = true;
        {
            Vec lv; lv.reserve (low.size()); std::vector<std::pair<double, double>> up;
            for (const auto& p : low) { sp += std::abs (20.0 * std::log10 (std::max (p.second, 1e-30)) - 20.0 * std::log10 (std::max (declip::lin (cy[h], cx[h], p.first), 1e-30))); lv.push_back (p.first); }
            const double med = declip::median (lv); for (const auto& p : low) if (p.first >= med) up.push_back (p);
            rh += (double) low.size() * declip::rankCorr (up); wt += (double) low.size();
        }
        for (int c = 0; c < 2; ++c)
            for (size_t i = 0; i < n; ++i)
            {
                const double a = pol * (*ys[c])[i]; if (! W.u[c][i] || ! (a > 0.0)) continue;
                if (a < th) { (*xs[c])[i] = pol * declip::lin (cy[h], cx[h], a); W2.u[c][i] = 0; --W2.unknown; }
                else (*xs[c])[i] = pol * cx[h].back();
            }
    }
    spread = sp / std::max (wt, 1.0); rho = rh / std::max (wt, 1.0);
    return did;
}
// How much rougher the second pass (out1) leaves the tops than the first (out0): the root of the summed squares of the
// second differences inside the tops, one over the other. A top: a run of topRun values at least of a soft half-wave's
// zone (one whose curve was read: did), where the first pass has worked (seen; null: everywhere).
inline double flatRough (const Stereo& y, const FlatStart& W, const bool* did, const Stereo& out0, const Stereo& out1, const declip::Flags* seen = nullptr)
{
    double r0 = 0.0, r1 = 0.0; const size_t n = y.size(); const Vec* ys[2] = { &y.l, &y.r }; const Vec* a0[2] = { &out0.l, &out0.r }; const Vec* a1[2] = { &out1.l, &out1.r };
    for (int h = 0; h < 2; ++h)
    {
        if (! did[h]) continue;
        const double pol = h == 0 ? 1.0 : -1.0;
        for (int c = 0; c < 2; ++c)
        {
            const Vec& v = *ys[c]; const Vec& p0 = *a0[c]; const Vec& p1 = *a1[c]; const declip::Flags& u = W.u[c];
            auto in = [&] (size_t k) { return u[k] && pol * v[k] > 0.0 && (seen == nullptr || seen[c][k]); };
            for (size_t i = 0; i < n;)
            {
                if (! in (i)) { ++i; continue; }
                size_t j = i; while (j < n && in (j)) ++j;
                if (j - i >= topRun) for (size_t k = i; k + 2 < j; ++k) { const double d0 = p0[k] - 2.0 * p0[k + 1] + p0[k + 2], d1 = p1[k] - 2.0 * p1[k + 1] + p1[k + 2]; r0 += d0 * d0; r1 += d1 * d1; }
                i = j;
            }
        }
    }
    return std::sqrt (r1 / std::max (r0, 1e-30));
}
// A rebuilt half-wave's curve from what the engine did: under the first height the walk that was made, above it the
// curve the rebuilt values tell. A plain cut: the line up to the cut, flat from there. xhat: the sound before the bend,
// u: what was still unknown in the last pass, cy, cx: the second pass's curve (empty: none).
inline std::shared_ptr<const Curve> flatTable (int h, const Stereo& y, const Stereo& xhat, const declip::Flags* u, const FlatStart::Half& H, const Vec& cy, const Vec& cx, int kind, const declip::Stat& st, double vp)
{
    const double pol = h == 0 ? 1.0 : -1.0, xp = peakOf (xhat, pol);
    if (kind == 1) return std::make_shared<const Curve> (Curve::nodes (st.c, Vec { st.c }, Vec { st.c }, xp, st.c));
    Vec xn, yn; const long len = (long) std::ceil ((H.os + 96.0) * 16.0);
    for (long i = 0; i <= len; ++i) { const double lb = i < len ? -96.0 + (double) i / 16.0 : H.os; xn.push_back (vp * declip::db (lb + declip::lin (H.o, H.q, (size_t) NO, lb))); yn.push_back (vp * declip::db (lb)); }
    xn.insert (xn.end(), cx.begin(), cx.end()); yn.insert (yn.end(), cy.begin(), cy.end());
    std::vector<std::pair<double, double>> pr = flatPairs (y, xhat, u, pol);
    if (pr.size() >= (size_t) declip::LEAST)
    {
        Vec ym, xm; declip::curveFrom (pr, yn.back(), xn.back(), ym, xm);
        xn.insert (xn.end(), xm.begin() + 1, xm.end()); yn.insert (yn.end(), ym.begin() + 1, ym.end());
    }
    Vec X { xn[0] }, Y { yn[0] };          // both rising: a node that does not lie further out than the one before it is left out
    for (size_t i = 1; i < xn.size(); ++i) if (xn[i] > X.back() * (1.0 + 1e-9) && yn[i] >= Y.back()) { X.push_back (xn[i]); Y.push_back (yn[i]); }
    return std::make_shared<const Curve> (Curve::nodes (xp, X, Y, xp, vp));
}
// What the declipper makes of a sound. y: the sound's own samples at `rate`; F: the meter's answer for it. xhat: the
// sound before the bend at that rate, in the scale where small values pass as they are (when the answer says did; a
// half-wave without a flat top is walked back the careful way in it). Null: given up.
// A soft flat top's first pass costs the most rounds, and its use is to tell the curve: on a long sound it works on
// parts spread evenly over the sound (first hops per channel in all: declip::choose), the curve it tells serves the
// whole sound, and the second pass rebuilds all of it. Should the second pass not be kept, the first is made on the
// whole sound after all. A sound of more than flatLongest seconds is not rebuilt at all (as if the self-test had failed).
// Refused (the self-test failed, or the first pass contradicts itself): a plain cut stays as it is, a soft top goes
// the careful way.
// look: for studies, what each pass made.
struct FlatLook { Stereo first, second; };
inline std::shared_ptr<Flat> flatTops (const Stereo& y, double rate, const Found& F, Stereo& xhat, const Stop& stop = {}, int first = declip::FIRST, int part = declip::PART, FlatLook* look = nullptr)
{
    auto R = std::make_shared<Flat>(); const int N = declip::window (rate); const size_t n = y.size();
    if (! F.looked || n < (size_t) (4 * N)) return R;
    const declip::Stat st[2] = { declip::stats (y, 1.0, F.vp[0]), declip::stats (y, -1.0, F.vp[1]) }; int kinds[2]; declip::regimes (st, F.found, kinds);
    R->c[0] = st[0].c; R->c[1] = st[1].c;
    if (kinds[0] == 0 && kinds[1] == 0) return R;
    FlatStart W = flatStart (y, F, st, kinds); R->kind[0] = W.kind[0]; R->kind[1] = W.kind[1]; R->share = (double) W.unknown / (double) (2 * n);
    R->cut = (double) ((W.kind[0] == 1 ? st[0].tn : 0) + (W.kind[1] == 1 ? st[1].tn : 0)) / (double) (2 * n);
    if (W.unknown == 0) return R;
    const bool soft = W.kind[0] == 2 || W.kind[1] == 2; const int* bud = soft ? budFirst : budHard;
    const bool tooLong = (double) n > flatLongest * rate;
    auto refused = [&] () -> std::shared_ptr<Flat>         // rebuilding would do harm: a plain cut stays as it is, a soft top goes the careful way
    {
        const int k2[2] = { W.kind[0] == 1 ? 1 : 0, W.kind[1] == 1 ? 1 : 0 }; R->kind[0] = k2[0]; R->kind[1] = k2[1];
        if (k2[0] == 0 && k2[1] == 0) return R;
        W = flatStart (y, F, st, k2); xhat = W.x;
        for (size_t i = 0; i < n; ++i) { if (W.u[0][i]) xhat.l[i] = y.l[i]; if (W.u[1][i]) xhat.r[i] = y.r[i]; }
        for (int h = 0; h < 2; ++h) if (k2[h] == 1)
        {
            double r[NG]; tidy (F.S[h], spreadMost, r); const double c = st[h].c;
            R->tab[h] = std::make_shared<const Curve> (Curve::nodes (c, Vec { c }, Vec { c }, c * std::pow (10.0, r[NG - 1] / 20.0), c));
        }
        R->did = true; return R;
    };
    if (! tooLong) { if (! declip::selfTest (W.x, W.bp, W.bn, bud[0], bud[1], rate, N, R->test, stop)) return nullptr; R->tested = true; }
    if (tooLong || R->test < testMin) return refused();
    std::vector<declip::Piece> only[2]; declip::Flags seen[2];
    const bool whole = ! soft || first <= 0 || declip::choose (W.u, N / 2, (size_t) first, part, only, seen); R->whole = whole;
    Stereo out; if (! declip::rebuild (W.x, W.u, out, bud[0], bud[1], N, stop, whole ? nullptr : only)) return nullptr;
    if (look != nullptr) look->first = out;
    FlatStart W2; Vec cy[2], cx[2]; const declip::Flags* uf = W.u;
    if (soft)
    {
        Vec ty[2], tx[2];
        if (flatSecond (y, W, out, st, W2, ty, tx, R->spread, R->rho, whole ? nullptr : seen))
        {
            if (R->spread > spreadMax) return refused();         // the first pass contradicts itself: such tops follow no curve
            Stereo out1 = W2.x; if (W2.unknown > 0 && ! declip::rebuild (W2.x, W2.u, out1, budSecond[0], budSecond[1], N, stop)) return nullptr;
            if (look != nullptr) look->second = out1;
            const bool did[2] = { ! ty[0].empty(), ! ty[1].empty() }; R->rough = flatRough (y, W, did, out, out1, whole ? nullptr : seen);
            if (R->rough <= roughMax && R->rho > 0.0) { out = std::move (out1); uf = W2.u; for (int h = 0; h < 2; ++h) { cy[h] = ty[h]; cx[h] = tx[h]; } R->second = true; }
        }
        if (! R->second && ! whole && ! declip::rebuild (W.x, W.u, out, bud[0], bud[1], N, stop)) return nullptr;       // the first pass is the answer after all: on the whole sound
    }
    for (int h = 0; h < 2; ++h) if (W.kind[h] != 0) R->tab[h] = flatTable (h, y, out, uf, W.half[h], cy[h], cx[h], W.kind[h], st[h], F.vp[h]);
    xhat = std::move (out); R->did = R->rebuilt = true; return R;
}
// The declipper's part of a found bend, put into F. sound: the sound at the app's rate. raw, rate: the file's own
// samples where the file is at another rate (its tops are flat only there), else null. A plain cut whose peaks were
// rebuilt by topCut dB or more shows by itself, whatever the meter said.
inline void addFlat (Found& F, const Stereo& sound, const Stereo* raw = nullptr, double rate = SR, const Stop& stop = {}, int first = declip::FIRST, int part = declip::PART, FlatLook* look = nullptr)
{
    if (! F.looked || F.stopped) return;
    const Stereo& y = raw != nullptr ? *raw : sound; Stereo xhat;
    std::shared_ptr<Flat> R = flatTops (y, raw != nullptr ? rate : SR, F, xhat, stop, first, part, look);
    if (R == nullptr) { if (stop && stop()) F.stopped = true; return; }         // given up; or the rebuilding could not be done: the meter's curve alone, as before
    if (R->did)
    {
        // the sound before the bend less the sound in hand, where this engine made it (a half-wave without a flat top is the meter's curve's to walk back)
        Stereo d; d.l.assign (y.size(), 0.0); d.r.assign (y.size(), 0.0); const Vec* ys[2] = { &y.l, &y.r }; const Vec* hs[2] = { &xhat.l, &xhat.r }; Vec* ds[2] = { &d.l, &d.r };
        for (int c = 0; c < 2; ++c) for (size_t i = 0; i < y.size(); ++i) { const double v = (*ys[c])[i]; if (v != 0.0 && R->tab[v > 0.0 ? 0 : 1] != nullptr) (*ds[c])[i] = (*hs[c])[i] - v; }
        if (raw != nullptr) d = resampled (d, rate, SR);
        d.l.resize (sound.size(), 0.0); d.r.resize (sound.size(), 0.0);
        R->dl.assign (d.l.begin(), d.l.end()); R->dr.assign (d.r.begin(), d.r.end());
        bool cut = false, plain = false; double top = 0.0;
        for (int h = 0; h < 2; ++h)
        {
            if (R->tab[h] == nullptr) { plain = true; continue; }
            top = std::max (top, R->tab[h]->top); if (R->kind[h] == 1 && R->rebuilt && R->tab[h]->top >= topCut) cut = true;
        }
        if (plain) top = std::max (top, F.top);
        if (cut) { F.found = true; F.sure = std::max (F.sure, sureCut); F.cut = true; }
        F.top = top;
    }
    F.flat = R;
}
// A half-wave's curve as the engine has it. own: for the sound itself and its picture, a flat top's own curve. For a
// second sound (own false) the meter's, as in build 0.5.1, but for a plain cut whose tops were rebuilt: the line up to
// the cut and flat from there, and the rebuilt tops say how far past the cut the sound was driven. Of a cut whose tops
// were not rebuilt that is not known, and the meter's reading stands.
inline Curve curveOf (const Found& F, int h, bool own)
{
    if (F.flat != nullptr && F.flat->did && F.flat->tab[h] != nullptr && (own || (F.flat->kind[h] == 1 && F.flat->rebuilt))) return *F.flat->tab[h];
    return Curve (F.S[h], F.vp[h]);
}

// The sound itself where the declipper had a part. With x the sound before the bend (rebuilt where its tops were flat)
// and f a half-wave's curve: f (g x) + w (y - f (x)), g the drive the fader stands for. What the curve does not
// explain of the sound in hand (little: the spread of rebuilt tops around the curve) comes back as the fader nears its
// middle, w = g squared up to 1, so that the middle is the sound itself and far left the rebuilt sound.
inline Stereo ownFlat (const Stereo& y, const Found& F, double s, const Stop& stop = {})
{
    const Flat& A = *F.flat; double Sp[NG], Sn[NG];
    if (s < 1.0) { careful (F.S[0], Sp); careful (F.S[1], Sn); } else { std::copy (F.S[0], F.S[0] + NG, Sp); std::copy (F.S[1], F.S[1] + NG, Sn); }
    const Curve mp (Sp, F.vp[0]), mn (Sn, F.vp[1]); const bool tab[2] = { A.tab[0] != nullptr, A.tab[1] != nullptr }; const Curve* cv[2] = { tab[0] ? A.tab[0].get() : &mp, tab[1] ? A.tab[1].get() : &mn };
    const double g = drive (std::clamp (s, 0.0, 2.0), std::max (cv[0]->top, cv[1]->top)), w = std::min (g, 1.0) * std::min (g, 1.0);
    Stereo o = y; Vec* os[2] = { &o.l, &o.r }; const std::vector<float>* ds[2] = { &A.dl, &A.dr };
    for (int c = 0; c < 2; ++c)
    {
        if (stop && stop()) return y;
        Vec& ch = *os[c]; const std::vector<float>& d = *ds[c];
        for (size_t i = 0; i < ch.size(); ++i)
        {
            const double v = ch[i]; const int h = v >= 0.0 ? 0 : 1; const double pol = h == 0 ? 1.0 : -1.0, a = pol * v;
            const double x = tab[h] ? std::max (a + pol * (double) d[i], 0.0) : cv[h]->back (a);
            ch[i] = pol * (g <= 0.0 ? x : cv[h]->fwd (g * x) + w * (a - cv[h]->fwd (x)));
        }
    }
    sameLevel (o, y); return o;
}
// The sound itself. s: 0 = the bend walked back (as far as careful() goes; flat tops as the declipper rebuilt them),
// 1 = untouched, 2 = pushed as far again. The sound's peak stays where it was (sameLevel).
inline Stereo own (const Stereo& y, const Found& F, double s, const Stop& stop = {})
{
    if (! F.found || s == 1.0 || y.size() == 0) return y;
    if (F.flat != nullptr && F.flat->did && F.flat->dl.size() == y.size() && F.flat->dr.size() == y.size()) return ownFlat (y, F, s, stop);
    double Sp[NG], Sn[NG];
    if (s < 1.0) { careful (F.S[0], Sp); careful (F.S[1], Sn); } else { std::copy (F.S[0], F.S[0] + NG, Sp); std::copy (F.S[1], F.S[1] + NG, Sn); }
    const Curve cp (Sp, F.vp[0]), cn (Sn, F.vp[1]); const double g = drive (std::clamp (s, 0.0, 2.0), std::max (cp.top, cn.top));
    Stereo o = y;
    for (Vec* ch : { &o.l, &o.r })
    {
        if (stop && stop()) return y;
        for (double& v : *ch)
        {
            const double x = v >= 0.0 ? cp.back (v) : -cn.back (-v), d = g * x;
            v = g <= 0.0 ? x : d >= 0.0 ? cp.fwd (d) : -cn.fwd (-d);
        }
    }
    sameLevel (o, y); return o;
}
// Another sound through the found curve. s: 0 = none, 1 = its peak pushed as far into the curve as the first sound's
// was, 2 = as far again. Its peak stays where it was.
// A plain cut whose tops were rebuilt goes on as a cut, the second sound's peak as far past it as the first sound's
// rebuilt peak stands. (Cutting as many of the second sound's values as of the first sound's was tried: the same drive
// cuts very different shares of two sounds, and on six cases that rule scored -9 where this one scores 72.) A soft flat
// top, and a cut whose tops were not rebuilt, go onto a second sound as the meter read them.
inline Stereo onto (const Stereo& b, const Found& F, double s, const Stop& stop = {})
{
    if (! F.found || s <= 0.0 || b.size() == 0) return b;
    const double pb = peakOf (b, 0.0); if (! (pb > 1e-12)) return b;
    const Curve cp = curveOf (F, 0, false), cn = curveOf (F, 1, false); const double k = drive (std::min (s, 2.0), std::max (cp.top, cn.top)) * std::max (cp.xp, cn.xp) / pb;
    Stereo o = b;
    for (Vec* ch : { &o.l, &o.r })
    {
        if (stop && stop()) return b;
        for (double& v : *ch) { const double d = k * v; v = d >= 0.0 ? cp.fwd (d) : -cn.fwd (-d); }
    }
    const double c = pb / std::max (peakOf (o, 0.0), 1e-30); for (Vec* ch : { &o.l, &o.r }) for (double& v : *ch) v *= c;
    return o;
}

} // namespace bend
} // namespace dr
