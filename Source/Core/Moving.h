#pragma once
// The moving-tone engine (PHASER, FILTER SWEEP): a filter whose curve moves over time.
//   find()      reads the moving curve off one sound: for each side, per moment and band, by how many dB the filter changed
//               the level; where along the sound it is; which of the two kinds it is and how sure
//   apply()     s = -1: the found curve taken off the sound it was found on; s = +1: put onto another sound at the same
//               times; in between and up to +2: scaled (in dB). Speed: the curve moves slower or faster.
//   where()     the stretches of the sound that have the moving filter (for the caption and the line under the waveform)
// How: a phaser and a filter sweep leave no line in a cepstrum, as a flanger's copy does; what they leave is in the picture
// of the band levels over time (96 bands, 12 an octave, a frame every 21 ms): an edge, a peak or a few notches that travel
// up and down while the sound's own partials stay where they are. A small network of plain 3x3 layers (Net: the picture is
// halved in time five times and across the bands three times on the way down and brought back up with the finer levels
// joined in) was fitted on sounds put through moving filters whose true curve was read with a steady probe signal through
// the same filter (Research/moving). It answers two things: the curve per band and moment, and per moment whether a filter
// sweep or a phaser is at work. Stretches where it says so for a second or more are the effect; the curve works there only.
// A sound longer than 6 s is read in blocks of 6 s, one every 3 s, each on its own (as the model was fitted), and the
// answers are faded into one another. The curve is put on and taken off in a short-time transform (2048 points, four
// windows over every sample): each bin is turned up or down by the curve at its frequency, never lifted by more than
// LIFT dB. A port of the Python prototype Research/moving/moving.py (Tools/moving_parity.py). 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Room.h"      // dr::Vec, dr::Stereo (Dsp.h) and room::Transform

namespace dr
{
namespace moving
{
using Stop = std::function<bool()>;                     // asked now and then while measuring: true = give up (the answer is then not used)
constexpr int N = 2048, HOP = 1024, NB = 96, BINS = N / 2 + 1;      // the picture: window, frame step, bands
constexpr double F0 = 70.0, BPO = 12.0;                 // the first band's centre, bands an octave (the last sits at 16.9 kHz)
constexpr double FLOOR = 70.0;                          // a cell this far (dB) under the loudest cell counts as silent
constexpr int RN = 2048, RH = 512;                      // the transform the curve is put on with
constexpr double FPS = SR / HOP;
constexpr int BLOCK = 282, BHOP = 141;                  // frames: the model reads 6 s at a time, every 3 s
constexpr double ON = 0.5;                              // a frame counts from this probability of a moving filter
constexpr int GAP = 24, SHORT = 47, RAMP = 5, SMOOTH = 15;          // frames: holes under 0.5 s are closed, stretches under 1 s dropped; 0.1 s fades; the say-so is averaged over 0.3 s
constexpr double PERIODIC = 0.8;                        // the curve repeats if it is this like itself one period on
constexpr double GS = 10.0;                             // the model's unit for the curve: 10 dB
constexpr double LIFT = 24.0, CUT = 60.0;               // the most a bin is turned up or down

inline int frames (size_t n) { return (int) (n / (size_t) HOP) + 1; }

// ---------------------------------------------------------------- the picture
// Band b takes the bins around its centre with a triangle that reaches one band (at least one bin) to each side.
struct Bands
{
    double fc[NB]; int first[NB], count[NB], off[NB]; Vec w;
    Bands()
    {
        for (int b = 0; b < NB; ++b)
        {
            fc[b] = F0 * std::pow (2.0, b / BPO);
            const double lo = std::min (fc[b] * std::pow (2.0, -1.0 / BPO), fc[b] - SR / N), hi = std::max (fc[b] * std::pow (2.0, 1.0 / BPO), fc[b] + SR / N);
            auto at = [&] (int k) { const double f = k * SR / N; return std::clamp (std::min ((f - lo) / (fc[b] - lo), (hi - f) / (hi - fc[b])), 0.0, 1.0); };
            first[b] = -1; count[b] = 0; off[b] = (int) w.size(); double sum = 0;
            for (int k = 0; k < BINS; ++k) if (at (k) > 0) { if (first[b] < 0) first[b] = k; count[b] = k - first[b] + 1; }
            for (int k = first[b]; k < first[b] + count[b]; ++k) { w.push_back (at (k)); sum += at (k); }
            for (int i = 0; i < count[b]; ++i) w[(size_t) (off[b] + i)] /= sum;
        }
    }
};

// Band levels in dB per frame, [frame * NB + band]; frame t is centred on sample t * HOP.
inline Vec picture (const Vec& x)
{
    static const Bands B; static const room::Transform F (N);
    static const Vec win = [] { Vec w ((size_t) N); for (int i = 0; i < N; ++i) w[(size_t) i] = 0.5 - 0.5 * std::cos (2 * PI * i / N); return w; }();
    const int T = frames (x.size()); const long n = (long) x.size(); Vec LE ((size_t) T * NB), re ((size_t) N), im ((size_t) N), p ((size_t) BINS);
    for (int t = 0; t < T; ++t)
    {
        const long s = (long) t * HOP - N / 2;
        for (int i = 0; i < N; ++i) { const long j = s + i; re[(size_t) i] = j >= 0 && j < n ? x[(size_t) j] * win[(size_t) i] : 0.0; im[(size_t) i] = 0.0; }
        F.run (re, im);
        for (int k = 0; k < BINS; ++k) p[(size_t) k] = re[(size_t) k] * re[(size_t) k] + im[(size_t) k] * im[(size_t) k];
        for (int b = 0; b < NB; ++b)
        {
            double v = 0; const double* w = &B.w[(size_t) B.off[b]]; const double* q = &p[(size_t) B.first[b]];
            for (int i = 0; i < B.count[b]; ++i) v += w[i] * q[i];
            LE[(size_t) t * NB + (size_t) b] = 10.0 * std::log10 (v + 1e-13);
        }
    }
    return LE;
}

constexpr int PLANES = 4;
// What the model reads of T frames of the picture, [plane][frame][band]: the level under the loudest cell, the level against
// the band's own middle level over these frames, that middle level, and the band's place.
inline std::vector<float> inputs (const double* LE, int T)
{
    const size_t cells = (size_t) T * NB; std::vector<float> X ((size_t) PLANES * cells);
    double top = -1e300; for (size_t i = 0; i < cells; ++i) top = std::max (top, LE[i]);
    Vec col; col.reserve ((size_t) T);
    for (int b = 0; b < NB; ++b)
    {
        col.clear(); for (int t = 0; t < T; ++t) { const double v = LE[(size_t) t * NB + (size_t) b]; if (v > top - FLOOR) col.push_back (v); }
        double med = top - FLOOR;
        if (! col.empty())
        {
            const size_t h = col.size() / 2; std::nth_element (col.begin(), col.begin() + (long) h, col.end()); med = col[h];
            if (col.size() % 2 == 0) med = 0.5 * (med + *std::max_element (col.begin(), col.begin() + (long) h));
        }
        const float mp = (float) (std::clamp ((med - top) / 20.0, -3.5, 0.0) + 1.75), pos = (float) (-1.0 + 2.0 * b / (NB - 1));
        for (int t = 0; t < T; ++t)
        {
            const size_t i = (size_t) t * NB + (size_t) b; const double v = LE[i];
            X[i] = (float) (std::clamp ((v - top) / 20.0, -3.5, 0.0) + 1.75);
            X[cells + i] = v > top - FLOOR ? (float) (std::clamp (v - med, -30.0, 30.0) / 15.0) : -2.0f;
            X[2 * cells + i] = mp; X[3 * cells + i] = pos;
        }
    }
    return X;
}

// ---------------------------------------------------------------- the model
struct Net
{
    struct Layer { int in = 0, out = 0; std::vector<float> w, b; };      // w[out][in][time][band]
    int planes = PLANES, bpool = 3;         // input planes; the levels (from the second on) that also halve the bands
    std::vector<int> ch;                    // channels per level, finest first
    std::vector<Layer> down, up;            // two layers per level; `up` starts one above the coarsest level
    Layer head;                             // 1x1: the finest level's channels -> the curve and the two scores per cell
    bool ready() const { return ! ch.empty(); }

    // Reads the file written by Tools/export_moving_model.py from memory. Throws std::runtime_error, and stays as it was,
    // when the block is not a complete model of the sizes its own header gives.
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
        auto layer = [&] (int in, int out, int k) { Layer l; l.in = in; l.out = out; floats (l.w, (size_t) out * (size_t) in * (size_t) (k * k)); floats (l.b, (size_t) out); return l; };
        char magic[4]; take (magic, 4); if (std::memcmp (magic, "DRMV", 4) != 0) fail ("not a moving-tone model file");
        if (u32() != 1) fail ("unknown format version");
        Net m; if (u32() != (uint32_t) PLANES) fail ("made for a different number of input planes");
        const uint32_t levels = u32(); if (levels < 2 || levels > 10) fail ("bad number of levels");
        const uint32_t bp = u32(); if (bp >= levels || (NB >> bp) < 3) fail ("bad number of band halvings"); m.bpool = (int) bp;
        for (uint32_t i = 0; i < levels; ++i) { const uint32_t c = u32(); if (c < 1 || c > 1024) fail ("bad channel count"); m.ch.push_back ((int) c); }
        const int L = (int) levels;
        for (int i = 0; i < L; ++i) { m.down.push_back (layer (i > 0 ? m.ch[(size_t) i - 1] : m.planes, m.ch[(size_t) i], 3)); m.down.push_back (layer (m.ch[(size_t) i], m.ch[(size_t) i], 3)); }
        for (int i = L - 2; i >= 0; --i) { m.up.push_back (layer (m.ch[(size_t) i + 1] + m.ch[(size_t) i], m.ch[(size_t) i], 3)); m.up.push_back (layer (m.ch[(size_t) i], m.ch[(size_t) i], 3)); }
        m.head = layer (m.ch[0], 3, 1);
        if (left != 0) fail ("more data than the header gives");
        *this = std::move (m);
    }

    // The answer for T frames (at most a few hundred: a block): x holds the input planes [plane][frame][band]. g: the curve
    // in dB [frame][band]; c: the scores for 'filter sweep' and 'phaser' [frame][2] (against 0 for 'nothing'). The picture is
    // made a whole number of coarsest steps long by repeating its last frame. Stopped on the way: false, g and c unusable.
    bool run (const std::vector<float>& x, int T, std::vector<float>& g, std::vector<float>& c, const Stop& stop = {}) const
    {
        if (! ready()) throw std::runtime_error ("moving-tone model: not loaded");
        if (T < 1 || x.size() != (size_t) planes * (size_t) T * NB) throw std::runtime_error ("moving-tone model: the input is not " + std::to_string (planes) + " planes of T frames");
        const int L = (int) ch.size(), mult = 1 << (L - 1), Tp = (T + mult - 1) / mult * mult;
        Map cur (planes, Tp, NB);
        for (int t = 0; t < Tp; ++t)
            for (int q = 0; q < planes; ++q)
                std::memcpy (cur.at (t, q), &x[((size_t) q * (size_t) T + (size_t) std::min (t, T - 1)) * NB], NB * sizeof (float));
        std::vector<Map> skip;
        for (int i = 0; i < L; ++i)                              // down: half as many frames (and bands, for the first levels), then two layers
        {
            if (stop && stop()) return false;
            if (i > 0) cur = halve (skip.back(), i <= bpool);
            Map mid (ch[(size_t) i], cur.T, cur.B), out (ch[(size_t) i], cur.T, cur.B);
            conv (down[(size_t) (2 * i)], cur, nullptr, mid); conv (down[(size_t) (2 * i + 1)], mid, nullptr, out);
            skip.push_back (std::move (out));
        }
        cur = std::move (skip.back()); skip.pop_back();
        for (int i = L - 2, j = 0; i >= 0; --i, ++j)             // back up: every frame (and band) twice, joined with what the way down left there
        {
            if (stop && stop()) return false;
            const Map wide = twice (cur, i + 1 <= bpool); Map mid (ch[(size_t) i], wide.T, wide.B), out (ch[(size_t) i], wide.T, wide.B);
            conv (up[(size_t) (2 * j)], wide, &skip.back(), mid); skip.pop_back();
            conv (up[(size_t) (2 * j + 1)], mid, nullptr, out); cur = std::move (out);
        }
        g.assign ((size_t) T * NB, 0.0f); c.assign ((size_t) T * 2, 0.0f);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < 3; ++o)
            {
                float acc[NB]; std::fill (acc, acc + NB, head.b[(size_t) o]);
                for (int q = 0; q < cur.C; ++q) { const float w = head.w[(size_t) o * (size_t) cur.C + (size_t) q]; const float* r = cur.at (t, q); for (int h = 0; h < NB; ++h) acc[h] += w * r[h]; }
                if (o == 0) for (int h = 0; h < NB; ++h) g[(size_t) t * NB + (size_t) h] = (float) GS * acc[h];
                else { double m = 0; for (int h = 0; h < NB; ++h) m += acc[h]; c[(size_t) t * 2 + (size_t) (o - 1)] = (float) (m / NB); }
            }
        return true;
    }

private:
    [[noreturn]] static void fail (const char* why) { throw std::runtime_error (std::string ("moving-tone model: ") + why); }

    // Activations [frame][channel][band] with a border of zeros (a frame before and after, a band below and above), so the
    // 3x3 layers need no edge cases.
    struct Map
    {
        int C = 0, T = 0, B = 0, W = 0; std::vector<float> v;
        Map (int c, int t, int b) : C (c), T (t), B (b), W (b + 2), v ((size_t) (t + 2) * (size_t) c * (size_t) (b + 2), 0.0f) {}
        float* row (int r, int c) { return &v[((size_t) r * (size_t) C + (size_t) c) * (size_t) W]; }              // r = frame + 1; points at the zero below band 0
        const float* row (int r, int c) const { return &v[((size_t) r * (size_t) C + (size_t) c) * (size_t) W]; }
        float* at (int t, int c) { return row (t + 1, c) + 1; }
        const float* at (int t, int c) const { return row (t + 1, c) + 1; }
    };
    // y = max (0, 3x3 layer over the channels of a, then those of b).
    static void conv (const Layer& l, const Map& a, const Map* b, Map& y)
    {
        const int B = y.B; std::vector<float> acc ((size_t) B);
        for (int t = 0; t < y.T; ++t)
            for (int o = 0; o < l.out; ++o)
            {
                std::fill (acc.begin(), acc.end(), l.b[(size_t) o]);
                const float* w = &l.w[(size_t) o * (size_t) l.in * 9];
                for (int q = 0; q < l.in; ++q, w += 9)
                {
                    const bool first = q < a.C; const Map& m = first ? a : *b; const int cc = first ? q : q - a.C;
                    const float* r0 = m.row (t, cc); const float* r1 = m.row (t + 1, cc); const float* r2 = m.row (t + 2, cc);
                    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7], w8 = w[8]; float* ac = acc.data();
                    for (int h = 0; h < B; ++h)
                        ac[h] += w0 * r0[h] + w1 * r0[h + 1] + w2 * r0[h + 2] + w3 * r1[h] + w4 * r1[h + 1] + w5 * r1[h + 2] + w6 * r2[h] + w7 * r2[h + 1] + w8 * r2[h + 2];
                }
                float* out = y.at (t, o);
                for (int h = 0; h < B; ++h) out[h] = std::max (0.0f, acc[(size_t) h]);
            }
    }
    static Map halve (const Map& x, bool bands)         // the average of each pair of frames (and of bands)
    {
        Map y (x.C, x.T / 2, bands ? x.B / 2 : x.B);
        for (int t = 0; t < y.T; ++t)
            for (int q = 0; q < x.C; ++q)
            {
                const float* a = x.at (2 * t, q); const float* b = x.at (2 * t + 1, q); float* o = y.at (t, q);
                if (bands) for (int h = 0; h < y.B; ++h) o[h] = 0.25f * (a[2 * h] + a[2 * h + 1] + b[2 * h] + b[2 * h + 1]);
                else for (int h = 0; h < y.B; ++h) o[h] = 0.5f * (a[h] + b[h]);
            }
        return y;
    }
    static Map twice (const Map& x, bool bands)         // every frame (and band) twice
    {
        Map y (x.C, x.T * 2, bands ? x.B * 2 : x.B);
        for (int t = 0; t < y.T; ++t)
            for (int q = 0; q < x.C; ++q)
            {
                const float* a = x.at (t / 2, q); float* o = y.at (t, q);
                if (bands) for (int h = 0; h < y.B; ++h) o[h] = a[h / 2];
                else std::memcpy (o, a, (size_t) y.B * sizeof (float));
            }
        return y;
    }
};

// ---------------------------------------------------------------- what the model says of one channel
struct Read { int T = 0; std::vector<float> g; Vec p; bool ok = false; };          // g[frame * NB + band] dB; p[frame * 3 + kind]: nothing, filter sweep, phaser
inline std::vector<std::pair<int, int>> blocks (int T)
{
    std::vector<std::pair<int, int>> b;
    if (T <= BLOCK) { b.push_back ({ 0, T }); return b; }
    for (int a = 0; a < T - BLOCK; a += BHOP) b.push_back ({ a, a + BLOCK });
    b.push_back ({ T - BLOCK, T }); return b;
}
inline Read read (const Net& net, const Vec& x, const Stop& stop = {})
{
    const Vec LE = picture (x); Read R; R.T = frames (x.size()); const int T = R.T; Vec G ((size_t) T * NB, 0.0), P ((size_t) T * 3, 0.0), W ((size_t) T, 0.0);
    std::vector<float> g, c;
    for (const auto& bl : blocks (T))
    {
        const int a = bl.first, n = bl.second - bl.first;
        if (! net.run (inputs (&LE[(size_t) a * NB], n), n, g, c, stop)) return R;
        for (int t = 0; t < n; ++t)
        {
            const double w = (double) std::min (t + 1, n - t), z1 = c[(size_t) t * 2], z2 = c[(size_t) t * 2 + 1], zm = std::max ({ 0.0, z1, z2 });
            const double e0 = std::exp (-zm), e1 = std::exp (z1 - zm), e2 = std::exp (z2 - zm), sum = e0 + e1 + e2; const size_t i = (size_t) (a + t);
            for (int b = 0; b < NB; ++b) G[i * NB + (size_t) b] += w * (double) g[(size_t) t * NB + (size_t) b];
            P[i * 3] += w * e0 / sum; P[i * 3 + 1] += w * e1 / sum; P[i * 3 + 2] += w * e2 / sum; W[i] += w;
        }
    }
    R.g.resize ((size_t) T * NB); R.p.resize ((size_t) T * 3);
    for (int t = 0; t < T; ++t)
    {
        for (int b = 0; b < NB; ++b) R.g[(size_t) t * NB + (size_t) b] = (float) (G[(size_t) t * NB + (size_t) b] / W[(size_t) t]);
        for (int k = 0; k < 3; ++k) R.p[(size_t) t * 3 + (size_t) k] = P[(size_t) t * 3 + (size_t) k] / W[(size_t) t];
    }
    R.ok = true; return R;
}

// ---------------------------------------------------------------- a whole sound
struct Found
{
    bool found = false; std::string kind;           // "filtersweep" or "phaser"
    double sure = 0;                                // 0 .. 1: the model's say-so over the stretches
    double period = 0;                              // frames: how long the curve of the longest stretch takes to repeat (0 = it does not)
    int T = 0; size_t n = 0;                        // frames and samples of the sound it was found on
    std::vector<std::pair<int, int>> spans;         // the stretches that have the filter: first frame, end frame
    Vec on, gate;                                   // per frame: the say-so; 1 inside the stretches (0.1 s fades at ends inside the sound)
    std::vector<float> raw[2];                      // the model's curve per side, [frame * NB + band] dB
};

// Frames with a moving filter -> stretches: holes shorter than GAP closed, stretches shorter than SHORT dropped.
inline std::vector<std::pair<int, int>> stretches (const std::vector<char>& on)
{
    const int T = (int) on.size(); std::vector<std::pair<int, int>> sp, j, out;
    for (int t = 0; t < T;) { if (on[(size_t) t]) { const int a = t; while (t < T && on[(size_t) t]) ++t; sp.push_back ({ a, t }); } else ++t; }
    for (const auto& s : sp) { if (! j.empty() && s.first - j.back().second < GAP) j.back().second = s.second; else j.push_back (s); }
    for (const auto& s : j) if (s.second - s.first >= std::min (SHORT, T)) out.push_back (s);
    return out;
}
// Weights for the n frames from frame a on of a stretch: a 0.1 s fade at each end that lies inside a sound of T frames.
inline Vec fades (int a, int n, int T)
{
    Vec m ((size_t) n, 1.0);
    for (int i = 0; i < n; ++i)
    {
        double v = 1.0;
        if (a > 0) v = std::min (v, (double) (i + 1) / RAMP);
        if (a + n < T) v = std::min (v, (double) (n - i) / RAMP);
        m[(size_t) i] = std::clamp (v, 0.0, 1.0);
    }
    return m;
}
// The curve's period in frames if it repeats (at least twice within its T frames), else 0. G[frame * NB + band].
inline double periodOf (const Vec& G, int T)
{
    if (T < 48) return 0.0;
    Vec e ((size_t) T, 0.0), cum ((size_t) T + 1, 0.0); double all = 0;
    for (int t = 0; t < T; ++t) { double s = 0; for (int b = 0; b < NB; ++b) { const double v = G[(size_t) t * NB + (size_t) b]; s += v * v; } e[(size_t) t] = s; all += s; cum[(size_t) t + 1] = cum[(size_t) t] + s; }
    if (all <= 0) return 0.0;
    const int top = T / 2; Vec r ((size_t) top + 1, 0.0);
    for (int L = 12; L <= top; ++L)
    {
        double s = 0; const size_t m = (size_t) (T - L) * NB; const double* a = G.data(); const double* b = G.data() + (size_t) L * NB;
        for (size_t i = 0; i < m; ++i) s += a[i] * b[i];
        r[(size_t) L] = s / std::max (std::sqrt (cum[(size_t) (T - L)] * (cum[(size_t) T] - cum[(size_t) L])), 1e-12);
    }
    for (int L = 13; L < top; ++L)
        if (r[(size_t) L] >= PERIODIC && r[(size_t) L] >= r[(size_t) L - 1] && r[(size_t) L] >= r[(size_t) L + 1])
        {
            const double d = r[(size_t) L - 1] - 2 * r[(size_t) L] + r[(size_t) L + 1];
            return L + (d < 0 ? 0.5 * (r[(size_t) L - 1] - r[(size_t) L + 1]) / d : 0.0);
        }
    return 0.0;
}

// What is on a sound. The two sides are read at the same time (once if they are the same). Stopped on the way: nothing found.
inline Found find (const Net& net, const Stereo& y, const Stop& stop = {})
{
    Found F; F.n = y.size(); F.T = frames (y.size()); const int T = F.T;
    const bool same = y.l == y.r; Read rl, rr;
    if (same) { rl = read (net, y.l, stop); rr = rl; }
    else { auto job = std::async (std::launch::async, [&] { return read (net, y.r, stop); }); rl = read (net, y.l, stop); rr = job.get(); }
    if (! rl.ok || ! rr.ok) return F;
    F.on.resize ((size_t) T); for (int t = 0; t < T; ++t) F.on[(size_t) t] = std::max (1.0 - rl.p[(size_t) t * 3], 1.0 - rr.p[(size_t) t * 3]);
    const int k = std::min (SMOOTH, T); std::vector<char> yes ((size_t) T);
    for (int t = 0; t < T; ++t)
    {
        double s = 0; for (int i = 0; i < k; ++i) s += F.on[(size_t) std::clamp (t - k / 2 + i, 0, T - 1)];
        yes[(size_t) t] = s / k >= ON;
    }
    F.spans = stretches (yes); F.gate.assign ((size_t) T, 0.0); F.raw[0] = std::move (rl.g); F.raw[1] = same ? F.raw[0] : std::move (rr.g);
    if (F.spans.empty()) return F;
    for (const auto& s : F.spans) { const Vec m = fades (s.first, s.second - s.first, T); for (int t = s.first; t < s.second; ++t) F.gate[(size_t) t] = m[(size_t) (t - s.first)]; }
    double sum = 0, ps = 0, pp = 0; int cnt = 0;
    for (int t = 0; t < T; ++t) if (F.gate[(size_t) t] > 0) { sum += F.on[(size_t) t]; ++cnt; ps += rl.p[(size_t) t * 3 + 1] + rr.p[(size_t) t * 3 + 1]; pp += rl.p[(size_t) t * 3 + 2] + rr.p[(size_t) t * 3 + 2]; }
    F.found = true; F.sure = sum / cnt; F.kind = ps >= pp ? "filtersweep" : "phaser";
    std::pair<int, int> lg = F.spans[0]; for (const auto& s : F.spans) if (s.second - s.first > lg.second - lg.first) lg = s;
    const int n = lg.second - lg.first; Vec G ((size_t) n * NB);
    for (int t = 0; t < n; ++t) for (int b = 0; b < NB; ++b) { const size_t i = (size_t) (lg.first + t) * NB + (size_t) b; G[(size_t) t * NB + (size_t) b] = (double) F.raw[0][i] + (double) F.raw[1][i]; }
    F.period = periodOf (G, n);
    return F;
}

// The stretches of the sound that have the moving filter, in samples.
inline std::vector<std::pair<size_t, size_t>> where (const Found& F)
{
    std::vector<std::pair<size_t, size_t>> w;
    for (const auto& s : F.spans) w.push_back ({ std::min (F.n, (size_t) s.first * HOP), std::min (F.n, (size_t) s.second * HOP) });
    return w;
}

// Side c's curve for a sound of T2 frames, moving `speed` times as fast: [frame * NB + band] dB. Every stretch keeps its
// place in time; inside it the curve is read faster or slower from the stretch's start, and when it runs out it carries
// on: a curve that repeats goes round its last period, any other goes back and forth. A stretch that reaches the end of
// the sound it was found on runs on to the end of a longer sound in the same way.
inline Vec curve (const Found& F, int c, int T2, double speed = 1.0)
{
    const std::vector<float>& raw = F.raw[c]; const int T = F.T; Vec out ((size_t) T2 * NB, 0.0);
    if (T2 <= T && speed == 1.0)
    {
        for (int t = 0; t < T2; ++t) { const double m = F.gate[(size_t) t]; if (m > 0) for (int b = 0; b < NB; ++b) out[(size_t) t * NB + (size_t) b] = (double) raw[(size_t) t * NB + (size_t) b] * m; }
        return out;
    }
    const double P = F.period;
    for (size_t k = 0; k < F.spans.size(); ++k)
    {
        const int a = F.spans[k].first, b = F.spans[k].second, e = (k + 1 == F.spans.size() && b >= T - 15) ? std::max (b, T2) : std::min (b, T2);
        if (e <= a) continue;
        const int L = std::max (b - 1 - a, 1); const Vec m = fades (a, e - a, T2);
        for (int t = a; t < e; ++t)
        {
            double u = a + (t - a) * speed;
            if (P > 0 && P < L) { if (u > b - 1) u = b - 1 - P + std::fmod (u - (b - 1), P); }
            else { const double v = std::fmod (u - a, 2.0 * L); u = a + (v <= L ? v : 2.0 * L - v); }
            const int i0 = std::clamp ((int) u, 0, std::max (T - 2, 0)), i1 = std::min (i0 + 1, T - 1); const double f = std::clamp (u - i0, 0.0, 1.0), g = m[(size_t) (t - a)];
            for (int h = 0; h < NB; ++h) out[(size_t) t * NB + (size_t) h] = ((double) raw[(size_t) i0 * NB + (size_t) h] * (1 - f) + (double) raw[(size_t) i1 * NB + (size_t) h] * f) * g;
        }
    }
    return out;
}

// One channel with the tone curve G ([frame * NB + band], T frames of the picture) put on: every bin of a short-time
// transform is turned by the curve at its frequency and time (straight lines between the bands and between the frames),
// never up by more than `lift` dB nor down by more than CUT.
inline Vec render (const Vec& x, const Vec& G, int T, double lift = LIFT)
{
    static const room::Transform F (RN);
    static const Vec win = [] { Vec w ((size_t) RN); for (int i = 0; i < RN; ++i) w[(size_t) i] = std::sqrt (0.5 - 0.5 * std::cos (2 * PI * i / RN)); return w; }();
    struct Place { int i0[RN / 2 + 1]; double fr[RN / 2 + 1]; Place() { for (int k = 0; k <= RN / 2; ++k) { const double lb = std::log2 (std::max (k * SR / RN, 1.0) / F0) * BPO; i0[k] = std::clamp ((int) std::floor (lb), 0, NB - 2); fr[k] = std::clamp (lb - i0[k], 0.0, 1.0); } } };
    static const Place pl;
    const long n = (long) x.size(); const int J = (int) (n / RH) + 1 + RN / RH; Vec out ((size_t) n, 0.0), nrm ((size_t) n, 0.0), re ((size_t) RN), im ((size_t) RN), g ((size_t) NB);
    for (int j = 0; j < J; ++j)
    {
        const long s = (long) j * RH - RN;                          // the frame's first sample in x
        double t = ((double) j * RH - RN / 2) / HOP; t = std::min (std::max (t, 0.0), T - 1.0);
        const int k = T > 1 ? std::min ((int) t, T - 2) : 0; const double f = T > 1 ? t - k : 0.0;
        bool flat = true;
        for (int b = 0; b < NB; ++b) { g[(size_t) b] = T > 1 ? G[(size_t) k * NB + (size_t) b] * (1 - f) + G[(size_t) (k + 1) * NB + (size_t) b] * f : G[(size_t) b]; if (g[(size_t) b] != 0.0) flat = false; }
        for (int i = 0; i < RN; ++i) { const long q = s + i; re[(size_t) i] = q >= 0 && q < n ? x[(size_t) q] * win[(size_t) i] : 0.0; im[(size_t) i] = 0.0; }
        if (! flat)
        {
            F.run (re, im);
            for (int b = 0; b <= RN / 2; ++b)
            {
                const double db = std::clamp (g[(size_t) pl.i0[b]] * (1 - pl.fr[b]) + g[(size_t) pl.i0[b] + 1] * pl.fr[b], -CUT, lift), a = std::pow (10.0, db / 20.0);
                re[(size_t) b] *= a; im[(size_t) b] *= a;
                if (b > 0 && b < RN / 2) { re[(size_t) (RN - b)] *= a; im[(size_t) (RN - b)] *= a; }
            }
            F.back (re, im);
        }
        for (int i = 0; i < RN; ++i) { const long q = s + i; if (q >= 0 && q < n) { out[(size_t) q] += re[(size_t) i] * win[(size_t) i]; nrm[(size_t) q] += win[(size_t) i] * win[(size_t) i]; } }
    }
    for (long i = 0; i < n; ++i) out[(size_t) i] /= std::max (nrm[(size_t) i], 1e-9);
    return out;
}

// y with the found curve scaled by s. own: y is the sound the curve was found on (s = -1 takes it off; with another speed
// the found curve goes out and the new one, 1 + s times, comes in). Otherwise s times the curve goes onto y.
inline Stereo apply (const Stereo& y, const Found& F, double s, double speed = 1.0, bool own = true, double lift = LIFT)
{
    if (! F.found) return y;
    const int T2 = frames (y.size()); Stereo out;
    auto side = [&] (const Vec& x, int c)
    {
        Vec g = curve (F, c, T2, speed);
        if (own && speed != 1.0) { const Vec was = curve (F, c, T2); for (size_t i = 0; i < g.size(); ++i) g[i] = (1.0 + s) * g[i] - was[i]; }
        else for (auto& v : g) v *= s;
        return render (x, g, T2, lift);
    };
    auto job = std::async (std::launch::async, [&] { return side (y.r, 1); });
    out.l = side (y.l, 0); out.r = job.get();
    return out;
}

} // namespace moving
} // namespace dr
