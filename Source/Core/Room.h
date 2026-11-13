#pragma once
// Room or sound: says for every band and moment of a sound how far it has to come down to leave
// the direct sound, and renders the sound with its room taken out, left alone or doubled.
// Port of the prototype's common.py (what is computed from a sound), net.py (the model) and
// render.py (from band gains to sound). 48 kHz stereo of any length. Plain C++17, no JUCE.
//
//   room::Picture P = room::analyse (y);          what the model sees
//   std::vector<float> g = net.gains (P);         its answer: dB per frame and band, 0 .. -30
//   Stereo out = room::render (y, g, amount);     0 = room out, 1 = as it is, 2 = room twice as loud
//
// (The app includes RoomEngine.h, which does all of this in one call and goes on from there.)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Dsp.h"

namespace dr
{
namespace room
{
constexpr int NF = 2048, HOP = 512, BINS = NF / 2 + 1;     // 43 ms windows, a new one every 10.7 ms
constexpr int NB = 41;                                      // bands
constexpr int K64 = 6, K256 = 24;                           // 64 and 256 ms in frames: the two averaging times of the unshared share
constexpr double DT = HOP / SR;                             // seconds from one frame to the next

// A long job asks `stop` now and then and gives up when the answer is true: what it then returns is not
// complete (a sound comes back empty). `progress` hears how far the job is, 0 .. 1. Both can be left out.
using Stop = std::function<bool()>;
using Progress = std::function<void (double)>;

// Frames of a sound of n samples: a window of silence comes in front and behind.
inline int frames (size_t n) { return (int) (n / HOP) + 6; }
// Square root of a Hann window: used going in and coming out, so the overlapped windows add up to a constant.
inline Vec window() { Vec w = hannPeriodic (NF); for (auto& v : w) v = std::sqrt (v); return w; }
// Frame t of a sound, windowed: the NF samples that end at sample t * HOP (silence outside the sound). Both
// channels ride through one transform, left as the real and right as the imaginary part. False: all silence.
inline bool frame (const Stereo& y, int t, const Vec& win, Vec& re, Vec& im)
{
    const long long n = (long long) y.size(), p0 = (long long) t * HOP - NF; bool any = false;
    for (int j = 0; j < NF; ++j)
    {
        const long long p = p0 + j; const bool in = p >= 0 && p < n; const size_t i = (size_t) j;
        re[i] = in ? y.l[(size_t) p] * win[i] : 0.0; im[i] = in ? y.r[(size_t) p] * win[i] : 0.0;
        any = any || re[i] != 0.0 || im[i] != 0.0;
    }
    return any;
}
// One bin of both channels out of the joint transform, left = (Z[k] + conj Z[-k]) / 2 and right = (Z[k] - conj Z[-k]) / 2i,
// and back into it (the bin and its mirror image).
struct Bin { double lr, li, rr, ri; };
inline Bin bin (const Vec& re, const Vec& im, int k)
{
    const size_t a = (size_t) k, m = (size_t) ((NF - k) % NF);
    return { 0.5 * (re[a] + re[m]), 0.5 * (im[a] - im[m]), 0.5 * (im[a] + im[m]), 0.5 * (re[m] - re[a]) };
}
inline void setBin (Vec& re, Vec& im, int k, double lr, double li, double rr, double ri)
{
    const size_t a = (size_t) k, m = (size_t) ((NF - k) % NF);
    re[m] = lr + ri; im[m] = rr - li; re[a] = lr - ri; im[a] = li + rr;
}
// The way back: a frame (left in re, right in im) is windowed once more and added at its place ...
inline void overlap (Stereo& out, int t, const Vec& win, const Vec& re, const Vec& im)
{
    const long long n = (long long) out.size(), p0 = (long long) t * HOP - NF;
    for (long long j = std::max (0LL, -p0); j < std::min ((long long) NF, n - p0); ++j)
    {
        const size_t p = (size_t) (p0 + j), i = (size_t) j;
        out.l[p] += re[i] * win[i]; out.r[p] += im[i] * win[i];
    }
}
// ... and when all frames are in, every sample (it lies under four windows) is divided by the sum of their squares.
inline void overlapDone (Stereo& out, const Vec& win)
{
    Vec norm (HOP, 0.0);
    for (int r = 0; r < HOP; ++r) for (int j = NF - HOP + r; j >= 0; j -= HOP) norm[(size_t) r] += win[(size_t) j] * win[(size_t) j];
    for (size_t p = 0; p < out.size(); ++p) { const double d = std::max (norm[p % HOP], 1e-9); out.l[p] /= d; out.r[p] /= d; }
}

// The transform of dr::fft for n points (a power of two) with its sines and cosines worked out once: run() and
// back(). For a long convolution the two halves of the work can be used apart, which saves the reordering:
// spread() leaves the bins in bit-reversed order and gather() takes them in that order.
struct Transform
{
    int n; Vec c, s; std::vector<int> rev;      // per stage its own run of cosines and sines (the stage with `half` points apart starts at half - 1); the bit-reversed order
    explicit Transform (int size = NF, bool ordered = true) : n (size)      // not ordered: only spread() and gather() will be used
    {
        if (ordered) { rev.assign ((size_t) n, 0); for (int i = 1, j = 0; i < n; ++i) { int bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; rev[(size_t) i] = j; } }
        const int top = n / 2; c.assign ((size_t) std::max (n - 1, 0), 1.0); s.assign (c.size(), 0.0);
        for (int k = 0; k < top; ++k) { c[(size_t) (top - 1 + k)] = std::cos (PI * k / top); s[(size_t) (top - 1 + k)] = -std::sin (PI * k / top); }
        for (int half = top / 2; half >= 1; half /= 2)          // the shorter stages use every second value of the next longer one
            for (int k = 0; k < half; ++k) { c[(size_t) (half - 1 + k)] = c[(size_t) (2 * half - 1 + 2 * k)]; s[(size_t) (half - 1 + k)] = s[(size_t) (2 * half - 1 + 2 * k)]; }
    }
    void run (Vec& re, Vec& im) const
    {
        for (int i = 0; i < n; ++i) { const int j = rev[(size_t) i]; if (i < j) { std::swap (re[(size_t) i], re[(size_t) j]); std::swap (im[(size_t) i], im[(size_t) j]); } }
        gather (re.data(), im.data());
    }
    void back (Vec& re, Vec& im) const { run (im, re); for (auto& v : re) v /= n; for (auto& v : im) v /= n; }   // the way back is the same transform with real and imaginary part swapped
    // Bins in bit-reversed order in, plain order out. The short stages are done stretch by stretch of `block`
    // points (a stretch then stays in the processor's fast memory), the long stages come after.
    void gather (double* re, double* im) const
    {
        const int block = std::min (n, 1 << 12);
        for (int i = 0; i < n; i += block) for (int half = 1; half < block; half *= 2) stage (re, im, i, i + block, half, false);
        for (int half = block; half < n; half *= 2) stage (re, im, 0, n, half, false);
    }
    // Plain order in, bins in bit-reversed order out: the same work back to front.
    void spread (double* re, double* im) const
    {
        const int block = std::min (n, 1 << 12);
        for (int half = n / 2; half >= block; half /= 2) stage (re, im, 0, n, half, true);
        for (int i = 0; i < n; i += block) for (int half = block / 2; half >= 1; half /= 2) stage (re, im, i, i + block, half, true);
    }
private:
    void stage (double* re, double* im, int from, int to, int half, bool spreading) const      // the pairs `half` apart among the points from .. to
    {
        const double* wc = &c[(size_t) half - 1]; const double* ws = &s[(size_t) half - 1];
        for (int i = from; i < to; i += 2 * half)
        {
            double* ur = re + i; double* vr = ur + half; double* ui = im + i; double* vi = ui + half;
            if (spreading)
                for (int k = 0; k < half; ++k)
                {
                    const double xr = ur[k] - vr[k], xi = ui[k] - vi[k];
                    ur[k] += vr[k]; ui[k] += vi[k]; vr[k] = xr * wc[k] - xi * ws[k]; vi[k] = xr * ws[k] + xi * wc[k];
                }
            else
                for (int k = 0; k < half; ++k)
                {
                    const double tr = vr[k] * wc[k] - vi[k] * ws[k], ti = vr[k] * ws[k] + vi[k] * wc[k];
                    vr[k] = ur[k] - tr; vi[k] = ui[k] - ti; ur[k] += tr; ui[k] += ti;
                }
        }
    }
};

// The bands: 7 of two FFT bins each (47..375 Hz), then 34 a sixth of an octave wide, up to 19 kHz.
struct Bands
{
    int edge[NB + 1];           // band b is the bins edge[b] .. edge[b + 1] - 1
    double fc[NB];              // centre, Hz
    Bands()
    {
        for (int i = 0; i < 8; ++i) edge[i] = 2 + 2 * i;
        for (int k = 1; k <= NB - 7; ++k) edge[7 + k] = (int) std::lround (375.0 * std::pow (2.0, k / 6.0) / (SR / NF));
        for (int b = 0; b < NB; ++b) fc[b] = std::sqrt ((double) (edge[b] * edge[b + 1])) * SR / NF;
    }
};

// ---------------------------------------------------------------- what the model sees
// Per frame and band, in dB: LE the level; U64 and U256 how much of the power the two channels
// do not share (0 = nothing shared, -40 = one source at a fixed place), averaged over 64 and 256 ms.
struct Picture
{
    int frames = 0;
    Vec LE, U64, U256;          // [frame * NB + band]
    double top() const { return LE.empty() ? 0.0 : *std::max_element (LE.begin(), LE.end()); }   // the loudest cell
};

// Stopped on the way: the picture comes back with no frames.
inline Picture analyse (const Stereo& y, const Stop& stop = {}, const Progress& progress = {})
{
    const Bands B; const Transform F; const Vec win = window(); const int T = frames (y.size());
    Picture P; P.frames = T; P.LE.assign ((size_t) T * NB, 10.0 * std::log10 (1e-12)); P.U64 = P.LE; P.U256 = P.LE;     // the level of silence
    // Per bin: power left, power right and the product left x conj (right) (real, imaginary), for the last
    // K256 frames. Only the bins of the bands and two beyond (the share is averaged over +-2 bins).
    const int bins = std::min (BINS, B.edge[NB] + 2); const size_t row = (size_t) 4 * (size_t) bins;
    Vec ring ((size_t) K256 * row, 0.0), sum (row), re (NF), im (NF);

    // The unshared share for the frame in the middle of the last K (an even K has its extra frame on
    // the side of the past): the four numbers are averaged over the K frames and +-2 bins, with zeros
    // outside the picture; twice the smaller eigenvalue of that 2x2 matrix is the power not shared.
    // Plain sums on purpose: silence gives exactly "nothing unshared", whatever came before.
    auto unshared = [&] (int K, int t, Vec& U)
    {
        const int i = t - (K - 1) / 2; if (i < 0 || i >= T) return;
        std::fill (sum.begin(), sum.end(), 0.0);
        for (int j = std::max (0, t - K + 1); j <= t; ++j)
        {
            const double* q = &ring[(size_t) (j % K256) * row];
            for (size_t k = 0; k < row; ++k) sum[k] += q[k];
        }
        const double sc = 1.0 / (5.0 * K);
        for (int b = 0; b < NB; ++b)
        {
            double un = 0.0, all = 0.0;
            for (int f = B.edge[b]; f < B.edge[b + 1]; ++f)
            {
                double a[4] = { 0.0, 0.0, 0.0, 0.0 };
                for (int g = std::max (0, f - 2); g <= std::min (bins - 1, f + 2); ++g)
                    for (int q = 0; q < 4; ++q) a[q] += sum[(size_t) q * (size_t) bins + (size_t) g];
                const double pl = a[0] * sc, pr = a[1] * sc, cr = a[2] * sc, ci = a[3] * sc;
                un += std::max ((pl + pr) - std::sqrt (std::max ((pl - pr) * (pl - pr) + 4.0 * (cr * cr + ci * ci), 0.0)), 0.0);
                all += pl + pr;
            }
            U[(size_t) i * NB + (size_t) b] = 10.0 * std::log10 (un / std::max (all, 1e-30) + 1e-4);
        }
    };

    const int last = T + (K256 - 1) / 2;
    for (int t = 0; t < last; ++t)
    {
        if ((t & 255) == 0)
        {
            if (stop && stop()) return Picture();
            if (progress) progress ((double) t / last);
        }
        double* q = &ring[(size_t) (t % K256) * row]; std::fill (q, q + row, 0.0);
        if (t < T && frame (y, t, win, re, im))
        {
            F.run (re, im);
            for (int k = 0; k < bins; ++k)       // left = (Z[k] + conj Z[-k]) / 2, right = (Z[k] - conj Z[-k]) / 2i
            {
                const size_t a = (size_t) k, m = (size_t) ((NF - k) % NF);
                const double lr = 0.5 * (re[a] + re[m]), li = 0.5 * (im[a] - im[m]), rr = 0.5 * (im[a] + im[m]), ri = 0.5 * (re[m] - re[a]);
                q[a] = lr * lr + li * li; q[(size_t) bins + a] = rr * rr + ri * ri;
                q[(size_t) (2 * bins) + a] = lr * rr + li * ri; q[(size_t) (3 * bins) + a] = li * rr - lr * ri;
            }
            for (int b = 0; b < NB; ++b)
            {
                double e = 0.0; for (int f = B.edge[b]; f < B.edge[b + 1]; ++f) e += q[(size_t) f] + q[(size_t) (bins + f)];
                P.LE[(size_t) t * NB + (size_t) b] = 10.0 * std::log10 (e + 1e-12);
            }
        }
        unshared (K64, t, P.U64); unshared (K256, t, P.U256);
    }
    return P;
}

// The model's four input planes [plane][frame][band]: the level under the sound's loudest cell (70 dB
// range), the two unshared shares, and the band's place from bottom (-1) to top (+1). Worked out in
// single precision from single-precision pictures, as the prototype does for training and in use.
inline std::vector<float> inputs (const Picture& P)
{
    const size_t n = (size_t) P.frames * NB; std::vector<float> x (4 * n);
    const float top = (float) P.top();
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = std::clamp (((float) P.LE[i] - top) / 20.0f, -3.5f, 0.0f) + 1.75f;
        x[n + i] = std::clamp ((float) P.U64[i], -40.0f, 0.0f) / 20.0f + 1.0f;
        x[2 * n + i] = std::clamp ((float) P.U256[i], -40.0f, 0.0f) / 20.0f + 1.0f;
        x[3 * n + i] = (float) (-1.0 + (double) (i % NB) * (2.0 / (NB - 1)));
    }
    return x;
}

// ---------------------------------------------------------------- the model
// A small network of plain 3x3 layers (over time and band) at several time scales: the picture is
// halved in time on the way down and brought back up with the finer levels joined in, so long
// tails are seen at the coarse levels and onsets at the fine ones. Sizes come from the file.
struct Net
{
    struct Layer { int in = 0, out = 0; std::vector<float> w, b; };      // w[out][in][time][band]
    int planes = 4;                     // input planes: 4 (see inputs()), or 7 for a second pass (those 4 and the 3 of secondLook())
    std::vector<int> ch;                // channels per level, finest first
    float deepest = 30.0f;              // the answer runs from 0 down to minus this many dB
    std::vector<Layer> down, up;        // two layers per level; `up` starts one above the coarsest level
    Layer head;                         // 1x1: the finest level's channels -> one number per cell

    bool ready() const { return ! ch.empty(); }

    // Reads the file written by Tools/export_room_model.py from memory. Throws std::runtime_error, and
    // stays as it was, when the block is not a complete model of the sizes its own header gives.
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

        char magic[4]; take (magic, 4); if (std::memcmp (magic, "DRRM", 4) != 0) fail ("not a room model file");
        if (u32() != 1) fail ("unknown format version");
        Net m; const uint32_t seen = u32(); if (seen != 4 && seen != 7) fail ("made for a different number of input planes"); m.planes = (int) seen;
        take (&m.deepest, 4); if (! (m.deepest > 0.0f && m.deepest <= 120.0f)) fail ("bad gain range");
        const uint32_t levels = u32(); if (levels < 1 || levels > 12) fail ("bad number of levels");
        for (uint32_t i = 0; i < levels; ++i) { const uint32_t c = u32(); if (c < 1 || c > 1024) fail ("bad channel count"); m.ch.push_back ((int) c); }
        const int L = (int) levels;
        for (int i = 0; i < L; ++i) { m.down.push_back (layer (i > 0 ? m.ch[(size_t) i - 1] : m.planes, m.ch[(size_t) i], 3)); m.down.push_back (layer (m.ch[(size_t) i], m.ch[(size_t) i], 3)); }
        for (int i = L - 2; i >= 0; --i) { m.up.push_back (layer (m.ch[(size_t) i + 1] + m.ch[(size_t) i], m.ch[(size_t) i], 3)); m.up.push_back (layer (m.ch[(size_t) i], m.ch[(size_t) i], 3)); }
        m.head = layer (m.ch[0], 1, 1);
        if (left != 0) fail ("more data than the header gives");
        *this = std::move (m);
    }

    // The answer for a whole picture: x holds the input planes ([plane][frame][band], T frames); returns
    // the gain in dB [frame][band]. The picture is made a whole number of coarsest steps long by repeating
    // its last frame. A long picture is run a piece at a time (`piece` frames), each with enough of
    // its neighbours on both sides that the answer is the same as in one go: the working memory stays
    // where it is for 45 s of sound (about 40 MB with the present sizes) however long the sound is.
    // Stopped on the way (asked before every level of every piece): the answer comes back empty.
    std::vector<float> run (const std::vector<float>& x, int T, int piece = 4096, const Stop& stop = {}, const Progress& progress = {}) const
    {
        if (! ready()) throw std::runtime_error ("room model: not loaded");
        if (T < 1 || x.size() != (size_t) planes * (size_t) T * NB) throw std::runtime_error ("room model: the input is not " + std::to_string (planes) + " planes of T frames");
        const int L = (int) ch.size(), mult = 1 << (L - 1), Tp = (T + mult - 1) / mult * mult;
        // The zeros beyond a piece's cut end spoil one more frame per layer, and every step back up doubles the
        // stretch: less than 6 coarsest steps in all, so that many frames of the neighbours are run along.
        const int margin = 6 * mult, core = std::max (1, piece / mult) * mult;
        std::vector<float> g ((size_t) T * NB);
        for (int a = 0; a < Tp; a += core)
        {
            const int b = std::min (Tp, a + core), a0 = std::max (0, a - margin), b0 = std::min (Tp, b + margin);
            int done = 0;                                           // levels behind us in this piece, of 2 L - 1
            auto next = [&] { if (progress) progress (((double) a + (double) (b - a) * done / (2 * L - 1)) / Tp); ++done; return ! (stop && stop()); };
            Map cur (planes, b0 - a0);
            for (int t = a0; t < b0; ++t)
                for (int c = 0; c < planes; ++c)
                    std::memcpy (cur.at (t - a0 + 1, c) + 1, &x[((size_t) c * (size_t) T + (size_t) std::min (t, T - 1)) * NB], NB * sizeof (float));
            std::vector<Map> skip;                                  // per level, what the way down left there
            for (int i = 0; i < L; ++i)                             // down: two layers per level, then half as many frames
            {
                if (! next()) return {};
                if (i > 0) cur = halve (skip.back());
                Map mid (ch[(size_t) i], cur.T), out (ch[(size_t) i], cur.T);
                conv (down[(size_t) (2 * i)], cur, false, nullptr, mid); conv (down[(size_t) (2 * i + 1)], mid, false, nullptr, out);
                skip.push_back (std::move (out));
            }
            cur = std::move (skip.back()); skip.pop_back();
            for (int i = L - 2, j = 0; i >= 0; --i, ++j)             // back up: the coarser level (each frame twice) joined with what the way down left
            {
                if (! next()) return {};
                Map mid (ch[(size_t) i], skip.back().T), out (ch[(size_t) i], skip.back().T);
                conv (up[(size_t) (2 * j)], cur, true, &skip.back(), mid); skip.pop_back();
                conv (up[(size_t) (2 * j + 1)], mid, false, nullptr, out);
                cur = std::move (out);
            }
            for (int t = a; t < std::min (b, T); ++t)
                for (int h = 0; h < NB; ++h)
                {
                    float z = head.b[0];
                    for (int c = 0; c < cur.C; ++c) z += head.w[(size_t) c] * cur.at (t - a0 + 1, c)[h + 1];
                    g[(size_t) t * NB + (size_t) h] = (float) (-(double) deepest / (1.0 + std::exp (-(double) z)));
                }
        }
        if (progress) progress (1.0);
        return g;
    }
    std::vector<float> gains (const Picture& P) const { return run (inputs (P), P.frames); }       // for a model of 4 input planes

private:
    [[noreturn]] static void fail (const char* why) { throw std::runtime_error (std::string ("room model: ") + why); }

    // Activations [frame][channel][band] with a border of zeros (a frame before and after, a band
    // below and above), so the 3x3 layers need no edge cases.
    static constexpr int W = NB + 2;
    struct Map
    {
        int C = 0, T = 0; std::vector<float> v;
        Map (int c, int t) : C (c), T (t), v ((size_t) (t + 2) * (size_t) c * W, 0.0f) {}
        float* at (int r, int c) { return &v[((size_t) r * (size_t) C + (size_t) c) * W]; }                // r = frame + 1; points at the zero below band 0
        const float* at (int r, int c) const { return &v[((size_t) r * (size_t) C + (size_t) c) * W]; }
    };
    // y = max (0, 3x3 layer over the channels of a, then those of b). half: a has half as many frames as y, each standing for two.
    static void conv (const Layer& l, const Map& a, bool half, const Map* b, Map& y)
    {
        for (int t = 0; t < y.T; ++t)
            for (int o = 0; o < l.out; ++o)
            {
                float acc[NB]; std::fill (acc, acc + NB, l.b[(size_t) o]);
                const float* w = &l.w[(size_t) o * (size_t) l.in * 9];
                for (int c = 0; c < l.in; ++c, w += 9)
                {
                    const bool first = c < a.C; const Map& m = first ? a : *b; const int cc = first ? c : c - a.C;
                    const bool h2 = first && half;                  // frames t-1, t, t+1 of y are rows t, t+1, t+2; at half rate rows (t+1)/2, (t+2)/2, (t+3)/2
                    const float* r0 = m.at (h2 ? (t + 1) >> 1 : t, cc); const float* r1 = m.at (h2 ? (t + 2) >> 1 : t + 1, cc); const float* r2 = m.at (h2 ? (t + 3) >> 1 : t + 2, cc);
                    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7], w8 = w[8];
                    for (int h = 0; h < NB; ++h)
                        acc[h] += w0 * r0[h] + w1 * r0[h + 1] + w2 * r0[h + 2] + w3 * r1[h] + w4 * r1[h + 1] + w5 * r1[h + 2] + w6 * r2[h] + w7 * r2[h + 1] + w8 * r2[h + 2];
                }
                float* out = y.at (t + 1, o) + 1;
                for (int h = 0; h < NB; ++h) out[h] = std::max (0.0f, acc[h]);
            }
    }
    static Map halve (const Map& x)         // the average of each pair of frames
    {
        Map y (x.C, x.T / 2); const size_t n = (size_t) x.C * W;
        for (int t = 0; t < y.T; ++t)
        {
            const float* a = x.at (2 * t + 1, 0); const float* b = x.at (2 * t + 2, 0); float* o = y.at (t + 1, 0);
            for (size_t i = 0; i < n; ++i) o[i] = 0.5f * (a[i] + b[i]);
        }
        return y;
    }
};

// ---------------------------------------------------------------- from band gains to sound
// Band values spread over the FFT bins: straight lines between band centres along log frequency,
// flat below the first and above the last centre (bins under 20 Hz count as 20 Hz).
struct Spread
{
    int lo[BINS]; double w[BINS];
    Spread()
    {
        const Bands B; double lc[NB]; for (int b = 0; b < NB; ++b) lc[b] = std::log (B.fc[b]);
        for (int k = 0; k < BINS; ++k)
        {
            const double lf = std::log (std::max ((double) k * SR / NF, 20.0));
            int i = 0; while (i < NB && lc[i] < lf) ++i;
            lo[k] = std::clamp (i - 1, 0, NB - 2); w[k] = std::clamp ((lf - lc[lo[k]]) / (lc[lo[k] + 1] - lc[lo[k]]), 0.0, 1.0);
        }
    }
    double at (const float* band, int k) const { return (double) band[lo[k]] * (1.0 - w[k]) + (double) band[lo[k] + 1] * w[k]; }
    double at (const double* band, int k) const { return band[lo[k]] * (1.0 - w[k]) + band[lo[k] + 1] * w[k]; }
};

// The sound with its room at amount s, from the model's answer gDb ([frame][band], dB): 0 = room out
// (the full answer), 1 = untouched, 2 = the room twice as loud. Up to 1 a part of the answer is
// applied (in dB); above 1 the room's share of each bin's power, 1 - 10^(g/10), is made s*s times as strong.
// Stopped on the way: comes back empty.
inline Stereo render (const Stereo& y, const std::vector<float>& gDb, double s, const Stop& stop = {})
{
    const int T = frames (y.size());
    if (gDb.size() != (size_t) T * NB) throw std::runtime_error ("room: the gains do not belong to this sound");
    if (s == 1.0) return y;
    const Spread S; const Transform F; const Vec win = window(); const double ln10 = std::log (10.0);
    Stereo out; out.l.assign (y.size(), 0.0); out.r.assign (y.size(), 0.0);
    Vec re (NF), im (NF), g (BINS);
    for (int t = 0; t < T; ++t)
    {
        if ((t & 255) == 0 && stop && stop()) return {};
        if (! frame (y, t, win, re, im)) continue;
        F.run (re, im);
        const float* band = &gDb[(size_t) t * NB];
        if (s <= 1.0) for (int k = 0; k < BINS; ++k) g[(size_t) k] = std::exp (S.at (band, k) * (1.0 - s) * ln10 / 20.0);
        else for (int k = 0; k < BINS; ++k) g[(size_t) k] = std::sqrt (1.0 + (s * s - 1.0) * (1.0 - std::exp (S.at (band, k) * ln10 / 10.0)));
        for (int k = 0; k < BINS; ++k)          // the same real gain on a bin and its mirror image: left and right stay apart
        {
            const size_t a = (size_t) k, m = (size_t) ((NF - k) % NF);
            re[a] *= g[a]; im[a] *= g[a];
            if (m != a) { re[m] *= g[a]; im[m] *= g[a]; }
        }
        F.back (re, im); overlap (out, t, win, re, im);
    }
    overlapDone (out, win);
    return out;
}

// As render(), with the room part of every band and moment made F times as strong (F: [frame * NB + band], power,
// 1 = as it is) and its width changed (0 .. 2: the room part of the side signal is scaled, 1 = as it is). Per bin
// the power is split by the answer into a direct share G and a room share 1 - G; the result holds
// G + A F (1 - G) of it, where A is what the amount alone leaves of the room share (so with F = 1 and width 1
// this is render()). Stopped on the way: comes back empty.
inline Stereo reshaped (const Stereo& y, const std::vector<float>& gDb, double s, const Vec& F, double width = 1.0, const Stop& stop = {})
{
    const int T = frames (y.size());
    if (gDb.size() != (size_t) T * NB || F.size() != gDb.size()) throw std::runtime_error ("room: the gains do not belong to this sound");
    const Spread S; const Transform tr; const Vec win = window(); const double ln10 = std::log (10.0); const bool wide = ! (std::abs (width - 1.0) < 1e-9);
    Stereo out; out.l.assign (y.size(), 0.0); out.r.assign (y.size(), 0.0);
    Vec re (NF), im (NF), gm (BINS), gs (BINS); double fdb[NB];
    for (int t = 0; t < T; ++t)
    {
        if ((t & 255) == 0 && stop && stop()) return {};
        if (! frame (y, t, win, re, im)) continue;
        tr.run (re, im);
        const float* band = &gDb[(size_t) t * NB];
        for (int b = 0; b < NB; ++b) fdb[b] = 10.0 * std::log10 (std::max (F[(size_t) t * NB + (size_t) b], 1e-6));
        for (int k = 0; k < BINS; ++k)
        {
            const double gb = S.at (band, k), G = std::exp (gb * ln10 / 10.0), room = 1.0 - G, more = std::exp (S.at (fdb, k) * ln10 / 10.0);
            const double A = s > 1.0 ? s * s : room > 1e-9 ? (std::exp (gb * (1.0 - s) * ln10 / 10.0) - G) / std::max (room, 1e-9) : s;
            gm[(size_t) k] = std::sqrt (G + A * more * room); gs[(size_t) k] = wide ? std::sqrt (G + A * more * room * width * width) : gm[(size_t) k];
        }
        for (int k = 0; k < BINS; ++k)          // left + right takes the one gain, left - right the other
        {
            const Bin x = bin (re, im, k); const double p = 0.5 * (gm[(size_t) k] + gs[(size_t) k]), q = 0.5 * (gm[(size_t) k] - gs[(size_t) k]);
            setBin (re, im, k, p * x.lr + q * x.rr, p * x.li + q * x.ri, q * x.lr + p * x.rr, q * x.li + p * x.ri);
        }
        tr.back (re, im); overlap (out, t, win, re, im);
    }
    overlapDone (out, win);
    return out;
}

} // namespace room
} // namespace dr
