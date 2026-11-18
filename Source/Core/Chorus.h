#pragma once
// The chorus engine (CHORUS): copies of the sound whose delay moves round and round, 3 to 45 ms behind it.
//   find()      reads the voices of a chorus off one sound: for each side every voice's delay over time (how late it is,
//               how far and how fast it swings, as a sine or a triangle) and its loudness in eight frequency ranges; how sure
//   apply()     on the sound the chorus was found on: s = -1 takes the voices out, 0 leaves the sound as it is, +1 makes
//               the chorus twice as strong. On another sound: s = 1 puts the chorus on as found, 2 twice as strong.
//               Speed and depth: the voices move faster / further (1 = as found).
//   where()     the stretches of the sound that have the chorus (for the line under the waveform)
// How: a copy whose delay grows by r per second is the sound played r slower. Every frame of the sound (85 ms, one every
// 21 ms, at 24 kHz) is therefore compared with the sound itself stretched by -6 .. 6 % about the frame's middle and read
// up to 45 ms earlier: against the right stretch a moving copy is a sharp peak at its delay (in a plain comparison it
// smears over the delays it passes through). The eight strongest peaks of every frame (delay, stretch, height) are the
// material: peaks that follow one another as a moving copy's would are linked into chains, the rates at which the chains'
// peaks come round again are tried (one round of the movement must look like the others), and per rate a sine or triangle
// curve is laid through each chain (delay and stretch together); the curves that collect the most peaks are the voices.
// A voice's loudness needs no dry sound: with the voices taken out by the loudness so far (x[n] = y[n] - sum of
// g x(n - d[n]), range by range), what is still left of a voice is the part of the result that is the result itself read
// along the voice's curve; that is added, and again, six times. A real voice is gone from the result (nothing is found
// along its curve any more); a curve laid through the sound's own pitch or vibrato is not, and is dropped.
// What is no voice: the sound's own pitch drifting looks like a copy too. Two tests tell them apart before any loudness
// is measured: a real voice shows on the whole of its round (the pitch of a drum or of an engine lies along one flank of a
// curve at best), and it stands alone (the sound's own pitch shows at every multiple of its period: its peaks have twins).
// A single copy may be LOUDER than the sound itself. Taken out as above, the louder part would be kept and the sound
// removed in its place. Which of the two is the louder is read off four things (which comes first at sharp events, the
// weaker or the stronger; and, with the copy taken out both ways, which result is freer of it, more peaked, more uneven
// in time); a louder copy is taken out from the end of the sound backwards (backwards()).
// A sound longer than 12 s is read in pieces of 12 s, one every 6 s, each on its own; their results are faded into one
// another. Onto another sound go the voices of the piece that shows them most clearly, for the whole of that sound.
// A port of the Python prototype Research/chorus (Tools/chorus_parity.py). 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Room.h"      // dr::Vec, dr::Stereo (Dsp.h) and room::Transform

namespace dr
{
namespace chorus
{
using Stop = std::function<bool()>;                     // asked now and then while measuring: true = give up (the answer is then not used)
constexpr int DEC = 2;                                  // the voices are looked for at 24 kHz
constexpr double FS = SR / DEC;
constexpr int W = 2048, HOP = 512;                      // frames of 85 ms every 21 ms (longer frames smear a voice that turns quickly)
constexpr double FPS = FS / HOP;
constexpr int TMIN = 72, TMAX = 1080;                   // delays looked at: 3 to 45 ms, in samples at FS
constexpr int NRHO = 121;                               // the copy's delay may grow or shrink by up to 6 % of the time, looked at in steps of 0.1 %
constexpr double RSTEP = 0.001, RHO0 = -0.06;
constexpr double LEVEL = 1e-6;                          // frequencies this far under the frame's strongest (as power) count less and less
constexpr int NPK = 8;                                  // peaks kept per frame
constexpr double TOL = 1.5, RTOL = 0.004;               // a peak this near a curve (samples at FS, and stretch) belongs to it
constexpr double SLOPE = 0.5 * TOL / RTOL;              // how a miss in stretch counts against a miss in delay
constexpr int NFFT = 4096, LREF = W + TMAX, NL = TMAX - TMIN, BINS = NFFT / 2 + 1, FRONT = 2255;      // each frame is compared with a longer piece of the stretched sound, reaching TMAX further back
constexpr int NB = 8;                                   // frequency ranges a voice's loudness is measured in
constexpr double EDGES[NB - 1] = { 150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0 };
constexpr int BN = 4096, BH = 1024;                     // the transform the ranges are cut with
constexpr int SN = 2048, SH = 1024, SB = SN / 2 + 1;    // the transform the loudness is measured on
constexpr double MOST = 0.9;                            // the voices' loudness in one range may add up to this at most (above 1 the taking out runs away)
constexpr double NEED = 0.25, LOUD = 0.08;              // a voice must show in this share of the frames with peaks, and be this loud between 600 Hz and 4.8 kHz
constexpr double LEAST = 0.1;                           // a real voice shows on the whole of its round: in every eighth of the round this share of the frames with peaks have one on its curve
constexpr double TWIN = 0.4;                            // the sound's own pitch has twins (peaks at simple ratios of its delay) in this share of its peaks; a chorus voice stands alone
constexpr double STEEP = 0.056;                         // a curve steeper than this cannot be seen (the stretches looked at end at 6 %)
constexpr double GONE = 0.45;                           // a real voice is gone after taking it out: what is left along its curve is under this share of what was there
constexpr double UNSURE = -0.03;                        // a single voice whose lean is over this is also taken out the other way round (as the louder of the two), and the better way is kept
constexpr double FLAT = 0.85;                           // taking a chorus out never makes the waveform less peaked; under this share the copies were louder than the sound itself, which this way of taking out cannot undo
constexpr size_t STEP = 281 * (size_t) (HOP * DEC), LONG = 2 * STEP, FADE = 2 * 48000, LEAD = 48000;       // pieces of a long sound (12 s, one every 6 s; each starts on a whole frame of the picture, or its voices' curves would sit up to 5 ms off the sound), the fade from one to the next, the run-in before a piece is used

struct Voice
{
    int kind = 0;                                       // 0 sine, 1 triangle
    double rate = 0, phase = 0, mid = 0, half = 0;      // rounds a second; where in the round it is at the piece's start; the delay's middle and swing in samples at FS
    double on = 0, height = 0, piled = 0;               // the share of the frames with a peak on the curve, their mean height, how alike the rounds were
    double gain[NB] = {};                               // loudness per range
    double loud = 0, before = 0, after = 0;             // loudness in the middle ranges; what is found along the curve before and after taking the voices out
    double cover[8] = {}, least = 0, twin = 0;          // the share of the frames with a peak on the curve for each eighth of the round (-1: cannot be said) and the least of them; the share of its peaks that have a twin
    bool ahead = false; double lean = 0;                // the copy is LOUDER than the sound itself (gain then holds the sound's level against the copy); which of the two comes first at sharp events (under 0: the weaker, as usual)
    bool ok = false;
};
// How clear a voice is, 0 .. 1: seen in most frames, loud enough, there all the way round its movement, and well gone once taken out.
inline double clear (const Voice& v) { return v.on * std::min (1.0, v.loud / 0.2) * std::min (1.0, v.least / 0.35) * std::min (1.0, (1.0 - v.after / std::max (GONE * v.before, 1e-12)) / 0.3); }
struct Piece { size_t a = 0, n = 0; std::vector<Voice> voices[2]; double peaked[2][2] = {}; };      // a stretch of the sound read on its own (the voices' curves count from its start); how peaked the waveform is before and after
struct Found
{
    bool found = false; double sure = 0.0; size_t n = 0; std::vector<Piece> pieces; std::vector<std::string> notes;
    const Piece* best() const                           // the piece with the clearest voices (for the caption and the picture)
    {
        const Piece* out = nullptr; double top = -1.0;
        for (auto& p : pieces) for (int c = 0; c < 2; ++c) for (auto& v : p.voices[c]) { const double q = clear (v); if (q > top) { top = q; out = &p; } }
        return out;
    }
};

// ---------------------------------------------------------------- small tools
// fn(i, work) for i = 0 .. count - 1 on the machine's cores; every core makes its own work space once.
template <typename Make, typename Fn> inline void spread (int count, Make make, Fn fn, int most = 0)
{
    int k = (int) std::thread::hardware_concurrency(); if (k < 1) k = 1; if (most > 0) k = std::min (k, most); k = std::min (k, count);
    if (k <= 1) { auto w = make(); for (int i = 0; i < count; ++i) fn (i, w); return; }
    std::atomic<int> next (0); std::vector<std::thread> th;
    for (int j = 0; j < k; ++j) th.emplace_back ([&] { auto w = make(); for (;;) { const int i = next.fetch_add (1); if (i >= count) break; fn (i, w); } });
    for (auto& t : th) t.join();
}

// The transform of n real points by way of one of n / 2 complex points (half the work of a transform of n complex points).
struct Real
{
    int n, m; room::Transform F; Vec c, s;
    explicit Real (int size) : n (size), m (size / 2), F (size / 2), c ((size_t) size / 2 + 1), s ((size_t) size / 2 + 1) { for (int k = 0; k <= m; ++k) { c[(size_t) k] = std::cos (2 * PI * k / n); s[(size_t) k] = std::sin (2 * PI * k / n); } }
    // x (n points) -> re, im (n / 2 + 1 bins). a, b: work space of n / 2 points each.
    void run (const double* x, double* re, double* im, Vec& a, Vec& b) const
    {
        for (int i = 0; i < m; ++i) { a[(size_t) i] = x[2 * i]; b[(size_t) i] = x[2 * i + 1]; }
        F.run (a, b);
        for (int k = 0; k <= m; ++k)
        {
            const size_t k1 = (size_t) (k % m), k2 = (size_t) ((m - k) % m);
            const double zr = a[k1], zi = b[k1], yr = a[k2], yi = -b[k2];
            const double er = 0.5 * (zr + yr), ei = 0.5 * (zi + yi), orr = 0.5 * (zi - yi), oi = -0.5 * (zr - yr);      // the transforms of the even and of the odd points
            re[k] = er + c[(size_t) k] * orr + s[(size_t) k] * oi; im[k] = ei + c[(size_t) k] * oi - s[(size_t) k] * orr;
        }
    }
    // re, im (n / 2 + 1 bins of a real signal) -> x (n points)
    void back (const double* re, const double* im, double* x, Vec& a, Vec& b) const
    {
        for (int k = 0; k < m; ++k)
        {
            const double xr = re[k], xi = im[k], yr = re[m - k], yi = -im[m - k];
            const double er = 0.5 * (xr + yr), ei = 0.5 * (xi + yi), dr_ = 0.5 * (xr - yr), di = 0.5 * (xi - yi);
            const double orr = c[(size_t) k] * dr_ - s[(size_t) k] * di, oi = c[(size_t) k] * di + s[(size_t) k] * dr_;
            a[(size_t) k] = er - oi; b[(size_t) k] = ei + orr;
        }
        F.back (a, b);
        for (int i = 0; i < m; ++i) { x[2 * i] = a[(size_t) i]; x[2 * i + 1] = b[(size_t) i]; }
    }
};

// The running mean over `size` (odd) points with the end values held, never under 0.
inline void smooth (const double* x, int n, int size, double* out)
{
    const int h = size / 2; double s = 0.0;
    for (int i = -h; i <= h; ++i) s += x[std::clamp (i, 0, n - 1)];
    for (int i = 0; i < n; ++i) { out[i] = std::max (s / size, 0.0); s += x[std::min (i + h + 1, n - 1)] - x[std::max (i - h, 0)]; }
}

// The sound at half the sample rate: 41 taps (a sinc under a Kaiser window, beta 5), as scipy's resample_poly(y, 1, 2).
inline Vec dec2 (const Vec& y)
{
    static const Vec taps = []
    {
        auto i0 = [] (double x) { double s = 1.0, t = 1.0; for (int k = 1; k < 40; ++k) { const double q = x / (2.0 * k); t *= q * q; s += t; } return s; };
        Vec h (41); double sum = 0.0;
        for (int j = 0; j < 41; ++j) { const double m = j - 20.0, q = m / 20.0, a = 0.5 * m; h[(size_t) j] = 0.5 * (m == 0.0 ? 1.0 : std::sin (PI * a) / (PI * a)) * i0 (5.0 * std::sqrt (std::max (0.0, 1.0 - q * q))) / i0 (5.0); sum += h[(size_t) j]; }
        for (auto& v : h) v /= sum;
        return h;
    }();
    const long n = (long) y.size(), m = (n + 1) / 2; Vec out ((size_t) m, 0.0);
    for (long k = 0; k < m; ++k)
    {
        double s = 0.0; const long j0 = std::max (0L, 2 * k + 20 - (n - 1)), j1 = std::min (40L, 2 * k + 20);
        for (long j = j0; j <= j1; ++j) s += taps[(size_t) j] * y[(size_t) (2 * k + 20 - j)];
        out[(size_t) k] = s;
    }
    return out;
}

// ---------------------------------------------------------------- the picture, frame by frame, and its peaks
struct Peak { int t; double d, r, h; };                 // frame, the copy's delay at the frame's middle (samples at FS), stretch, height
using Peaks = std::vector<Peak>;

struct Tables
{
    Vec hann, tap, fr; std::vector<int> i0;             // per stretch and point of the longer piece: where it is read (from the frame's middle) and how far between two samples
    Tables() : hann ((size_t) W), tap ((size_t) LREF, 1.0), fr ((size_t) NRHO * LREF), i0 ((size_t) NRHO * LREF)
    {
        for (int i = 0; i < W; ++i) hann[(size_t) i] = 0.5 - 0.5 * std::cos (2 * PI * i / (W - 1));
        for (int i = 0; i < 64; ++i) { tap[(size_t) i] = 0.5 - 0.5 * std::cos (PI * i / 64); tap[(size_t) (LREF - 1 - i)] = tap[(size_t) i]; }
        for (int j = 0; j < NRHO; ++j)
        {
            const double rho = RHO0 + j * RSTEP;
            for (int i = 0; i < LREF; ++i) { const double u = (i - (W / 2 + TMAX)) * (1.0 - rho), f = std::floor (u); i0[(size_t) j * LREF + (size_t) i] = (int) f; fr[(size_t) j * LREF + (size_t) i] = u - f; }
        }
    }
};
inline const Tables& tables() { static const Tables t; return t; }
inline const Real& big() { static const Real f (NFFT); return f; }

struct Work
{
    Vec f, sre, sim, p0, pw, r, rre, rim, pr, q, xre, xim, a, wa, wb; std::vector<float> A, B;
    Work() : f ((size_t) NFFT, 0.0), sre ((size_t) BINS), sim ((size_t) BINS), p0 ((size_t) BINS), pw ((size_t) BINS), r ((size_t) NFFT, 0.0), rre ((size_t) BINS), rim ((size_t) BINS), pr ((size_t) BINS), q ((size_t) BINS),
             xre ((size_t) BINS), xim ((size_t) BINS), a ((size_t) NFFT), wa ((size_t) NFFT / 2), wb ((size_t) NFFT / 2) {}
};

// The sound at 24 kHz with room in front and behind, its frames, which of them sound, and the weight per frequency that
// takes the sound's own tilt off before anything is compared (from the spectrum averaged over the whole sound: a moving
// copy's comb is not in that average, a steep fall from bass to treble is; without it a dull sound would be compared by
// its lowest octave alone).
struct Frames
{
    Vec xp, tilt; int T = 0; std::vector<char> act;
    const double* mid (int t) const { return xp.data() + FRONT + (size_t) t * HOP; }
    // the frame's spectrum with the tilt off (the frame sits TMAX into the buffer), and its smoothed strength
    void spectrum (int t, Work& w, bool tilted = true) const
    {
        const Tables& tb = tables(); const double* m = mid (t) - W / 2;
        for (int i = 0; i < W; ++i) w.f[(size_t) (TMAX + i)] = m[i] * tb.hann[(size_t) i];
        big().run (w.f.data(), w.sre.data(), w.sim.data(), w.wa, w.wb);
        if (! tilted) return;
        for (int k = 0; k < BINS; ++k) { w.sre[(size_t) k] *= tilt[(size_t) k]; w.sim[(size_t) k] *= tilt[(size_t) k]; w.pw[(size_t) k] = w.sre[(size_t) k] * w.sre[(size_t) k] + w.sim[(size_t) k] * w.sim[(size_t) k]; }
        smooth (w.pw.data(), BINS, 129, w.p0.data());
    }
    // how well the frame matches the piece r (already in w.r): w.a[lag]
    void match (Work& w) const
    {
        big().run (w.r.data(), w.rre.data(), w.rim.data(), w.wa, w.wb);
        for (int k = 0; k < BINS; ++k) { w.rre[(size_t) k] *= tilt[(size_t) k]; w.rim[(size_t) k] *= tilt[(size_t) k]; w.pw[(size_t) k] = w.rre[(size_t) k] * w.rre[(size_t) k] + w.rim[(size_t) k] * w.rim[(size_t) k]; }
        smooth (w.pw.data(), BINS, 129, w.pr.data());
        double top = 0.0;
        for (int k = 0; k < BINS; ++k) { w.q[(size_t) k] = std::sqrt (w.p0[(size_t) k] * w.pr[(size_t) k]); top = std::max (top, w.q[(size_t) k]); }
        const double add = LEVEL * top + 1e-30;         // every frequency is divided by the sound's smoothed strength around it (not by its own strength: then a copy that hardly moves would not show at all)
        for (int k = 0; k < BINS; ++k)
        {
            const double den = w.q[(size_t) k] + add;
            w.xre[(size_t) k] = (w.sre[(size_t) k] * w.rre[(size_t) k] + w.sim[(size_t) k] * w.rim[(size_t) k]) / den; w.xim[(size_t) k] = (w.sim[(size_t) k] * w.rre[(size_t) k] - w.sre[(size_t) k] * w.rim[(size_t) k]) / den;
        }
        big().back (w.xre.data(), w.xim.data(), w.a.data(), w.wa, w.wb);
    }
};
inline Frames framesOf (const Vec& y1)
{
    Frames F; const Vec x = dec2 (y1); F.T = (int) (x.size() / (size_t) HOP) + 1; F.xp.assign ((size_t) FRONT + x.size() + (size_t) (W + HOP), 0.0); std::copy (x.begin(), x.end(), F.xp.begin() + FRONT);
    Vec e ((size_t) F.T, 0.0); double top = 0.0;
    for (int t = 0; t < F.T; ++t) { const double* m = F.mid (t) - W / 2; double s = 0.0; for (int i = 0; i < W; ++i) s += m[i] * m[i]; e[(size_t) t] = s; top = std::max (top, s); }
    F.act.assign ((size_t) F.T, 0); int count = 0;
    for (int t = 0; t < F.T; ++t) if (e[(size_t) t] > top * 1e-4) { F.act[(size_t) t] = 1; ++count; }
    Vec P ((size_t) BINS, count ? 0.0 : 1.0); Work w;
    for (int t = 0; t < F.T; ++t) if (F.act[(size_t) t]) { F.spectrum (t, w, false); for (int k = 0; k < BINS; ++k) P[(size_t) k] += w.sre[(size_t) k] * w.sre[(size_t) k] + w.sim[(size_t) k] * w.sim[(size_t) k]; }
    if (count) for (auto& v : P) v /= count;
    Vec Ps ((size_t) BINS); smooth (P.data(), BINS, 33, Ps.data()); const double pm = *std::max_element (Ps.begin(), Ps.end());
    F.tilt.resize ((size_t) BINS); for (int k = 0; k < BINS; ++k) F.tilt[(size_t) k] = 1.0 / std::sqrt (Ps[(size_t) k] + 1e-9 * pm + 1e-30);
    return F;
}

// The strongest peaks of every sounding frame, frames rising, within a frame the highest first. A peak is a cell (stretch,
// delay) that tops its neighbours four cells each way and stands over four times the usual size of a cell (the middle one
// of the frames' own middle values); delay and stretch are read to a fraction of a cell.
inline Peaks peaksOf (const Frames& F, const Stop& stop = {})
{
    const Tables& tb = tables(); const int T = F.T; std::vector<std::array<Peak, NPK>> got ((size_t) T); std::vector<int> count ((size_t) T, 0); std::vector<float> med ((size_t) T, 0.0f); std::atomic<bool> quit (false);
    spread (T, [] { Work w; w.A.resize ((size_t) NRHO * NL); w.B.resize ((size_t) NRHO * NL); return w; }, [&] (int t, Work& w)
    {
        if (! F.act[(size_t) t] || quit.load()) return;
        if ((t & 15) == 0 && stop && stop()) { quit.store (true); return; }
        F.spectrum (t, w); const double* base = F.mid (t); float* A = w.A.data();
        for (int j = 0; j < NRHO; ++j)
        {
            const int* i0 = &tb.i0[(size_t) j * LREF]; const double* fr = &tb.fr[(size_t) j * LREF];
            for (int i = 0; i < LREF; ++i) { const double* p = base + i0[i]; w.r[(size_t) i] = (p[0] * (1.0 - fr[i]) + p[1] * fr[i]) * tb.tap[(size_t) i]; }
            F.match (w);
            float* row = A + (size_t) j * NL; for (int i = 0; i < NL; ++i) row[i] = (float) w.a[(size_t) (TMIN + i)];
        }
        struct Top { float h; int j, i; }; Top top[NPK]; int n = 0;
        for (int j = 0; j < NRHO; ++j)
            for (int i = 0; i < NL; ++i)
            {
                const size_t at = (size_t) j * NL + (size_t) i; const float v = A[at];
                if (n == NPK && v <= top[NPK - 1].h) continue;
                if ((i > 0 && A[at - 1] > v) || (i < NL - 1 && A[at + 1] > v) || (j > 0 && A[at - NL] > v) || (j < NRHO - 1 && A[at + NL] > v)) continue;
                bool pk = true;
                for (int jj = std::max (0, j - 4); jj <= std::min (NRHO - 1, j + 4) && pk; ++jj) { const float* row = A + (size_t) jj * NL; for (int ii = std::max (0, i - 4); ii <= std::min (NL - 1, i + 4); ++ii) if (row[ii] > v) { pk = false; break; } }
                if (! pk) continue;
                int k = n < NPK ? n++ : NPK - 1;
                while (k > 0 && top[k - 1].h < v) { top[k] = top[k - 1]; --k; }
                top[k] = { v, j, i };
            }
        for (int k = 0; k < n; ++k)
        {
            const int j = top[k].j, i = top[k].i; const size_t at = (size_t) j * NL + (size_t) i; const float b = top[k].h; double di = 0.0, dj = 0.0;
            if (i > 0 && i < NL - 1) { const float a = A[at - 1], c = A[at + 1], d = a - 2 * b + c; if (d < 0) di = 0.5f * (a - c) / d; }
            if (j > 0 && j < NRHO - 1) { const float a = A[at - NL], c = A[at + NL], d = a - 2 * b + c; if (d < 0) dj = 0.5f * (a - c) / d; }
            const double rho = RHO0 + j * RSTEP + dj * RSTEP;
            got[(size_t) t][(size_t) k] = { t, (i + di + TMIN) * (1.0 - rho), rho, (double) b };       // against the frame stretched by rho the copy sits at delay / (1 - rho)
        }
        count[(size_t) t] = n;
        const size_t N = (size_t) NRHO * NL, h = N / 2; float* B = w.B.data();
        for (size_t k = 0; k < N; ++k) B[k] = std::abs (A[k]);
        std::nth_element (B, B + h, B + N); med[(size_t) t] = 0.5f * (*std::max_element (B, B + h) + B[h]);
    });
    Peaks P; if (quit.load()) return P;
    std::vector<float> m; for (int t = 0; t < T; ++t) if (F.act[(size_t) t]) m.push_back (med[(size_t) t]);
    if (m.empty()) return P;
    std::sort (m.begin(), m.end()); const float mm = (m.size() & 1) ? m[m.size() / 2] : 0.5f * (m[m.size() / 2 - 1] + m[m.size() / 2]); const double floor = 4.0 * (double) mm;
    for (int t = 0; t < T; ++t) for (int k = 0; k < count[(size_t) t]; ++k) if (got[(size_t) t][(size_t) k].h > floor) P.push_back (got[(size_t) t][(size_t) k]);
    return P;
}

// ---------------------------------------------------------------- curves through the peaks
inline double wrap (double x) { return x - std::floor (x); }
// One round of a voice's movement, -1 .. 1: a sine (0) or a triangle (1), and its slope per round.
inline void shape (double ph, int kind, double& s, double& ds)
{
    if (kind == 0) { s = std::sin (2 * PI * ph); ds = 2 * PI * std::cos (2 * PI * ph); return; }
    const double p = ph - std::floor (ph); s = 1.0 - 4.0 * std::abs (p - 0.5); ds = p < 0.5 ? 4.0 : -4.0;
}
// A voice's delay (samples at FS) and its stretch at frame t.
inline void at (const Voice& v, double t, double& d, double& r) { double s, ds; shape (v.rate * t / FPS + v.phase, v.kind, s, ds); d = v.mid + v.half * s; r = v.half * ds * v.rate / FS; }
inline bool near (const Peak& p, const Voice& v, double k = 1.0) { double d, r; at (v, p.t, d, r); return std::abs (p.d - d) < k * TOL && std::abs (p.r - r) < k * RTOL; }
// Moves (0.08 ms each way at least), stays inside the delays looked at, goes round one and a half times within the sound.
inline bool fits (const Voice& v, int T) { return v.ok && std::abs (v.half) >= 2.0 && v.mid - std::abs (v.half) >= TMIN && v.mid + std::abs (v.half) <= TMAX && v.rate * T / FPS >= 1.5; }

// Per eighth of the voice's round: the share of the frames with peaks that have one on the curve (-1: cannot be said, under
// 3 frames there or the curve too steep there to be seen). Returns the least of them (0 if fewer than six can be said).
// The sound's own pitch drifting, or a drum's falling tone, lies along one flank of a curve at best; a real voice is there
// all the way round.
inline double cover (const Peaks& P, const Voice& v, double* out)
{
    int all[8] = {}, hit[8] = {}; size_t i = 0;
    while (i < P.size())
    {
        const int t = P[i].t; bool on = false; size_t e = i; for (; e < P.size() && P[e].t == t; ++e) if (near (P[e], v)) on = true;
        const int k = std::min (7, (int) (wrap (v.rate * t / FPS + v.phase) * 8.0)); ++all[k]; if (on) ++hit[k];
        i = e;
    }
    double least = 1.0; int said = 0;
    for (int k = 0; k < 8; ++k)
    {
        double s, ds; shape ((k + 0.5) / 8.0, v.kind, s, ds);
        if (all[k] < 3 || std::abs (v.half * ds * v.rate / FS) > STEEP) { out[k] = -1.0; continue; }
        out[k] = (double) hit[k] / all[k]; least = std::min (least, out[k]); ++said;
    }
    return said >= 6 ? least : 0.0;
}
// The share of the voice's peaks with a twin: another peak in the same frame at a simple ratio of its delay, the stretch in
// the same ratio (the largest share over the ratios; a ratio counts if it can be looked at for 30 % of the peaks). At whole
// multiples only a twin at least three quarters as high counts: a voice that is fed back has a fainter one at twice its delay.
inline double twin (const Peaks& P, const Voice& v)
{
    static const int R[16][2] = { { 2, 1 }, { 3, 1 }, { 4, 1 }, { 1, 2 }, { 1, 3 }, { 1, 4 }, { 3, 2 }, { 2, 3 }, { 4, 3 }, { 3, 4 }, { 5, 2 }, { 2, 5 }, { 5, 3 }, { 3, 5 }, { 5, 4 }, { 4, 5 } };
    int can[16] = {}, got[16] = {}, mine = 0; size_t i = 0;
    while (i < P.size())
    {
        const int t = P[i].t; size_t e = i; while (e < P.size() && P[e].t == t) ++e;
        for (size_t a = i; a < e; ++a)
        {
            if (! near (P[a], v)) continue;
            ++mine;
            for (int q = 0; q < 16; ++q)
            {
                const double f = (double) R[q][0] / R[q][1], d = P[a].d * f, r = P[a].r * f, k = 2.0 * std::max (f, 1.0);
                if (d < TMIN + 2 || d > TMAX - 2 || std::abs (r) > STEEP) continue;
                ++can[q];
                for (size_t b = i; b < e; ++b) if (b != a && std::abs (P[b].d - d) < k * TOL && std::abs (P[b].r - r) < k * RTOL && (R[q][1] != 1 || P[b].h >= 0.75 * P[a].h)) { ++got[q]; break; }
            }
        }
        i = e;
    }
    double top = 0.0;
    for (int q = 0; q < 16; ++q) if (can[q] > 0 && can[q] >= 0.3 * mine) top = std::max (top, (double) got[q] / can[q]);
    return top;
}

// m x m equations by elimination; false if they cannot be solved.
inline bool solveSmall (double* A, double* b, int m)
{
    double big = 0.0; for (int i = 0; i < m * m; ++i) big = std::max (big, std::abs (A[i]));
    if (! (big > 0.0) || ! std::isfinite (big)) return false;
    for (int c = 0; c < m; ++c)
    {
        int p = c; for (int r = c + 1; r < m; ++r) if (std::abs (A[r * m + c]) > std::abs (A[p * m + c])) p = r;
        if (std::abs (A[p * m + c]) < 1e-13 * big) return false;
        if (p != c) { for (int k = 0; k < m; ++k) std::swap (A[p * m + k], A[c * m + k]); std::swap (b[p], b[c]); }
        for (int r = c + 1; r < m; ++r) { const double f = A[r * m + c] / A[c * m + c]; for (int k = c; k < m; ++k) A[r * m + k] -= f * A[c * m + k]; b[r] -= f * b[c]; }
    }
    for (int c = m - 1; c >= 0; --c) { double s = b[c]; for (int k = c + 1; k < m; ++k) s -= A[c * m + k] * b[k]; b[c] = s / A[c * m + c]; }
    for (int c = 0; c < m; ++c) if (! std::isfinite (b[c])) return false;
    return true;
}

// The curve of this rate (and, for a triangle, phase) through the peaks idx: delay and stretch fitted together.
inline Voice solve (const Peaks& P, const std::vector<int>& idx, double rate, int kind, double ph = 0.0)
{
    const double om = 2 * PI * rate; const int m = kind == 0 ? 3 : 2; double A[9] = {}, b[3] = {}; Voice v; v.kind = kind; v.rate = rate;
    for (int i : idx)
    {
        const Peak& p = P[(size_t) i]; const double t = p.t / FPS, w2 = p.h; double r1[3] = { 1.0, 0.0, 0.0 }, r2[3] = { 0.0, 0.0, 0.0 };
        if (kind == 0) { const double s = std::sin (om * t), c = std::cos (om * t); r1[1] = s; r1[2] = c; r2[1] = om * c / FS * SLOPE; r2[2] = -om * s / FS * SLOPE; }
        else { double s, ds; shape (rate * t + ph, 1, s, ds); r1[1] = s; r2[1] = ds * rate / FS * SLOPE; }
        for (int a = 0; a < m; ++a) { for (int c = 0; c < m; ++c) A[a * m + c] += w2 * (r1[a] * r1[c] + r2[a] * r2[c]); b[a] += w2 * (r1[a] * p.d + r2[a] * p.r * SLOPE); }
    }
    if (! solveSmall (A, b, m)) return v;
    v.ok = true; v.mid = b[0];
    if (kind == 0) { v.phase = std::atan2 (b[2], b[1]) / (2 * PI); v.half = std::hypot (b[1], b[2]); } else { v.phase = ph; v.half = b[1]; }
    return v;
}

// All four numbers of the curve (also its rate) fitted to the peaks idx, far-off peaks counting less (a miss of e counts
// as 2 (sqrt(1 + (e / 0.45)^2) - 1)).
inline Voice refine (const Peaks& P, const std::vector<int>& idx, const Voice& v)
{
    const int n = (int) idx.size(); const double scale[4] = { 1.0, 1.0, 0.01, 0.002 }, fs = TOL * 0.3; double x[4] = { v.mid, v.half, v.phase, v.rate };
    Vec f ((size_t) (2 * n)), f2 ((size_t) (2 * n)), J ((size_t) (8 * n));
    auto eval = [&] (const double* q, Vec& res, bool jac) -> double
    {
        double cost = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const Peak& p = P[(size_t) idx[(size_t) i]]; const double w = std::sqrt (p.h), tt = p.t / FPS; double s, ds; shape (q[3] * tt + q[2], v.kind, s, ds);
            res[(size_t) (2 * i)] = (q[0] + q[1] * s - p.d) * w; res[(size_t) (2 * i + 1)] = (q[1] * ds * q[3] / FS - p.r) * SLOPE * w;
            if (jac)
            {
                const double dds = v.kind == 0 ? -4 * PI * PI * s : 0.0; double* a = &J[(size_t) (8 * i)]; double* b = a + 4;
                a[0] = w; a[1] = s * w; a[2] = q[1] * ds * w; a[3] = q[1] * ds * tt * w;
                b[0] = 0.0; b[1] = ds * q[3] / FS * SLOPE * w; b[2] = q[1] * dds * q[3] / FS * SLOPE * w; b[3] = q[1] * (dds * tt * q[3] + ds) / FS * SLOPE * w;
                for (int k = 0; k < 4; ++k) { a[k] *= scale[k]; b[k] *= scale[k]; }
            }
        }
        for (int i = 0; i < 2 * n; ++i) { const double z = res[(size_t) i] / fs; cost += 2.0 * (std::sqrt (1.0 + z * z) - 1.0); }
        return 0.5 * fs * fs * cost;
    };
    double cost = eval (x, f, true), lam = 1e-3;
    if (! std::isfinite (cost)) return v;
    for (int it = 0; it < 60; ++it)
    {
        double H[16] = {}, g[4] = {};
        for (int i = 0; i < 2 * n; ++i)
        {
            const double z = f[(size_t) i] / fs, wi = 1.0 / std::sqrt (1.0 + z * z); const double* a = &J[(size_t) (4 * i)];
            for (int p = 0; p < 4; ++p) { g[p] += wi * a[p] * f[(size_t) i]; for (int q = 0; q < 4; ++q) H[p * 4 + q] += wi * a[p] * a[q]; }
        }
        bool better = false; double x2[4], c2 = cost, step = 0.0;
        for (int tries = 0; tries < 12 && ! better; ++tries)
        {
            double M[16], d[4];
            for (int p = 0; p < 4; ++p) { d[p] = -g[p]; for (int q = 0; q < 4; ++q) M[p * 4 + q] = H[p * 4 + q]; M[p * 4 + p] += lam * H[p * 4 + p] + 1e-300; }
            if (solveSmall (M, d, 4))
            {
                step = 0.0; for (int p = 0; p < 4; ++p) { x2[p] = x[p] + scale[p] * d[p]; step = std::max (step, std::abs (d[p])); }
                c2 = eval (x2, f2, false);
                if (std::isfinite (c2) && c2 < cost) { better = true; break; }
            }
            lam *= 10.0;
        }
        if (! better) break;
        const double gain = cost - c2; for (int p = 0; p < 4; ++p) x[p] = x2[p];
        cost = eval (x, f, true); lam = std::max (lam * 0.1, 1e-12);
        if (gain < 1e-13 * cost || step < 1e-9) break;
    }
    if (! (x[3] > 0.05 && x[3] < 5.0)) return v;
    Voice out = v; out.mid = x[0]; out.half = x[1]; out.phase = wrap (x[2]); out.rate = x[3];
    if (out.half < 0 && out.kind == 0) { out.half = -out.half; out.phase = wrap (out.phase + 0.5); }
    return out;
}

// Peaks that follow one another as a moving copy's would: the next peak sits where this one's stretch says the delay will
// be. Each chain: indices into P, frames rising, at least `shortest` peaks.
inline std::vector<std::vector<int>> chains (const Peaks& P, int gap = 3, int shortest = 6)
{
    struct Pair { double err; int ci, i; };
    std::vector<std::vector<int>> open, done; size_t i = 0;
    while (i < P.size())
    {
        const int t = P[i].t; size_t e = i; while (e < P.size() && P[e].t == t) ++e;
        std::vector<Pair> pairs;
        for (int ci = 0; ci < (int) open.size(); ++ci)
        {
            const Peak& l = P[(size_t) open[(size_t) ci].back()]; const double dt = t - l.t;
            for (size_t k = i; k < e; ++k)
            {
                const double rho = 0.5 * (l.r + P[k].r), err = std::abs (P[k].d - (l.d + rho * dt * HOP));
                if (err < TOL * (1 + 0.5 * (dt - 1)) && std::abs (P[k].r - l.r) < 0.006 * dt + 0.002) pairs.push_back ({ err, ci, (int) k });
            }
        }
        std::sort (pairs.begin(), pairs.end(), [] (const Pair& a, const Pair& b) { return a.err != b.err ? a.err < b.err : a.ci != b.ci ? a.ci < b.ci : a.i < b.i; });
        std::vector<char> taken (open.size(), 0), used (e - i, 0);
        for (auto& p : pairs) { if (taken[(size_t) p.ci] || used[(size_t) p.i - i]) continue; open[(size_t) p.ci].push_back (p.i); taken[(size_t) p.ci] = 1; used[(size_t) p.i - i] = 1; }
        std::vector<std::vector<int>> keep;
        for (size_t ci = 0; ci < open.size(); ++ci) { if (t - P[(size_t) open[ci].back()].t >= gap && ! taken[ci]) done.push_back (std::move (open[ci])); else keep.push_back (std::move (open[ci])); }
        for (size_t k = i; k < e; ++k) if (! used[k - i]) keep.push_back ({ (int) k });
        open = std::move (keep); i = e;
    }
    for (auto& c : open) done.push_back (std::move (c));
    std::vector<std::vector<int>> out; for (auto& c : done) if ((int) c.size() >= shortest) out.push_back (std::move (c));
    return out;
}

// The rates (Hz) at which the peaks' delays come round: those at which one round looks like the others. Each peak is first
// moved along its own stretch to the middle of its place in the round (so a quick voice does not smear), then counted per
// round, place in the round and delay. The score is how alike the rounds are (their counts multiplied round against
// round, 1 = all alike): what stands still (the sound's own pitch) is taken off first, and what happens once does not
// count at any rate. Strongest first: (rate, score). A longer sound needs the rates closer together.
inline std::vector<std::pair<double, double>> rounds (const Peaks& P, int T, int most = 8)
{
    const int K = 16, count = T <= 300 ? 1600 : (int) (1600.0 * T / 282.0); const double lo = TMIN * 0.5; const int nd = (int) ((TMAX * 1.1 - lo) / 2.0) + 3, nc = nd - 1;
    Vec rates ((size_t) count); { const double a = std::log10 (0.08), b = std::log10 (4.0), step = (b - a) / (count - 1); for (int i = 0; i < count; ++i) rates[(size_t) i] = std::pow (10.0, a + i * step); rates[0] = 0.08; rates[(size_t) count - 1] = 4.0; }
    Vec sc ((size_t) count, 0.0), S ((size_t) K * nc, 0.0), rows; std::vector<int> slot, touched, cells;
    for (int i = 0; i < count; ++i)
    {
        const double r = rates[(size_t) i], rT = r * T / FPS; if (rT < 1.5) continue;       // the curve must go round one and a half times within the sound
        const int n = (int) std::ceil (rT); slot.assign ((size_t) n * nc, -1); touched.clear(); rows.clear();
        auto add = [&] (int k, int b, int c, double h)
        {
            int& s = slot[(size_t) k * nc + (size_t) c];
            if (s < 0) { s = (int) touched.size(); touched.push_back (c); rows.resize (rows.size() + K, 0.0); }
            rows[(size_t) s * K + (size_t) b] += h;
        };
        for (auto& p : P)
        {
            const double x = r * (p.t / FPS); const int k = std::min ((int) x, n - 1); const double ph = x - std::floor (x); const int b = std::min ((int) (ph * K), K - 1);
            const double dp = p.d + p.r * ((b + 0.5) / K - ph) / r * FS; const int cell = std::clamp ((int) ((dp - lo) / 2.0), 0, nd - 1);
            if (cell < nc) add (k, b, cell, p.h);          // neighbouring delays are counted together
            if (cell > 0) add (k, b, cell - 1, p.h);
        }
        double own = 0.0; cells.clear();
        for (size_t s = 0; s < touched.size(); ++s)
        {
            double* row = &rows[s * K]; double m = 0.0; for (int b = 0; b < K; ++b) m += row[b]; m /= K; const int c = touched[s];
            for (int b = 0; b < K; ++b) { const double v = row[b] - m; own += v * v; S[(size_t) b * nc + (size_t) c] += v; }
            cells.push_back (c);
        }
        double all = 0.0;
        for (int c : cells) for (int b = 0; b < K; ++b) { double& v = S[(size_t) b * nc + (size_t) c]; all += v * v; v = 0.0; }
        if (own > 0.0) sc[(size_t) i] = (all - own) / ((n - 1) * own);
    }
    std::vector<std::pair<double, double>> out; Vec s2 = sc;
    for (int k = 0; k < most; ++k)
    {
        const int i = (int) (std::max_element (s2.begin(), s2.end()) - s2.begin());
        if (s2[(size_t) i] <= 0.0) break;
        out.push_back ({ rates[(size_t) i], sc[(size_t) i] });
        const auto a = std::lower_bound (rates.begin(), rates.end(), rates[(size_t) i] * 0.97) - rates.begin(), b = std::lower_bound (rates.begin(), rates.end(), rates[(size_t) i] * 1.03) - rates.begin();
        for (auto j = a; j < b; ++j) s2[(size_t) j] = 0.0;
    }
    return out;
}

using Mask = std::vector<char>;
inline int framesIn (const Peaks& P, const Mask& m) { int n = 0, last = -1; for (size_t i = 0; i < P.size(); ++i) if (m[i] && P[i].t != last) { ++n; last = P[i].t; } return n; }
inline std::vector<int> listOf (const Mask& m) { std::vector<int> out; for (size_t i = 0; i < m.size(); ++i) if (m[i]) out.push_back ((int) i); return out; }

// The voices of one kind at one rate: every chain of peaks gives a curve; the curve that collects the most (peak heights
// near it) is a voice, its peaks are taken away, and again. Returns what was collected in all.
inline double group (const Peaks& P, int T, const Mask& use, const std::vector<std::vector<int>>& seeds, double rate, int kind, int most, int framesWith, double need, std::vector<std::pair<Voice, Mask>>& out)
{
    const size_t N = P.size(); std::vector<std::pair<Voice, Mask>> cands;
    auto collect = [&] (const Voice& v, double k) { double e = 0.0; for (size_t i = 0; i < N; ++i) if (use[i] && near (P[i], v, k)) e += P[i].h; return e; };
    for (auto& seed : seeds)
    {
        Voice v;
        if (kind == 0) v = solve (P, seed, rate, 0);
        else
        {
            // a chain lies on one slope of a triangle and cannot say where the triangle turns: every place is tried (coarsely with a wide catch, then finely), the one that collects the most peaks is taken
            bool have = false; double top = 0.0; Voice best;
            for (int stage = 0; stage < 2; ++stage)
            {
                Vec phases; double k = 1.0;
                if (stage == 0) { for (int i = 0; i < 64; ++i) phases.push_back (i / 64.0); k = 3.0; }
                else { if (! have) break; for (int i = -8; i <= 8; ++i) phases.push_back (best.phase + i / 512.0); have = false; }
                for (double ph : phases)
                {
                    Voice q = solve (P, seed, rate, 1, wrap (ph));
                    if (q.half < 0) { q.half = -q.half; q.phase = wrap (q.phase + 0.5); }
                    if (! fits (q, T)) continue;
                    const double e = collect (q, k);
                    if (! have || e > top) { have = true; top = e; best = q; }
                }
            }
            if (have) v = best;
        }
        if (fits (v, T)) { Mask m (N); for (size_t i = 0; i < N; ++i) m[i] = use[i] && near (P[i], v); cands.push_back ({ v, std::move (m) }); }
    }
    double total = 0.0; Mask free (N, 1);
    for (int it = 0; it < most; ++it)
    {
        if (cands.empty()) break;
        int bi = 0; double bg = -1.0;
        for (size_t c = 0; c < cands.size(); ++c) { double g = 0.0; const Mask& m = cands[c].second; for (size_t i = 0; i < N; ++i) if (m[i] && free[i]) g += P[i].h; if (g > bg) { bg = g; bi = (int) c; } }
        Voice v = cands[(size_t) bi].first; Mask m (N); for (size_t i = 0; i < N; ++i) m[i] = cands[(size_t) bi].second[i] && free[i];
        for (int k = 0; k < 2; ++k)                           // fitted to what it collects, and collected again
        {
            const std::vector<int> idx = listOf (m); if (idx.size() < 8) break;
            const Voice q = refine (P, idx, v);
            if (! fits (q, T)) break;
            v = q; for (size_t i = 0; i < N; ++i) m[i] = use[i] && free[i] && near (P[i], v);
        }
        if (framesIn (P, m) < need * framesWith) break;
        for (size_t i = 0; i < N; ++i) if (m[i]) { total += P[i].h; free[i] = 0; }
        out.push_back ({ v, std::move (m) });
    }
    return total;
}

// Voices through the peaks P of T frames. For each rate at which the peaks come round and each kind (sine, triangle) the
// voices of that rate and kind are drawn out together; the rate and kind that explain the most are kept, their peaks
// taken away, and the search goes on for voices of another rate. Only chains that move count as a start: one that stands
// still is the sound's own pitch. A curve that is not there all the way round its movement, or whose peaks have twins, is
// no voice (those go to `dropped`).
inline std::vector<Voice> voices (const Peaks& P, int T, int most = 4, double need = NEED, std::vector<Voice>* dropped = nullptr)
{
    std::vector<Voice> found; const size_t N = P.size(); std::vector<std::vector<int>> C;
    for (auto& c : chains (P, 3, 5)) { double lo = 1e300, hi = -1e300; for (int i : c) { lo = std::min (lo, P[(size_t) i].d); hi = std::max (hi, P[(size_t) i].d); } if (hi - lo >= 2.0) C.push_back (c); }
    if (C.empty() || N < 20) return found;
    int framesWith = 0; { int last = -1; for (auto& p : P) if (p.t != last) { ++framesWith; last = p.t; } }
    Mask use (N, 1);
    for (int pass = 0; pass < 3; ++pass)
    {
        if ((int) found.size() >= most) break;
        std::vector<std::pair<double, std::vector<int>>> S;
        for (auto& c : C) { std::vector<int> u; double h = 0.0; for (int i : c) if (use[(size_t) i]) { u.push_back (i); h += P[(size_t) i].h; } if (u.size() >= 5) S.push_back ({ h, std::move (u) }); }
        std::stable_sort (S.begin(), S.end(), [] (const std::pair<double, std::vector<int>>& a, const std::pair<double, std::vector<int>>& b) { return a.first > b.first; });
        if (S.size() > 60) S.resize (60);
        if (S.empty()) break;
        std::vector<std::vector<int>> seeds; Mask in (N, 0); for (auto& s : S) { for (int i : s.second) in[(size_t) i] = 1; seeds.push_back (s.second); }
        Peaks Q; for (size_t i = 0; i < N; ++i) if (in[i]) Q.push_back (P[i]);
        bool have = false; double top = 0.0, rsBest = 0.0; std::vector<std::pair<Voice, Mask>> best;
        for (auto& rr : rounds (Q, T, 10))                    // the rates come from the moving chains' peaks alone
            for (int kind = 0; kind < 2; ++kind)
            {
                std::vector<std::pair<Voice, Mask>> grp; double total = group (P, T, use, seeds, rr.first, kind, most - (int) found.size(), framesWith, need, grp); total *= kind == 0 ? 1.0 : 0.95;       // a triangle has to be clearly the better one
                if (! grp.empty() && (! have || total > top)) { have = true; top = total; best = std::move (grp); rsBest = rr.second; }
            }
        if (! have) break;
        for (auto& g : best)
        {
            Voice v = g.first; Mask mm (N);
            for (int k = 0; k < 3; ++k)
            {
                for (size_t i = 0; i < N; ++i) mm[i] = use[i] && near (P[i], v);
                const std::vector<int> idx = listOf (mm); if (idx.size() < 8) break;
                v = refine (P, idx, v);
            }
            if (! fits (v, T)) continue;
            double h = 0.0; int n = 0; for (size_t i = 0; i < N; ++i) { mm[i] = use[i] && near (P[i], v); if (mm[i]) { h += P[i].h; ++n; } }
            v.on = (double) framesIn (P, mm) / std::max (framesWith, 1); v.height = n ? h / n : 0.0; v.piled = rsBest;
            v.least = cover (P, v, v.cover); v.twin = twin (P, v);
            if (v.on >= need && v.least >= LEAST && v.twin < TWIN) found.push_back (v);
            else if (v.on >= need && dropped != nullptr) dropped->push_back (v);
            for (size_t i = 0; i < N; ++i) if (near (P[i], v, 1.5)) use[i] = 0;
        }
    }
    return found;
}

// ---------------------------------------------------------------- taking out and putting on
// x at the place p (not whole), four points.
inline double lag4 (const double* x, long n, double p)
{
    const long i = (long) std::floor (p); const double f = p - i;
    if (i < 1 || i + 2 >= n) return 0.0;
    const double a = x[i - 1], b = x[i], c = x[i + 1], d = x[i + 2];
    return b + 0.5 * f * (c - a + f * (2.0 * a - 5.0 * b + 4.0 * c - d + f * (3.0 * (b - c) + d - a)));
}
// The voices' delays in samples at 48 kHz for `count` samples from sample `first` of the piece (never under 8 samples).
// Faster or slower: the movement turns about `pivot` (samples from the piece's start).
inline std::vector<Vec> curves (const std::vector<Voice>& V, long first, size_t count, double speed = 1.0, double depth = 1.0, double pivot = 0.0)
{
    std::vector<Vec> D (V.size(), Vec (count)); const double per = HOP * DEC;
    for (size_t v = 0; v < V.size(); ++v)
    {
        Voice q = V[v]; q.half = V[v].half * depth; q.rate = V[v].rate * speed; q.phase = V[v].phase + V[v].rate * (pivot / per) / FPS * (1.0 - speed);
        for (size_t i = 0; i < count; ++i) { double d, r; at (q, (first + (long) i) / per, d, r); D[v][i] = std::max (d * DEC, 8.0); }
    }
    return D;
}
// How far after sample m its copy is heard, a(m) = d(m + a(m)), for `count` samples from sample `first` of the piece.
inline Vec advance (const Voice& v, long first, size_t count)
{
    Vec a (count); const double per = HOP * DEC;
    for (size_t i = 0; i < count; ++i)
    {
        const double m = (double) (first + (long) i); double d, r; at (v, m / per, d, r); double q = std::max (d * DEC, 8.0);
        for (int k = 0; k < 4; ++k) { at (v, (m + q) / per, d, r); q = std::max (d * DEC, 8.0); }
        a[i] = q;
    }
    return a;
}
// The voices taken out of one range: x[n] = y[n] - sum over the voices of g x(n - d[n]) (the result's own earlier samples).
inline Vec inverse (const Vec& y, const std::vector<Vec>& D, const double* g)
{
    const long n = (long) y.size(); Vec x ((size_t) n, 0.0); const size_t V = D.size();
    for (long i = 0; i < n; ++i)
    {
        double s = y[(size_t) i];
        for (size_t v = 0; v < V; ++v) { const double p = i - D[v][(size_t) i]; if (p >= 1.0) s -= g[v] * lag4 (x.data(), n, p); }
        x[(size_t) i] = s;
    }
    return x;
}
// x read along a delay curve.
inline Vec along (const Vec& x, const Vec& d)
{
    const long n = (long) x.size(); Vec r ((size_t) n, 0.0);
    for (long i = 0; i < n; ++i) { const double p = i - d[(size_t) i]; if (p >= 1.0) r[(size_t) i] = lag4 (x.data(), n, p); }
    return r;
}

// The sound in NB frequency ranges that add up to the sound again: masks on a transform of 4096 points, a frame every
// 1024. A mask falls from 1 a quarter octave under its edge to 0 a quarter octave over it.
inline std::vector<Vec> bands (const Vec& y)
{
    static const Real fft (BN);
    static const Vec win = [] { Vec w ((size_t) BN); for (int i = 0; i < BN; ++i) w[(size_t) i] = std::sqrt (0.5 - 0.5 * std::cos (2 * PI * i / BN)); return w; }();
    static const Vec masks = []
    {
        const int nb = BN / 2 + 1; Vec M ((size_t) NB * nb), lp ((size_t) (NB - 1) * nb);
        for (int e = 0; e < NB - 1; ++e) for (int k = 0; k < nb; ++k) { const double u = std::clamp (std::log2 (std::max (k * SR / BN, 1e-9) / EDGES[e]) / 0.25, -1.0, 1.0); lp[(size_t) e * nb + (size_t) k] = 0.5 - 0.5 * std::sin (0.5 * PI * u); }
        for (int k = 0; k < nb; ++k)
        {
            M[(size_t) k] = lp[(size_t) k];
            for (int b = 1; b < NB - 1; ++b) M[(size_t) b * nb + (size_t) k] = lp[(size_t) b * nb + (size_t) k] - lp[(size_t) (b - 1) * nb + (size_t) k];
            M[(size_t) (NB - 1) * nb + (size_t) k] = 1.0 - lp[(size_t) (NB - 2) * nb + (size_t) k];
        }
        return M;
    }();
    const long n = (long) y.size(); const int J = (int) (n / BH) + 1 + BN / BH, nb = BN / 2 + 1; std::vector<Vec> out ((size_t) NB, Vec ((size_t) n, 0.0)); Vec nrm ((size_t) n, 0.0);
    Vec seg ((size_t) BN), re ((size_t) nb), im ((size_t) nb), xr ((size_t) nb), xi ((size_t) nb), z ((size_t) BN), wa ((size_t) BN / 2), wb ((size_t) BN / 2);
    for (int j = 0; j < J; ++j)
    {
        const long s = (long) j * BH - BN, lo = std::max (s, 0L), hi = std::min (s + BN, n);
        if (hi <= lo) continue;
        bool any = false;
        for (int i = 0; i < BN; ++i) { const long q = s + i; seg[(size_t) i] = q >= 0 && q < n ? y[(size_t) q] * win[(size_t) i] : 0.0; if (seg[(size_t) i] != 0.0) any = true; }
        for (long q = lo; q < hi; ++q) nrm[(size_t) q] += win[(size_t) (q - s)] * win[(size_t) (q - s)];
        if (! any) continue;
        fft.run (seg.data(), re.data(), im.data(), wa, wb);
        for (int b = 0; b < NB; ++b)
        {
            const double* m = &masks[(size_t) b * nb];
            for (int k = 0; k < nb; ++k) { xr[(size_t) k] = re[(size_t) k] * m[k]; xi[(size_t) k] = im[(size_t) k] * m[k]; }
            fft.back (xr.data(), xi.data(), z.data(), wa, wb);
            double* o = out[(size_t) b].data(); for (long q = lo; q < hi; ++q) o[q] += z[(size_t) (q - s)] * win[(size_t) (q - s)];
        }
    }
    for (int b = 0; b < NB; ++b) for (long q = 0; q < n; ++q) out[(size_t) b][(size_t) q] /= std::max (nrm[(size_t) q], 1e-9);
    return out;
}

// A plain picture for measuring: frames of 2048 every 1024, Hann. re, im: [frame * SB + bin].
struct Spec { int T = 0; Vec re, im; };
inline Spec spec (const Vec& x)
{
    static const Real fft (SN);
    static const Vec win = [] { Vec w ((size_t) SN); for (int i = 0; i < SN; ++i) w[(size_t) i] = 0.5 - 0.5 * std::cos (2 * PI * i / SN); return w; }();
    Spec S; const long n = (long) x.size(); S.T = n >= SN ? (int) ((n - SN) / SH) + 1 : 0; S.re.resize ((size_t) S.T * SB); S.im.resize ((size_t) S.T * SB); Vec seg ((size_t) SN), wa ((size_t) SN / 2), wb ((size_t) SN / 2);
    for (int t = 0; t < S.T; ++t) { const double* p = x.data() + (size_t) t * SH; for (int i = 0; i < SN; ++i) seg[(size_t) i] = p[i] * win[(size_t) i]; fft.run (seg.data(), &S.re[(size_t) t * SB], &S.im[(size_t) t * SB], wa, wb); }
    return S;
}

// Per range and voice: how much of x is x read along the voice's curve (g[range * V + voice]). Loud and quiet stretches
// and frequencies count alike: the products are divided by the sound's power there, smoothed widely (770 Hz, 0.2 s; a
// narrow smoothing follows the copy's own comb and reads too low).
inline Vec left (const Vec& x, const std::vector<Vec>& D)
{
    const size_t V = D.size(); Vec out ((size_t) NB * V, 0.0); const Spec X = spec (x); const int T = X.T; if (T == 0 || V == 0) return out;
    const size_t N = (size_t) T * SB; Vec P (N), a (N), w (N); double mean = 0.0;
    for (size_t i = 0; i < N; ++i) { P[i] = X.re[i] * X.re[i] + X.im[i] * X.im[i]; mean += P[i]; } mean /= (double) N;
    {   // the running mean over 33 bins, then over 9 frames, the edges held
        Vec row ((size_t) std::max (SB, T)), sm ((size_t) std::max (SB, T));
        for (int t = 0; t < T; ++t) { smooth (&P[(size_t) t * SB], SB, 33, &a[(size_t) t * SB]); }
        for (int k = 0; k < SB; ++k) { for (int t = 0; t < T; ++t) row[(size_t) t] = a[(size_t) t * SB + (size_t) k]; smooth (row.data(), T, 9, sm.data()); for (int t = 0; t < T; ++t) w[(size_t) t * SB + (size_t) k] = sm[(size_t) t]; }
    }
    const double top = *std::max_element (w.begin(), w.end());
    for (size_t i = 0; i < N; ++i) { const double ps = w[i]; w[i] = ps > top * 1e-7 ? 1.0 / (ps + 1e-4 * mean + 1e-30) : 0.0; }
    std::vector<Spec> R; for (size_t v = 0; v < V; ++v) R.push_back (spec (along (x, D[v])));
    int band[SB]; for (int k = 0; k < SB; ++k) { const double f = k * SR / SN; int b = 0; while (b < NB - 1 && EDGES[b] <= f) ++b; band[k] = b; }
    for (int b = 0; b < NB; ++b)
    {
        double A[16] = {}, rhs[4] = {};
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < SB; ++k)
            {
                if (band[k] != b) continue;
                const size_t i = (size_t) t * SB + (size_t) k; const double wi = w[i]; if (wi == 0.0) continue;
                for (size_t v = 0; v < V; ++v)
                {
                    rhs[v] += wi * (X.re[i] * R[v].re[i] + X.im[i] * R[v].im[i]);
                    for (size_t u = v; u < V; ++u) A[v * V + u] += wi * (R[v].re[i] * R[u].re[i] + R[v].im[i] * R[u].im[i]);
                }
            }
        double trace = 0.0; for (size_t v = 0; v < V; ++v) { trace += A[v * V + v]; for (size_t u = v + 1; u < V; ++u) A[u * V + v] = A[v * V + u]; }
        const double ridge = 1e-3 * trace / (double) V + 1e-30; for (size_t v = 0; v < V; ++v) A[v * V + v] += ridge;
        if (solveSmall (A, rhs, (int) V)) for (size_t v = 0; v < V; ++v) out[(size_t) b * V + v] = rhs[v];
    }
    return out;
}

inline const double* centres() { static const std::array<double, NB> c = [] { std::array<double, NB> q {}; for (int b = 0; b < NB; ++b) q[(size_t) b] = std::sqrt ((b == 0 ? 75.0 : EDGES[b - 1]) * (b == NB - 1 ? 19200.0 : EDGES[b])); return q; }(); return c.data(); }

// The voices' loudness per range from the wet sound alone (G[range * V + voice]) and the sound with them taken out.
// In a low range a voice's delay swings through less than one wave: there the sound's own steady tones are taken for the
// voice (or hide it), and the measure cannot be trusted. Such a range leans on the range above it.
// B: the sound's ranges (bands()).
inline Vec loudness (const std::vector<Vec>& B, const std::vector<Voice>& V, const std::vector<Vec>& D, Vec& G, int turns = 6)
{
    const size_t nv = V.size(); G.assign ((size_t) NB * nv, 0.0); const double* centre = centres();
    auto remove = [&] { Vec x (B[0].size(), 0.0); for (int b = 0; b < NB; ++b) { const Vec xb = inverse (B[(size_t) b], D, &G[(size_t) b * nv]); for (size_t i = 0; i < x.size(); ++i) x[i] += xb[i]; } return x; };
    for (int turn = 0; turn < turns; ++turn)
    {
        const Vec x = remove(), L = left (x, D);
        for (size_t i = 0; i < G.size(); ++i) G[i] += L[i];
        for (size_t v = 0; v < nv; ++v)
        {
            const double swing = 2.0 * std::abs (V[v].half) / FS;
            for (int b = NB - 2; b >= 0; --b) { const double w = std::min (1.0, centre[b] * swing); G[(size_t) b * nv + v] = w * G[(size_t) b * nv + v] + (1.0 - w) * G[(size_t) (b + 1) * nv + v]; }
        }
        for (int b = 0; b < NB; ++b) { double s = 0.0; for (size_t v = 0; v < nv; ++v) s += std::abs (G[(size_t) b * nv + v]); const double k = std::min (1.0, MOST / std::max (s, 1e-9)); for (size_t v = 0; v < nv; ++v) G[(size_t) b * nv + v] *= k; }
    }
    return remove();
}

// One copy that is LOUDER than the sound itself: y = x + g x(n - d), g over 1. Taken out the usual way, the louder part is
// kept and the sound itself removed in its place. Told from the copy's side the same sound is y = v + h v(n + a(n)): v the
// copy, h = 1 / g under 1, the sound itself an EARLIER, quieter copy of it. That is the usual case with time running
// backwards: the sound is turned round, h read and v taken out as always, and turned back; the sound itself is then v read
// a(n) later, h times as loud. H: h per range (read off the sound if `measure`). x: the sound without the copy. U (if
// asked for): per range the copy brought back to its sound's place (g x in that range).
inline void backwards (const Vec& y, const Voice& v, long first, Vec& H, bool measure, Vec& x, std::vector<Vec>* U)
{
    const size_t n = y.size(); const Vec A = advance (v, first, n); const Vec yr (y.rbegin(), y.rend()); const std::vector<Vec> Dr (1, Vec (A.rbegin(), A.rend()));
    const std::vector<Vec> Br = bands (yr);
    if (measure) { const std::vector<Voice> one (1, v); loudness (Br, one, Dr, H); }
    x.assign (n, 0.0); if (U != nullptr) U->assign ((size_t) NB, Vec (n, 0.0));
    for (int b = 0; b < NB; ++b)
    {
        Vec vb = inverse (Br[(size_t) b], Dr, &H[(size_t) b]); std::reverse (vb.begin(), vb.end());
        for (size_t i = 0; i < n; ++i) { const double u = lag4 (vb.data(), (long) n, (double) i + A[i]); x[i] += H[(size_t) b] * u; if (U != nullptr) (*U)[(size_t) b][i] = u; }
    }
}

// Which comes first at a sharp event, the weaker or the stronger of the two? With e the size of the sound (range by range,
// from 150 Hz up) and ed its size read along the voice's curve: log (sum e^2 ed / sum e ed^2). Under 0: the later one (the
// copy) is the weaker, as in a usual chorus. Over 0: the copy is the louder. About 0 for a sound without sharp events.
inline double lean (const std::vector<Vec>& B, const Vec& d)
{
    double A = 0.0, Q = 0.0;
    for (int b = 1; b < NB; ++b) { const Vec& x = B[(size_t) b]; const Vec r = along (x, d); for (size_t i = 0; i < x.size(); ++i) { const double e = std::abs (x[i]), ed = std::abs (r[i]); A += e * e * ed; Q += e * ed * ed; } }
    return std::log (std::max (A, 1e-300) / std::max (Q, 1e-300));
}
// How unevenly the sound's strength is spread in time, the ranges from 150 Hz up together.
inline double uneven (const Vec& x)
{
    const std::vector<Vec> B = bands (x); double m4 = 0.0, m2 = 0.0;
    for (int b = 1; b < NB; ++b) { double s2 = 0.0, s4 = 0.0; for (double v : B[(size_t) b]) { const double q = v * v; s2 += q; s4 += q * q; } const double n = (double) std::max<size_t> (1, B[(size_t) b].size()); m4 += s4 / n; m2 += (s2 / n) * (s2 / n); }
    return m4 / std::max (m2, 1e-300);
}

// How peaked the sound's waveform is (of its sample to sample changes).
inline double spiky (const Vec& x)
{
    double m2 = 0.0, m4 = 0.0; const size_t n = x.size(); if (n < 2) return 0.0;
    for (size_t i = 1; i < n; ++i) { const double z = x[i] - x[i - 1], q = z * z; m2 += q; m4 += q * q; }
    m2 /= (double) (n - 1); m4 /= (double) (n - 1); return m4 / std::max (m2 * m2, 1e-30) - 3.0;
}

// How well the sound matches itself along one voice's curve: the mean over the frames `use` of the picture's value at the
// voice's own delay and stretch (the best of the three nearest delays). Also gives back which frames sound.
inline double ride (const Vec& y1, const Voice& v, std::vector<char>* acts = nullptr, const std::vector<char>* use = nullptr)
{
    const Tables& tb = tables(); const Frames F = framesOf (y1); Work w; double sum = 0.0; int count = 0; const std::vector<char>& on = use ? *use : F.act;
    for (int t = 0; t < F.T && t < (int) on.size(); ++t)
    {
        if (! on[(size_t) t]) continue;
        double d, rho; at (v, t, d, rho); rho = std::clamp (rho, RHO0, RHO0 + (NRHO - 1) * RSTEP);
        F.spectrum (t, w); const double* base = F.mid (t);
        for (int i = 0; i < LREF; ++i) { const double u = (i - (W / 2 + TMAX)) * (1.0 - rho), f = std::floor (u); const double* p = base + (long) f; w.r[(size_t) i] = (p[0] * (1.0 - (u - f)) + p[1] * (u - f)) * tb.tap[(size_t) i]; }
        F.match (w);
        const int lag = std::clamp ((int) std::nearbyint (d / (1.0 - rho)), 1, NFFT / 2 - 2);
        sum += std::max ({ w.a[(size_t) lag - 1], w.a[(size_t) lag], w.a[(size_t) lag + 1] }); ++count;
    }
    if (acts) *acts = F.act;
    return count ? sum / count : 0.0;
}

// ---------------------------------------------------------------- find
// The pieces a sound of n samples is read in: (start, length). Every piece starts on a whole frame of the picture; the last
// one ends with the sound (and takes the place of a regular piece that would leave under 3 s after it).
inline std::vector<std::pair<size_t, size_t>> pieces (size_t n)
{
    std::vector<std::pair<size_t, size_t>> out; const size_t per = (size_t) (HOP * DEC);
    if (n <= LONG) { out.push_back ({ 0, n }); return out; }
    out.push_back ({ 0, LONG });
    for (size_t a = STEP;; a += STEP)
    {
        if (a + LONG + STEP / 2 >= n) { const size_t s = ((n - LONG) / per) * per; out.push_back ({ s, n - s }); break; }
        out.push_back ({ a, LONG });
    }
    return out;
}

// One side of one piece: the voices through its peaks (frames counted from the piece's start), their loudness, and the
// test that a voice is gone once it is taken out.
inline void sideOf (const Vec& y, const Peaks& P, int T, std::vector<Voice>& V, double peaked[2], std::vector<std::string>& notes, int c, const Stop& stop)
{
    std::vector<Voice> no; V = voices (P, T, 4, NEED, &no); const long n = (long) y.size();
    for (auto& v : no) { char s[240]; std::snprintf (s, sizeof s, "side %d: no voice, %s: %s %.4f Hz %.3f +- %.3f ms on %.2f, least %.2f, twins %.2f", c, v.twin >= TWIN ? "the sound's own pitch" : "not there all the way round", v.kind ? "tri" : "sin", v.rate, v.mid / FS * 1e3, std::abs (v.half) / FS * 1e3, v.on, v.least, v.twin); notes.push_back (s); }
    if (V.empty()) return;
    const std::vector<Vec> B = bands (y);
    while (! V.empty())
    {
        if (stop && stop()) { V.clear(); return; }
        const std::vector<Vec> D = curves (V, 0, (size_t) n); Vec G; Vec xh = loudness (B, V, D, G); std::vector<Voice> keep; const size_t nv = V.size();
        bool ahead = false, rode = false; double before = 0.0, after = 0.0;
        if (nv == 1)                                          // one copy: is it the louder of the two? Then it is taken out the other way round
        {
            const double L = lean (B, D[0]); V[0].lean = L;
            if (L > UNSURE)
            {
                Vec H, xo; backwards (y, V[0], 0, H, true, xo, nullptr);
                std::vector<char> act; before = ride (y, V[0], &act); const double bu = ride (xh, V[0], nullptr, &act), bo = ride (xo, V[0], nullptr, &act), pu = spiky (xh), po = spiky (xo);
                const int votes = (L > 0.02 ? 1 : 0) + (bo < 0.8 * bu ? 1 : 0) + (pu > 1.0 && po > 1.05 * pu ? 1 : 0) + (uneven (xo) > 1.05 * uneven (xh) ? 1 : 0); rode = true; after = bu;
                char t[240]; std::snprintf (t, sizeof t, "side %d: one voice, lean %+.3f: the other way round gets %d of 4 votes (along %.3f against %.3f, peaked %.1f against %.1f)", c, L, votes, bo, bu, po, pu); notes.push_back (t);
                const double k0 = spiky (y); auto stands = [&] (const Vec& g, double b, double p) { return (g[3] + g[4] + g[5]) / 3.0 >= LOUD && b < GONE * before && ! (k0 > 1.0 && p < FLAT * k0); };      // would the result stand?
                if (votes >= 3 || (stands (H, bo, po) && ! stands (G, bu, pu))) { ahead = true; G = H; xh = std::move (xo); after = bo; }      // ... also when the usual way fails its tests and the other way round passes them
            }
        }
        for (size_t i = 0; i < nv; ++i)
        {
            Voice& v = V[i]; v.loud = 0.0; v.ahead = ahead;
            for (int b = 0; b < NB; ++b) v.gain[b] = G[(size_t) b * nv + i];
            for (int b = 3; b <= 5; ++b) v.loud += v.gain[b] / 3.0;
            if (rode) { v.before = before; v.after = after; } else { std::vector<char> act; v.before = ride (y, v, &act); v.after = ride (xh, v, nullptr, &act); }
            if (v.loud >= LOUD && v.after < GONE * v.before) keep.push_back (v);
            else { char t[200]; std::snprintf (t, sizeof t, "side %d: dropped %s %.4f Hz %.3f +- %.3f ms on %.2f loud %.2f along %.3f -> %.3f", c, v.kind ? "tri" : "sin", v.rate, v.mid / FS * 1e3, std::abs (v.half) / FS * 1e3, v.on, v.loud, v.before, v.after); notes.push_back (t); }
        }
        if (keep.size() == V.size())
        {
            peaked[0] = spiky (y); peaked[1] = spiky (xh);
            if (peaked[0] > 1.0 && peaked[1] < FLAT * peaked[0]) { char t[200]; std::snprintf (t, sizeof t, "side %d: %d voices not used, taking them out makes the sound less peaked (%.1f -> %.1f)", c, (int) V.size(), peaked[0], peaked[1]); notes.push_back (t); V.clear(); }
            break;
        }
        V = keep;                                             // some were false: the loudness of the others is measured again without them
    }
}

inline Found find (const Stereo& y, const Stop& stop = {})
{
    Found F; F.n = y.size(); if (F.n < 48000) return F;
    const bool same = y.l == y.r; Peaks P[2];
    for (int c = 0; c < (same ? 1 : 2); ++c) { P[c] = peaksOf (framesOf (c == 0 ? y.l : y.r), stop); if (stop && stop()) return F; }
    if (same) P[1] = P[0];
    const auto lay = pieces (F.n); F.pieces.resize (lay.size()); const int per = HOP * DEC; std::vector<std::vector<std::string>> notes (lay.size() * 2);
    for (size_t k = 0; k < lay.size(); ++k) { F.pieces[k].a = lay[k].first; F.pieces[k].n = lay[k].second; }
    spread ((int) lay.size() * (same ? 1 : 2), [] { return 0; }, [&] (int job, int&)
    {
        const size_t k = (size_t) job / (same ? 1 : 2); const int c = same ? 0 : job % 2; Piece& pc = F.pieces[k];
        const Vec& src = c == 0 ? y.l : y.r; const Vec part (src.begin() + (long) pc.a, src.begin() + (long) (pc.a + pc.n));
        const int t0 = (int) (pc.a / (size_t) per), T = (int) (((pc.n + 1) / 2) / (size_t) HOP) + 1; Peaks Q;
        for (auto& p : P[c]) if (p.t >= t0 && p.t < t0 + T) { Peak q = p; q.t -= t0; Q.push_back (q); }
        sideOf (part, Q, T, pc.voices[c], pc.peaked[c], notes[(size_t) job], c, stop);
    }, 4);
    if (stop && stop()) { Found none; none.n = F.n; return none; }
    double best = 0.0;
    for (size_t k = 0; k < lay.size(); ++k)
    {
        Piece& pc = F.pieces[k];
        if (same) { pc.voices[1] = pc.voices[0]; pc.peaked[1][0] = pc.peaked[0][0]; pc.peaked[1][1] = pc.peaked[0][1]; }
        for (int c = 0; c < 2; ++c) for (auto& v : pc.voices[c]) { F.found = true; best = std::max (best, clear (v)); }
    }
    for (auto& ns : notes) for (auto& s : ns) F.notes.push_back (s);
    F.sure = F.found ? std::min (1.0, best / 0.6) : 0.0;
    return F;
}

// ---------------------------------------------------------------- apply
// How much piece k counts at sample i: the pieces are faded into one another over 2 s in the middle of what they share.
inline double weight (const Found& F, size_t k, double i)
{
    auto up = [&] (size_t q) -> double        // piece q coming in over piece q - 1
    {
        const double o1 = (double) F.pieces[q].a, o2 = (double) (F.pieces[q - 1].a + F.pieces[q - 1].n), c = 0.5 * (o1 + o2), f = std::max (1.0, std::min ((double) FADE, 0.5 * (o2 - o1)));
        return std::clamp ((i - (c - 0.5 * f)) / f, 0.0, 1.0);
    };
    double w = 1.0;
    if (k > 0) w *= up (k);
    if (k + 1 < F.pieces.size()) w *= 1.0 - up (k + 1);
    return w;
}
// The samples of a sound of n samples that piece k is in charge of (first, one past the last).
inline std::pair<size_t, size_t> span (const Found& F, size_t k, size_t n)
{
    double a = 0.0, b = (double) n;
    if (k > 0) { const double o1 = (double) F.pieces[k].a, o2 = (double) (F.pieces[k - 1].a + F.pieces[k - 1].n), f = std::max (1.0, std::min ((double) FADE, 0.5 * (o2 - o1))); a = 0.5 * (o1 + o2) - 0.5 * f; }
    if (k + 1 < F.pieces.size()) { const double o1 = (double) F.pieces[k + 1].a, o2 = (double) (F.pieces[k].a + F.pieces[k].n), f = std::max (1.0, std::min ((double) FADE, 0.5 * (o2 - o1))); b = 0.5 * (o1 + o2) + 0.5 * f; }
    const size_t ia = (size_t) std::clamp (std::floor (a), 0.0, (double) n), ib = (size_t) std::clamp (std::ceil (b), 0.0, (double) n);
    return { ia, std::max (ia, ib) };
}

// One side of the sound the voices were found on, with the voices scaled by s (s = -1 takes them out; with another speed
// or depth the found voices go out and the changed ones, 1 + s times, come in).
inline Vec side (const Vec& y, const Found& F, int c, double s, double speed, double depth)
{
    Vec out = y; const size_t n = y.size(); if (! F.found || n == 0 || n != F.n) return out;
    const bool plain = speed == 1.0 && depth == 1.0; std::vector<size_t> jobs;
    for (size_t k = 0; k < F.pieces.size(); ++k) if (! F.pieces[k].voices[c].empty() && span (F, k, n).second > span (F, k, n).first) jobs.push_back (k);
    if (jobs.empty() || (plain && s == 0.0)) return out;
    std::vector<Vec> eff (jobs.size()); std::vector<size_t> from (jobs.size());
    spread ((int) jobs.size(), [] { return 0; }, [&] (int j, int&)
    {
        const size_t k = jobs[(size_t) j]; const Piece& pc = F.pieces[k]; const std::vector<Voice>& V = pc.voices[c]; const size_t nv = V.size();
        const bool ahead = nv == 1 && V[0].ahead;             // one copy louder than the sound: taken out from the end backwards, so the run-in is needed after the stretch as well
        const auto sp = span (F, k, n); const size_t a = sp.first > LEAD ? std::max (sp.first - LEAD, std::min (pc.a, sp.first)) : 0, b = ahead ? std::max (sp.second, std::min ({ sp.second + LEAD, pc.a + pc.n, n })) : sp.second, m = b - a;      // with a run-in, but not from before the piece
        const Vec part (y.begin() + (long) a, y.begin() + (long) b); const long first = (long) a - (long) pc.a; const double pivot = 0.5 * (double) F.n - (double) pc.a;
        Vec e (m, 0.0);
        if (ahead)
        {
            Vec H (V[0].gain, V[0].gain + NB), x; std::vector<Vec> U; backwards (part, V[0], first, H, false, x, plain ? nullptr : &U);
            if (plain) for (size_t i = 0; i < m; ++i) e[i] = s * (part[i] - x[i]);
            else
            {
                const std::vector<Vec> D2 = curves (V, first, m, speed, depth, pivot); Vec u (m, 0.0);
                for (auto& ub : U) for (size_t i = 0; i < m; ++i) u[i] += ub[i];
                for (size_t i = 0; i < m; ++i) e[i] = x[i] + (1.0 + s) * lag4 (u.data(), (long) m, (double) i - D2[0][i]) - part[i];
            }
        }
        else
        {
            const std::vector<Vec> B = bands (part), D = curves (V, first, m); std::vector<Vec> Z (plain ? 0 : nv, Vec (m, 0.0)); double g[4];       // Z: per voice what it is fed with (the dry sound, each range by the voice's loudness there)
            for (int r = 0; r < NB; ++r)
            {
                for (size_t v = 0; v < nv; ++v) g[v] = V[v].gain[r];
                const Vec xb = inverse (B[(size_t) r], D, g);
                for (size_t i = 0; i < m; ++i) e[i] += B[(size_t) r][i] - xb[i];              // what the voices added in this range
                if (! plain) for (size_t v = 0; v < nv; ++v) { const double q = g[v]; double* z = Z[v].data(); for (size_t i = 0; i < m; ++i) z[i] += q * xb[i]; }
            }
            if (plain) for (auto& v : e) v *= s;
            else
            {
                const std::vector<Vec> D2 = curves (V, first, m, speed, depth, pivot);
                for (size_t i = 0; i < m; ++i) { double q = 0.0; for (size_t v = 0; v < nv; ++v) { const double p = (double) i - D2[v][i]; if (p >= 1.0) q += lag4 (Z[v].data(), (long) m, p); } e[i] = (1.0 + s) * q - e[i]; }
            }
        }
        from[(size_t) j] = a; eff[(size_t) j] = std::move (e);
    }, 4);
    for (size_t j = 0; j < jobs.size(); ++j) { const size_t a = from[j]; const Vec& e = eff[j]; for (size_t i = 0; i < e.size(); ++i) { const double w = weight (F, jobs[j], (double) (a + i)); if (w > 0.0) out[a + i] += w * e[i]; } }
    return out;
}

// The piece whose voices on side c are the clearest (nullptr: no voices on that side).
inline const Piece* bestFor (const Found& F, int c)
{
    const Piece* out = nullptr; double top = -1.0;
    for (auto& p : F.pieces) for (auto& v : p.voices[c]) { const double q = clear (v); if (q > top) { top = q; out = &p; } }
    return out;
}

// s times the found voices onto one side of another sound. The voices of the piece that shows them most clearly are used
// for the whole of the sound (their movement goes round and round, so it is known for any time), in stretches of 12 s
// that are faded into one another.
inline Vec onto (const Vec& y, const Found& F, int c, double s, double speed, double depth)
{
    Vec out = y; const size_t n = y.size(); const Piece* pc = F.found ? bestFor (F, c) : nullptr; if (pc == nullptr || n == 0 || s == 0.0) return out;
    const std::vector<Voice>& V = pc->voices[c]; const size_t nv = V.size(), XF = 4096, count = n <= LONG + LEAD ? 1 : (n + LONG - 1) / LONG;
    const bool ahead = nv == 1 && V[0].ahead; const double pivot = 0.5 * (double) F.n - (double) pc->a; std::vector<Vec> eff (count); std::vector<size_t> from (count);
    spread ((int) count, [] { return 0; }, [&] (int j, int&)
    {
        const size_t a0 = (size_t) j * LONG, a = count == 1 || a0 < LEAD ? 0 : a0 - LEAD, b = count == 1 ? n : std::min (n, a0 + LONG + XF), m = b - a;      // with a run-in
        const Vec part (y.begin() + (long) a, y.begin() + (long) b); const long first = (long) a - (long) pc->a;
        const std::vector<Vec> B = bands (part), D2 = curves (V, first, m, speed, depth, pivot); std::vector<Vec> Z (nv, Vec (m, 0.0)); Vec e (m, 0.0);       // Z: per voice what it is fed with (the sound, each range by the voice's loudness there)
        if (ahead)                                            // a copy louder than the sound: it is never put on louder than the sound came in; beyond that the sound itself is turned down
        {
            double* z = Z[0].data(); for (int r = 0; r < NB; ++r) { const double q = 1.0 / std::max (V[0].gain[r], 0.25); const double* x = B[(size_t) r].data(); for (size_t i = 0; i < m; ++i) z[i] += q * x[i]; }
            const double norm = std::max (1.0, s / std::max (V[0].loud, 0.25));
            for (size_t i = 0; i < m; ++i) e[i] = (part[i] + s * lag4 (z, (long) m, (double) i - D2[0][i])) / norm - part[i];
        }
        else
        {
            for (int r = 0; r < NB; ++r) for (size_t v = 0; v < nv; ++v) { const double q = V[v].gain[r]; double* z = Z[v].data(); const double* x = B[(size_t) r].data(); for (size_t i = 0; i < m; ++i) z[i] += q * x[i]; }
            for (size_t i = 0; i < m; ++i) { double q = 0.0; for (size_t v = 0; v < nv; ++v) { const double p = (double) i - D2[v][i]; if (p >= 1.0) q += lag4 (Z[v].data(), (long) m, p); } e[i] = s * q; }
        }
        from[(size_t) j] = a; eff[(size_t) j] = std::move (e);
    }, 4);
    for (size_t j = 0; j < count; ++j)
    {
        const size_t a = from[j]; const Vec& e = eff[j]; const double in = (double) (j * LONG), end = (double) ((j + 1) * LONG);
        for (size_t i = 0; i < e.size(); ++i)
        {
            const double pos = (double) (a + i); double w = 1.0;
            if (j > 0) w *= std::clamp ((pos - in) / (double) XF, 0.0, 1.0);
            if (j + 1 < count) w *= 1.0 - std::clamp ((pos - end) / (double) XF, 0.0, 1.0);
            if (w > 0.0) out[a + i] += w * e[i];
        }
    }
    return out;
}

inline Stereo apply (const Stereo& y, const Found& F, double s, double speed = 1.0, double depth = 1.0, bool own = true)
{
    if (! F.found) return y;
    Stereo out;
    if (y.l == y.r && F.pieces.size() && [&] { for (auto& p : F.pieces) { if (p.voices[0].size() != p.voices[1].size()) return false; for (size_t i = 0; i < p.voices[0].size(); ++i) { const Voice& a = p.voices[0][i]; const Voice& b = p.voices[1][i]; if (a.kind != b.kind || a.rate != b.rate || a.phase != b.phase || a.mid != b.mid || a.half != b.half) return false; for (int r = 0; r < NB; ++r) if (a.gain[r] != b.gain[r]) return false; } } return true; }())
    { out.l = own ? side (y.l, F, 0, s, speed, depth) : onto (y.l, F, 0, s, speed, depth); out.r = out.l; return out; }
    auto job = std::async (std::launch::async, [&] { return own ? side (y.r, F, 1, s, speed, depth) : onto (y.r, F, 1, s, speed, depth); });
    out.l = own ? side (y.l, F, 0, s, speed, depth) : onto (y.l, F, 0, s, speed, depth); out.r = job.get();
    return out;
}

// The stretches of the sound (first sample, one past the last) that have a chorus on either side.
inline std::vector<std::pair<size_t, size_t>> where (const Found& F)
{
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t k = 0; k < F.pieces.size(); ++k)
    {
        if (F.pieces[k].voices[0].empty() && F.pieces[k].voices[1].empty()) continue;
        const auto sp = span (F, k, F.n);
        if (! out.empty() && sp.first <= out.back().second) out.back().second = std::max (out.back().second, sp.second); else out.push_back (sp);
    }
    return out;
}

} // namespace chorus
} // namespace dr
