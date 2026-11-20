#pragma once
// The declipper's tools, for the bend engine (Bend.h): what it takes to rebuild the tops of a sound that were cut flat.
//   Rfft        a real sound's spectrum and back, for windows of a power of two
//   Solver      one stretch of one channel rebuilt. Some of its values are known, the others are only known to lie at
//               or beyond a bound. The answer is looked for among sounds that agree with the known values, stay beyond
//               the bounds, and need few cells of a time-frequency picture, where a cell counts with its neighbours in
//               time: a real sound is made of few partials that last a while, and the cut-off tops are what is missing
//               for them to go on smoothly (social sparsity with a persistent empirical Wiener shrinkage: Siedenburg,
//               Kowalski, Doerfler 2014). The cells' threshold comes down in steps (the schedule), a few rounds per step
//   rebuild()   a whole sound's unknown values, stretch by stretch (a long file never has its whole picture in memory),
//               on a few threads; each stretch is solved with some of its known surroundings
//   stats()     what kind of top a half-wave has: how many of its values sit exactly at the top, how many at the next
//               value down, and how dense the values just under the top are against those further down; regimes():
//               both half-waves judged together
//   curveFrom() the curve the rebuilt values tell: per level of the bent sound the middle one of the rebuilt values
//   selfTest()  the rebuilding tried on the sound itself, one step down: the known values are cut again 3 dB lower,
//               rebuilt, and held against what was there. A dense, noise-like sound fails it: its tops cannot be told
//               from what is around them
//   rankCorr()  whether one row of numbers rises with another (Spearman's rank correlation)
// The numbers are those of the prototype, Research/declip/dengine.py, where they were set on 84 sounds through
// outside clippers, distortions and saturations and on 162 more soft clippings.
#include <atomic>
#include <cstddef>
#include <functional>
#include <thread>
#include <utility>
#include "Dsp.h"

namespace dr
{
namespace declip
{
using Stop = std::function<bool()>;
using Flags = std::vector<unsigned char>;
constexpr int N0 = 2048;                        // the picture's window at 44.1 and 48 kHz; the hop is half of it
constexpr int MARGIN = 3, CORE = 192;           // hops of known sound kept around what is rebuilt; hops rebuilt in one go at most
constexpr double TOL = 1e-4, LAM_HI = -0.5, LAM_LO = -3.5;      // when a step of the schedule has settled; the schedule (powers of ten of the picture's top)
constexpr double TIE_DB = 0.01, LIVE_DB = -60.0, TIE_SHARE = 0.002, HARD_RATIO = 1.3, PILE_SHARE = 0.01, PILE_RATIO = 1.0; constexpr size_t TIE_N = 16;
constexpr double TIE_STEP = 3.0;                // a plain cut's ties stand alone: they are this many times the values at the next value down (in a sound cut down to few steps every step holds many)
constexpr double TIE_CAP = 0.75;                // a soft top with this share of its live values at the very top is left alone: there is nothing to go on
constexpr int BINS = 300, LEAST = 64, SMOOTH = 2, TEST_HOPS = 144;
constexpr int FIRST = 900, PART = 48;           // a soft top's first pass on a long sound: hops per channel it works on, in parts of this many hops spread evenly over the sound (see choose())
constexpr double TEST_DB = 3.0;

inline double db (double v) { return std::pow (10.0, v / 20.0); }
// The picture's window for a sound at this rate.
inline int window (double rate) { return (int) std::min (std::max ((double) N0 * std::pow (2.0, std::floor (std::log2 (rate / 48000.0) + 0.5)), 512.0), 8192.0); }
// How many threads the rebuilding may use.
inline size_t threads() { return (size_t) std::max (1u, std::min (8u, std::thread::hardware_concurrency())); }

// A real sound's spectrum and back: n numbers <-> n / 2 + 1 pairs (n a power of two from 16). One of these per thread.
struct Rfft
{
    int N = 0, M = 0; std::vector<int> rev; Vec wr, wi, pr, pi, re, im;
    explicit Rfft (int n) : N (n), M (n / 2), rev ((size_t) (n / 2)), wr ((size_t) (n / 2)), wi ((size_t) (n / 2)), pr ((size_t) (n / 2 + 1)), pi ((size_t) (n / 2 + 1)), re ((size_t) (n / 2)), im ((size_t) (n / 2))
    {
        int bits = 0; while ((1 << bits) < M) ++bits;
        for (int i = 0; i < M; ++i) { int r = 0; for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b); rev[(size_t) i] = r; }
        for (int h = 1; h < M; h <<= 1) for (int j = 0; j < h; ++j) { const double a = -PI * (double) j / (double) h; wr[(size_t) (h - 1 + j)] = std::cos (a); wi[(size_t) (h - 1 + j)] = std::sin (a); }
        for (int k = 0; k <= M; ++k) { const double a = -2.0 * PI * (double) k / (double) N; pr[(size_t) k] = std::cos (a); pi[(size_t) k] = std::sin (a); }
    }
    // M complex numbers in re, im (laid out in bit-reversed order) -> their spectrum, in place
    void core()
    {
        double* r = re.data(); double* q = im.data();
        for (int i = 0; i < M; i += 4)          // the first two rounds in one: their turns are 1 and -i
        {
            const double a0 = r[i] + r[i + 1], a1 = r[i] - r[i + 1], b0 = q[i] + q[i + 1], b1 = q[i] - q[i + 1], c0 = r[i + 2] + r[i + 3], c1 = r[i + 2] - r[i + 3], d0 = q[i + 2] + q[i + 3], d1 = q[i + 2] - q[i + 3];
            r[i] = a0 + c0; q[i] = b0 + d0; r[i + 2] = a0 - c0; q[i + 2] = b0 - d0; r[i + 1] = a1 + d1; q[i + 1] = b1 - c1; r[i + 3] = a1 - d1; q[i + 3] = b1 + c1;
        }
        for (int h = 4; h < M; h <<= 1)
        {
            const double* cr = wr.data() + (h - 1); const double* ci = wi.data() + (h - 1);
            for (int i = 0; i < M; i += 2 * h)
            {
                double* ar = r + i; double* ai = q + i; double* br = ar + h; double* bi = ai + h;
                for (int j = 0; j < h; ++j)
                {
                    const double tr = br[j] * cr[j] - bi[j] * ci[j], ti = br[j] * ci[j] + bi[j] * cr[j];
                    br[j] = ar[j] - tr; bi[j] = ai[j] - ti; ar[j] += tr; ai[j] += ti;
                }
            }
        }
    }
    // x: N numbers -> X: N / 2 + 1 pairs (the even and the odd samples go through one transform of half the size)
    void forward (const double* x, double* Xr, double* Xi)
    {
        for (int k = 0; k < M; ++k) { const size_t p = (size_t) rev[(size_t) k]; re[p] = x[2 * k]; im[p] = x[2 * k + 1]; }
        core();
        for (int k = 0; k <= M; ++k)
        {
            const size_t a = (size_t) (k == M ? 0 : k), b = (size_t) (k == 0 ? 0 : M - k); const double zr = re[a], zi = im[a], cr = re[b], ci = -im[b], dr = zr - cr, di = zi - ci;
            Xr[k] = 0.5 * (zr + cr + pr[(size_t) k] * di + pi[(size_t) k] * dr); Xi[k] = 0.5 * (zi + ci - pr[(size_t) k] * dr + pi[(size_t) k] * di);
        }
    }
    // the way back: what forward() made of x gives x again
    void inverse (const double* Xr, const double* Xi, double* x)
    {
        for (int k = 0; k < M; ++k)
        {
            const double ar = Xr[k], ai = Xi[k], br = Xr[M - k], bi = Xi[M - k], er = 0.5 * (ar + br), ei = 0.5 * (ai - bi), dr = ar - br, di = ai + bi;
            const double orr = 0.5 * (dr * pr[(size_t) k] + di * pi[(size_t) k]), oi = 0.5 * (di * pr[(size_t) k] - dr * pi[(size_t) k]); const size_t p = (size_t) rev[(size_t) k];
            re[p] = er - oi; im[p] = -(ei + orr);
        }
        core(); const double s = 1.0 / (double) M;
        for (int m = 0; m < M; ++m) { x[2 * m] = re[(size_t) m] * s; x[2 * m + 1] = -im[(size_t) m] * s; }
    }
};

// The picture: windows of N samples every N / 2, a sine window, so that going to the picture and back gives the sound
// again and the picture's energy is the sound's. Frame t of a sound of n samples holds its samples t * H - N and on
// (nothing lies before and after the sound). One Solver per thread.
struct Solver
{
    int N, H, B; Rfft fft; Vec w, fr, Ar, Ai, Zr, Zi, Ur, Ui, P, xs, gs;
    explicit Solver (int n) : N (n), H (n / 2), B (n / 2 + 1), fft (n), w ((size_t) n), fr ((size_t) n) { for (int i = 0; i < N; ++i) w[(size_t) i] = std::sin (PI * ((double) i + 0.5) / (double) N); }
    size_t frames (size_t n) const { return (n + (size_t) N) / (size_t) H + 2; }
    // the part of frame t that lies in the sound: [j0, j1) of the frame, sample s0 + j
    bool part (size_t t, size_t n, long& s0, long& j0, long& j1) const { s0 = (long) t * (long) H - (long) N; j0 = std::max (0L, -s0); j1 = std::min ((long) N, (long) n - s0); return j1 > j0; }
    bool frame (const double* x, size_t n, size_t t, double* r, double* q)
    {
        long s0, j0, j1; if (! part (t, n, s0, j0, j1)) return false;
        const double k = 1.0 / std::sqrt ((double) N);
        for (long j = 0; j < j0; ++j) fr[(size_t) j] = 0.0;
        for (long j = j0; j < j1; ++j) fr[(size_t) j] = x[s0 + j] * w[(size_t) j] * k;
        for (long j = j1; j < (long) N; ++j) fr[(size_t) j] = 0.0;
        fft.forward (fr.data(), r, q); return true;
    }
    void fwd (const double* x, size_t n, double* Xr, double* Xi)
    {
        const size_t T = frames (n);
        for (size_t t = 0; t < T; ++t)
        {
            double* r = Xr + t * (size_t) B; double* q = Xi + t * (size_t) B;
            if (! frame (x, n, t, r, q)) { std::fill (r, r + B, 0.0); std::fill (q, q + B, 0.0); }
        }
    }
    void back (const double* Xr, const double* Xi, double* x, size_t n)
    {
        std::fill (x, x + n, 0.0); const size_t T = frames (n); const double k = std::sqrt ((double) N);
        for (size_t t = 0; t < T; ++t)
        {
            long s0, j0, j1; if (! part (t, n, s0, j0, j1)) continue;
            fft.inverse (Xr + t * (size_t) B, Xi + t * (size_t) B, fr.data());
            for (long j = j0; j < j1; ++j) x[s0 + j] += fr[(size_t) j] * w[(size_t) j] * k;
        }
    }
    // The level the schedule hangs on: 999 of 1000 of the picture's cells are weaker (read off a table in steps of 0.05 dB).
    double topOf (const double* x, size_t n)
    {
        std::vector<size_t> hist (6000, 0); const size_t T = frames (n); Vec r ((size_t) B), q ((size_t) B);
        for (size_t t = 0; t < T; ++t)
        {
            if (! frame (x, n, t, r.data(), q.data())) { hist[0] += (size_t) B; continue; }
            for (int b = 0; b < B; ++b)
            {
                const double p = r[(size_t) b] * r[(size_t) b] + q[(size_t) b] * q[(size_t) b];
                ++hist[(size_t) std::clamp ((long) std::floor ((10.0 * std::log10 (std::max (p, 1e-30)) + 240.0) / 0.05), 0L, 5999L)];
            }
        }
        const double want = 0.999 * (double) (T * (size_t) B); size_t c = 0, k = 0;
        for (; k < 5999; ++k) { c += hist[k]; if ((double) c >= want) break; }
        return std::sqrt (std::pow (10.0, (-240.0 + 0.05 * (double) (k + 1)) / 10.0));
    }
    // One stretch. y: its known values and, where u says unknown, the bound (its sign says on which side of zero; the
    // value lies at the bound or further out). lams: the schedule, inner: rounds per step. out: the stretch rebuilt, the
    // known values as they are. False when given up.
    bool solve (const double* y, const unsigned char* u, size_t n, const double* lams, int nlam, int inner, double* out, const Stop& stop)
    {
        const size_t T = frames (n), C = T * (size_t) B;
        for (Vec* v : { &Ar, &Ai, &Zr, &Zi, &Ur, &Ui, &P }) v->resize (C);
        xs.resize (n); gs.resize (n);
        fwd (y, n, Ar.data(), Ai.data()); Zr = Ar; Zi = Ai; double t = 1.0;
        for (int l = 0; l < nlam; ++l)
        {
            const double lam2 = lams[l] * lams[l];
            for (int it = 0; it < inner; ++it)
            {
                if (stop && stop()) return false;
                back (Zr.data(), Zi.data(), xs.data(), n);
                for (size_t i = 0; i < n; ++i)          // what is missing towards the known values and the bounds
                {
                    const double d = y[i] - xs[i]; gs[i] = ! u[i] ? d : y[i] > 0.0 ? std::max (d, 0.0) : std::min (d, 0.0);
                }
                fwd (gs.data(), n, Ur.data(), Ui.data());
                for (size_t c = 0; c < C; ++c) { Ur[c] += Zr[c]; Ui[c] += Zi[c]; P[c] = Ur[c] * Ur[c] + Ui[c] * Ui[c]; }
                // a cell is kept by 1 - lam^2 / (the mean energy of itself and its two neighbours in time), never under 0;
                // then a step on from the last answer in the direction it moved (the momentum that makes few rounds do)
                const double tn = 0.5 * (1.0 + std::sqrt (1.0 + 4.0 * t * t)), m = (t - 1.0) / tn; double dn = 0.0, en = 0.0;
                for (size_t tt = 0; tt < T; ++tt)
                {
                    const double* p0 = P.data() + (tt == 0 ? 0 : tt - 1) * (size_t) B; const double* p1 = P.data() + tt * (size_t) B; const double* p2 = P.data() + (tt + 1 == T ? tt : tt + 1) * (size_t) B;
                    const size_t o = tt * (size_t) B;
                    for (int b = 0; b < B; ++b)
                    {
                        const size_t c = o + (size_t) b; const double gain = std::max (1.0 - lam2 / std::max ((p0[b] + p1[b] + p2[b]) / 3.0, 1e-30), 0.0);
                        const double nr = Ur[c] * gain, ni = Ui[c] * gain, dr = nr - Ar[c], di = ni - Ai[c], wgt = (b == 0 || b == B - 1) ? 1.0 : 2.0;
                        dn += wgt * (dr * dr + di * di); en += wgt * (nr * nr + ni * ni);
                        Zr[c] = nr + m * dr; Zi[c] = ni + m * di; Ar[c] = nr; Ai[c] = ni;
                    }
                }
                t = tn;
                if (std::sqrt (dn / std::max (en, 1e-30)) < TOL) break;
            }
        }
        back (Ar.data(), Ai.data(), xs.data(), n);
        for (size_t i = 0; i < n; ++i) out[i] = ! u[i] ? y[i] : y[i] > 0.0 ? std::max (xs[i], y[i]) : std::min (xs[i], y[i]);
        return true;
    }
};

// What is rebuilt in one go: the stretch [a, b) and the part [ca, cb) of it whose unknown values are taken from it.
// Unknown values less than 2 * MARGIN hops apart belong together; a long run is cut into equal parts of at most CORE
// hops, each solved with MARGIN hops of its surroundings.
struct Piece { size_t a = 0, b = 0, ca = 0, cb = 0; };
inline std::vector<Piece> pieces (const Flags& u, int H, int core = CORE)
{
    std::vector<Piece> out; const size_t n = u.size(), M = (size_t) MARGIN * (size_t) H, most = (size_t) core * (size_t) H;
    auto run = [&] (size_t a, size_t b)
    {
        const size_t k = (b - a + most - 1) / most, L = (b - a + k - 1) / k;
        for (size_t j = 0; j < k; ++j) { const size_t ca = a + j * L, cb = std::min (b, ca + L); out.push_back ({ ca > M ? ca - M : 0, std::min (n, cb + M), ca, cb }); }
    };
    bool open = false; size_t first = 0, last = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (! u[i]) continue;
        if (open && i - last > 2 * M) { run (first, last + 1); open = false; }
        if (! open) { first = i; open = true; }
        last = i;
    }
    if (open) run (first, last + 1);
    return out;
}

// A long sound's first pass (a soft flat top's, which costs the most rounds and whose use is to tell the curve). A
// channel whose unknown values span more than `first` hops: they are cut into parts of at most `part` hops, and of
// these every so-manieth is worked on, spread evenly over the sound, `first` hops in all. The curve they tell serves
// the whole sound. ps: per channel the pieces to work on; seen: per channel and sample whether it lies in one of
// their own parts. True: nothing is left out (any sound of up to `first` hops per channel): ps holds pieces().
inline bool choose (const Flags u[2], int H, size_t first, int part, std::vector<Piece> ps[2], Flags seen[2])
{
    bool whole = true;
    for (int c = 0; c < 2; ++c)
    {
        ps[c] = pieces (u[c], H); seen[c].assign (u[c].size(), 0); size_t span = 0; for (const Piece& p : ps[c]) span += p.cb - p.ca;
        if (span > first * (size_t) H)
        {
            const std::vector<Piece> small = pieces (u[c], H, part); const size_t m = std::min (small.size(), (first + (size_t) part - 1) / (size_t) part);
            if (m < small.size()) { ps[c].clear(); for (size_t j = 0; j < m; ++j) ps[c].push_back (small[(2 * j + 1) * small.size() / (2 * m)]); whole = false; }
        }
        for (const Piece& p : ps[c]) std::fill (seen[c].begin() + (std::ptrdiff_t) p.ca, seen[c].begin() + (std::ptrdiff_t) p.cb, (unsigned char) 1);
    }
    return whole;
}

// A sound's unknown values rebuilt. x: per channel the known values and, where u says unknown, the bound (see
// Solver::solve). nlam, inner: steps of the schedule, rounds per step. N: the window. only: per channel the pieces
// that are worked on (null: all of pieces()). False when given up, or when the work could not be done (out is then
// of no use).
inline bool rebuild (const Stereo& x, const Flags u[2], Stereo& out, int nlam, int inner, int N, const Stop& stop, const std::vector<Piece>* only = nullptr)
{
    out = x; const Vec* xs[2] = { &x.l, &x.r }; Vec* os[2] = { &out.l, &out.r };
    const bool twin = x.l == x.r && u[0] == u[1];          // both channels alike (a mono file): one is rebuilt
    struct Job { int c; Piece p; }; std::vector<Job> jobs; Vec lams[2];
    {
        Solver S (N);
        for (int c = 0; c < (twin ? 1 : 2); ++c)
        {
            const std::vector<Piece> ps = only != nullptr ? only[c] : pieces (u[c], N / 2); if (ps.empty()) continue;
            const double top = S.topOf (xs[c]->data(), xs[c]->size());
            for (int i = 0; i < nlam; ++i) lams[c].push_back (top * std::pow (10.0, nlam > 1 ? LAM_HI + (LAM_LO - LAM_HI) * (double) i / (double) (nlam - 1) : LAM_HI));
            for (const Piece& p : ps) jobs.push_back ({ c, p });
        }
    }
    std::atomic<size_t> next { 0 }; std::atomic<bool> bad { false };
    auto work = [&]
    {
        try
        {
            Solver S (N); Vec o;
            for (;;)
            {
                const size_t j = next.fetch_add (1); if (j >= jobs.size() || bad.load()) break;
                const Job& J = jobs[j]; const Piece& p = J.p; o.resize (p.b - p.a);
                if (! S.solve (xs[J.c]->data() + p.a, u[J.c].data() + p.a, p.b - p.a, lams[J.c].data(), nlam, inner, o.data(), stop)) { bad = true; break; }
                for (size_t i = p.ca; i < p.cb; ++i) if (u[J.c][i]) (*os[J.c])[i] = o[i - p.a];
            }
        }
        catch (...) { bad = true; }
    };
    std::vector<std::thread> th; const size_t nt = std::min (jobs.size(), threads());
    for (size_t i = 1; i < nt; ++i) { try { th.emplace_back (work); } catch (...) { break; } }
    work(); for (std::thread& t : th) t.join();
    if (bad.load()) return false;
    if (twin) out.r = out.l;
    return true;
}

// ---------------------------------------------------------------- what kind of top a half-wave has
// c: its top value. tn, t: the number and the share of its live values (those above LIVE_DB under its peak vp) within
// TIE_DB of the top. tn2: the number within TIE_DB of the next value down (the highest one under the ties). p25: the
// share within 0.25 dB of the peak. R: the density of values (per dB) just under the ties against that 1 to 2 dB under
// the top. P: the density in the top quarter dB under the peak (ties and all) against that 1 to 2 dB under the peak.
struct Stat { double c = 0, t = 0, p25 = 0, R = 0, P = 0; size_t n = 1, tn = 0, tn2 = 0; };
inline Stat stats (const Stereo& y, double pol, double vp)
{
    Stat s; const double live = vp * db (LIVE_DB); size_t n = 0; double c = 0.0;
    for (const Vec* ch : { &y.l, &y.r }) for (double v : *ch) { const double a = pol * v; if (a > live) { ++n; c = std::max (c, a); } }
    s.c = c; s.n = std::max<size_t> (n, 1); if (n == 0) return s;
    const double tie = c * db (-TIE_DB), q25 = c * db (-0.25), l1 = c * db (-1.0), l2 = c * db (-2.0), p25 = vp * db (-0.25), r1 = vp * db (-1.0), r2 = vp * db (-2.0);
    size_t tn = 0, dt = 0, dl = 0, np = 0, pl = 0; double c2 = 0.0;
    for (const Vec* ch : { &y.l, &y.r }) for (double v : *ch)
    {
        const double a = pol * v; if (! (a > live)) continue;
        if (a >= tie) ++tn; else { c2 = std::max (c2, a); if (a >= q25) ++dt; }
        if (a >= l2 && a < l1) ++dl;
        if (a >= p25) ++np;
        if (a >= r2 && a < r1) ++pl;
    }
    const double next = c2 * db (-TIE_DB);
    if (c2 > 0.0) for (const Vec* ch : { &y.l, &y.r }) for (double v : *ch) { const double a = pol * v; if (a > live && a < tie && a >= next) ++s.tn2; }
    const double N = (double) s.n; s.tn = tn; s.t = (double) tn / N; s.p25 = (double) np / N;
    s.R = ((double) dt / N / (0.25 - TIE_DB)) / std::max ((double) dl / N, 1e-9); s.P = (s.p25 / 0.25) / std::max ((double) pl / N, 1e-9);
    return s;
}
inline bool tied (const Stat& s) { return s.tn >= TIE_N && s.t >= TIE_SHARE; }
// One half-wave by itself. 1: a plain cut (its ties stand alone: the next value down holds far fewer). 2: a soft flat
// top (only looked for where the meter has found a bend, and not where TIE_CAP of the live values sit at the very
// top). 0: neither.
inline bool softLike (const Stat& s, bool kept) { return kept && s.p25 >= PILE_SHARE && s.P >= PILE_RATIO && s.t < TIE_CAP; }
inline int regime (const Stat& s, bool kept) { return tied (s) && s.R <= HARD_RATIO && (double) s.tn >= TIE_STEP * (double) s.tn2 ? 1 : softLike (s, kept) ? 2 : 0; }
// Both half-waves: one wave, one way. A side that reads as a plain cut next to a soft one, and fits a soft top too, is
// taken as soft (its ties are then the first height).
inline void regimes (const Stat* st, bool kept, int* kinds)
{
    for (int h = 0; h < 2; ++h) kinds[h] = regime (st[h], kept);
    for (int h = 0; h < 2; ++h) if (kinds[h] == 1 && kinds[1 - h] == 2 && softLike (st[h], kept)) kinds[h] = 2;
}

// ---------------------------------------------------------------- the curve the rebuilt values tell
inline double median (Vec& v)       // the middle value (of an even number: the mean of the middle two); v is reordered
{
    const size_t n = v.size(), h = n / 2; std::nth_element (v.begin(), v.begin() + (std::ptrdiff_t) h, v.end()); const double hi = v[h];
    return n % 2 ? hi : 0.5 * (hi + *std::max_element (v.begin(), v.begin() + (std::ptrdiff_t) h));
}
inline double quantile (Vec v, double p)
{
    std::sort (v.begin(), v.end()); const double t = p * (double) (v.size() - 1); const size_t i = (size_t) std::floor (t), j = std::min (i + 1, v.size() - 1);
    return v[i] + (t - (double) i) * (v[j] - v[i]);
}
// Spearman's rank correlation of the pairs' two sides (equal values share their mean rank). 0 for fewer than three pairs.
inline double rankCorr (const std::vector<std::pair<double, double>>& pr)
{
    const size_t n = pr.size(); if (n < 3) return 0.0;
    std::vector<size_t> idx (n); Vec ra (n), rb (n);
    auto ranks = [&pr, &idx, n] (bool second, Vec& r)
    {
        auto val = [&pr, second] (size_t k) { return second ? pr[k].second : pr[k].first; };
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        std::sort (idx.begin(), idx.end(), [&val] (size_t a, size_t b) { return val (a) < val (b); });
        for (size_t i = 0; i < n;)
        {
            size_t j = i; while (j + 1 < n && val (idx[j + 1]) == val (idx[i])) ++j;
            const double m = 0.5 * (double) (i + j); for (size_t k = i; k <= j; ++k) r[idx[k]] = m;
            i = j + 1;
        }
    };
    ranks (false, ra); ranks (true, rb);
    const double mean = 0.5 * (double) (n - 1); double saa = 0.0, sbb = 0.0, sab = 0.0;
    for (size_t i = 0; i < n; ++i) { const double a = ra[i] - mean, b = rb[i] - mean; saa += a * a; sbb += b * b; sab += a * b; }
    const double d = std::sqrt (saa * sbb); return d > 0.0 ? sab / d : 0.0;
}
// The curve from (y0, x0) on, read from pairs (bent value, rebuilt value), both positive: nodes (ym, xm). The pairs in
// order of the bent value, in up to BINS groups of equal size (LEAST pairs at least); per group the middle values;
// the row made rising, smoothed over SMOOTH groups to each side, made rising again.
inline void curveFrom (std::vector<std::pair<double, double>>& pr, double y0, double x0, Vec& ym, Vec& xm)
{
    std::stable_sort (pr.begin(), pr.end(), [] (const std::pair<double, double>& a, const std::pair<double, double>& b) { return a.first < b.first; });
    const size_t n = pr.size(), nb = std::max<size_t> (1, std::min<size_t> ((size_t) BINS, n / (size_t) LEAST)); Vec my { y0 }, mx { x0 }, w { 1e9 }, tmp;
    for (size_t k = 0; k < nb; ++k)
    {
        const size_t a = k * n / nb, b = (k + 1) * n / nb, m = b - a; if (b <= a) continue;
        my.push_back (m % 2 ? pr[a + m / 2].first : 0.5 * (pr[a + m / 2 - 1].first + pr[a + m / 2].first));
        tmp.resize (m); for (size_t i = 0; i < m; ++i) tmp[i] = pr[a + i].second;
        mx.push_back (std::max (median (tmp), x0)); w.push_back ((double) m);
    }
    {       // the closest rising row
        Vec val, wt; std::vector<size_t> cnt;
        for (size_t i = 0; i < mx.size(); ++i)
        {
            val.push_back (mx[i]); wt.push_back (w[i]); cnt.push_back (1);
            while (val.size() > 1 && val[val.size() - 2] > val.back())
            {
                const size_t e = val.size() - 1; val[e - 1] = (val[e - 1] * wt[e - 1] + val[e] * wt[e]) / (wt[e - 1] + wt[e]); wt[e - 1] += wt[e]; cnt[e - 1] += cnt[e];
                val.pop_back(); wt.pop_back(); cnt.pop_back();
            }
        }
        size_t o = 0; for (size_t i = 0; i < val.size(); ++i) for (size_t j = 0; j < cnt[i]; ++j) mx[o++] = val[i];
    }
    if (mx.size() > (size_t) (2 * SMOOTH + 1))
    {
        Vec s (mx.size()); const long L = (long) mx.size();
        for (long i = 0; i < L; ++i) { double a = 0.0; for (long j = -SMOOTH; j <= SMOOTH; ++j) a += mx[(size_t) std::clamp (i + j, 0L, L - 1)] * (1.0 / (double) (2 * SMOOTH + 1)); s[(size_t) i] = a; }
        s[0] = x0; for (size_t i = 1; i < s.size(); ++i) s[i] = std::max (s[i], s[i - 1]);
        mx = s;
    }
    ym.clear(); xm.clear();
    for (size_t i = 0; i < my.size(); ++i) if (i == 0 || my[i] > my[i - 1]) { ym.push_back (my[i]); xm.push_back (mx[i]); }
}
// Straight pieces between nodes (xs rising), the first and the last value outside them.
inline double lin (const double* xs, const double* ys, size_t n, double v)
{
    if (v <= xs[0]) return ys[0];
    if (v >= xs[n - 1]) return ys[n - 1];
    const size_t i = (size_t) (std::upper_bound (xs, xs + n, v) - xs) - 1;
    return ys[i] + (ys[i + 1] - ys[i]) * (v - xs[i]) / (xs[i + 1] - xs[i]);
}
inline double lin (const Vec& xs, const Vec& ys, double v) { return lin (xs.data(), ys.data(), xs.size(), v); }

// ---------------------------------------------------------------- the test on the sound itself
// A plain picture of a sound: power per frame (a window of N every N / 4) in 48 bands of equal width on a mel scale
// from 30 Hz to 20 kHz, both channels after one another. Rows of 48.
constexpr int BANDS = 48;
inline Vec picture (const Stereo& x, double rate, int N)
{
    Vec P; const size_t n = x.size(); if (n < (size_t) N) return P;
    auto mel = [] (double f) { return 2595.0 * std::log10 (1.0 + f / 700.0); }; auto inv = [] (double m) { return 700.0 * (std::pow (10.0, m / 2595.0) - 1.0); };
    int e[BANDS + 1]; const double m0 = mel (30.0), m1 = mel (std::min (20000.0, 0.5 * rate));
    for (int i = 0; i <= BANDS; ++i) e[i] = (int) std::clamp ((long) std::floor (inv (m0 + (m1 - m0) * (double) i / (double) BANDS) * (double) N / rate + 0.5), 1L, (long) (N / 2));
    Rfft f (N); Vec w ((size_t) N), fr ((size_t) N), r ((size_t) (N / 2 + 1)), q ((size_t) (N / 2 + 1)), cs ((size_t) (N / 2 + 2));
    for (int j = 0; j < N; ++j) w[(size_t) j] = 0.5 - 0.5 * std::cos (2.0 * PI * (double) j / (double) N);
    const size_t hop = (size_t) (N / 4), T = (n - (size_t) N) / hop + 1; P.reserve (2 * T * (size_t) BANDS);
    for (const Vec* ch : { &x.l, &x.r })
        for (size_t t = 0; t < T; ++t)
        {
            const double* s = ch->data() + t * hop; for (int j = 0; j < N; ++j) fr[(size_t) j] = s[j] * w[(size_t) j];
            f.forward (fr.data(), r.data(), q.data()); cs[0] = 0.0;
            for (int k = 0; k <= N / 2; ++k) cs[(size_t) (k + 1)] = cs[(size_t) k] + r[(size_t) k] * r[(size_t) k] + q[(size_t) k] * q[(size_t) k];
            for (int b = 0; b < BANDS; ++b) P.push_back (cs[(size_t) std::max (e[b + 1], e[b] + 1)] - cs[(size_t) e[b]]);
        }
    return P;
}
// How far picture B is from A (the reference): the mean difference in dB after the best level match, cells more than
// 60 dB under the reference's loudest counting as the floor, no cell for more than 30 dB.
inline double picDist (const Vec& A, const Vec& B)
{
    const size_t n = std::min (A.size(), B.size()); if (n == 0) return 0.0;
    double top = 0.0; for (size_t i = 0; i < n; ++i) top = std::max (top, A[i]);
    const double floor = (top + 1e-20) * 1e-6; Vec la (n), lb (n), d; double low = 1e300;
    for (size_t i = 0; i < n; ++i) { la[i] = 10.0 * std::log10 (std::max (A[i], floor)); lb[i] = 10.0 * std::log10 (std::max (B[i], 1e-30)); low = std::min (low, la[i]); }
    for (size_t i = 0; i < n; ++i) if (la[i] > low + 6.0) d.push_back (la[i] - lb[i]);
    const double g = d.empty() ? 0.0 : median (d); double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += std::min (std::abs (la[i] - std::max (lb[i] + g, low)), 30.0);
    return sum / (double) n;
}
// The rebuilding tried on the sound itself, one step down. held: the sound with its unknown values at their bounds bp
// (above zero) and bn (below; infinite where a side has none). The known values are cut again TEST_DB under the bounds,
// rebuilt from below, and held against what was there: 100 = the rebuilt band matches, 0 = no better than the cut, below
// 0 = worse. (Above the real bound the rebuilt values are cut at the bound, as the sound in hand is.) Tried on the
// TEST_HOPS hops of the sound that hold the most values above the test's cut, with the rounds the rebuilding itself
// has. False when given up.
inline bool selfTest (const Stereo& held, double bp, double bn, int nlam, int inner, double rate, int N, double& score, const Stop& stop)
{
    const double k = db (-TEST_DB), cp = bp * k, cn = bn * k; const size_t n = held.size(), H = (size_t) (N / 2), nh = n / H; size_t a = 0, len = n;
    auto beyond = [cp, cn] (double v) { return v >= cp || v <= cn; };
    if (nh > (size_t) TEST_HOPS)
    {
        std::vector<size_t> cs (nh + 1, 0);
        for (size_t h = 0; h < nh; ++h) { size_t c = 0; for (size_t i = h * H; i < (h + 1) * H; ++i) c += (size_t) beyond (held.l[i]) + (size_t) beyond (held.r[i]); cs[h + 1] = cs[h] + c; }
        size_t best = 0, at = 0; for (size_t h = 0; h + (size_t) TEST_HOPS <= nh; ++h) { const size_t c = cs[h + (size_t) TEST_HOPS] - cs[h]; if (h == 0 || c > best) { best = c; at = h; } }
        a = at * H; len = (size_t) TEST_HOPS * H;
    }
    Stereo ref, cut, out; Flags u[2]; ref.l.assign (held.l.begin() + (std::ptrdiff_t) a, held.l.begin() + (std::ptrdiff_t) (a + len)); ref.r.assign (held.r.begin() + (std::ptrdiff_t) a, held.r.begin() + (std::ptrdiff_t) (a + len)); cut = ref;
    const Vec* rs[2] = { &ref.l, &ref.r }; Vec* cs2[2] = { &cut.l, &cut.r };
    for (int c = 0; c < 2; ++c) { u[c].resize (len); for (size_t i = 0; i < len; ++i) { const double v = (*rs[c])[i]; u[c][i] = (unsigned char) beyond (v); (*cs2[c])[i] = std::min (std::max (v, cn), cp); } }
    if (! rebuild (cut, u, out, nlam, inner, N, stop)) return false;
    for (Vec* ch : { &out.l, &out.r }) for (double& v : *ch) v = std::min (std::max (v, bn), bp);
    const Vec PA = picture (ref, rate, N); const double d0 = picDist (PA, picture (cut, rate, N)), d1 = picDist (PA, picture (out, rate, N));
    score = 100.0 * (1.0 - d1 / std::max (d0, 1e-9)); return true;
}

} // namespace declip
} // namespace dr
