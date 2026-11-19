#pragma once
// The sides engine (HAAS WIDTH, STEREO WIDTH): what was done between the two sides of a sound.
//   find()    reads off one sound whether one side was made late (HAAS) or a sound from the middle was made wide (WIDTH),
//             and with what: the two short filters through which one sound reaches the two sides, or the filter through
//             which a copy of the middle was put at the sides; where along the sound it holds
//   own()     the sound itself with the group's Amount at s: 0 = the effect out, 1 = untouched, 2 = twice
//   onto()    the found effect on another sound: 0 = none, 1 = as found, 2 = twice
//   where()   the stretches of the sound that have it (for the line under the waveform)
// How:
//  (A) "the pair": one sound x reaches the two sides through two short, sparse filters: left = a * x, right = b * x, so
//      left * b = right * a. Both sides are brought to one level and sent through one long filter that flattens their
//      average spectrum (a steady note would otherwise explain itself from its own past as well as the other side
//      explains it). The taps (0 .. 40 ms) are looked for with a penalty on their number, over the whole sound; then
//      fitted exactly on the samples where every tap is inside the file, in pieces of 5 ms of which those that do not
//      obey the pair (a click, a plugin's glitch) are left out; the taps that explain next to nothing go. This is done
//      both ways round (the right side told from the left one and from its own past, and the other way).
//  (B) "the copy": the middle kept and a filtered copy of it put at the sides: side = h * middle (taps -5 .. +5 ms).
// Both are then checked on the sound itself: x = (A* L + B* R) / (|A|^2 + |B|^2) (or the copy) is worked out and what
// the two sides hold beyond it is measured piece by piece. That says where along the sound the effect is and how much of
// the sound it explains, and it chooses the way round. A pair counts when it has few taps that reach at least 1.25 ms,
// makes a side worth speaking of, is obeyed by half the sound or more, and explains far more than one sound at two
// levels (a sound simply panned somewhere) would. HAAS: the two sides' main taps are 1.25 ms or more apart; else WIDTH.
// The Amount fader: the sound is the rest plus the pair's part, and only the pair's part is changed, so the middle of
// the fader is the sound itself, bit for bit. One tap on each side (a plain delay): the late side moves in time and the
// levels come together, so that no comb is heard on the way. Anything else: a straight line between the plain sound and
// the pair; to the right everything but each side's direct sound grows.
// What cannot be read off a sound: a side that was simply made louder (nothing ties it to the middle).
// A sound longer than 60 s: the pair is looked for in the 60 s with the most energy, then checked on all of it.
// A port of the Python prototype Research/sides/sides.py (Tools/sides_parity.py). 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "Room.h"      // dr::Vec, dr::Stereo (Dsp.h) and room::Transform

namespace dr
{
namespace sides
{
using Stop = std::function<bool()>;                     // asked now and then while measuring: true = give up (the answer is then not used)
constexpr int K = 1920;                                 // (A): taps looked for at 0 .. 40 ms
constexpr int ORDER = 32;                               // (B): the whitening filter (linear prediction on the middle)
constexpr int WORD = 1920;                              // (A): the whitening filter both sides share (40 ms long)
constexpr int BLK = 240;                                // pieces of 5 ms
constexpr int MOST = 56;                                // (A): unknowns in the exact fit, at most
constexpr int NEAR = 24;                                // (A): a side's own past may only be used from 0.5 ms back
constexpr int PIECES_MOST = 1800;                       // (A): pieces in the exact fit, at most (a longer sound: evenly spread)
constexpr int KB = 240;                                 // (B): taps of h at -5 .. +5 ms
constexpr int NF = 1 << 17;                             // transform size of the block work
constexpr int CB = NF / 2 - 2 * K;                      // samples per block of the correlations
constexpr long LONG = 60 * 48000;                       // (A): a longer sound: looked for in the 60 s with the most energy
constexpr long LEAST = K + WORD + 20 * BLK;             // a sound under 180 ms: nothing is looked for
constexpr int APART = 60;                               // HAAS: the two sides' main taps at least 1.25 ms apart
constexpr int HAAS = 1, WIDTH = 2;                      // Found::kind
constexpr int PAIR = 1, COPY = 2;                       // Found::mode

struct Tap { int lag; double v; };
using Taps = std::vector<Tap>;                          // a sparse filter: taps in the order of their lags
using Spans = std::vector<std::pair<int, int>>;         // in pieces of 5 ms: first piece, one past the last

// The rule: what counts as found.
struct Rule { int taps = 8, reach = 60; double made = -20.0, covered = 0.5, inside = -20.0, beyond = -10.0, share = 0.9, side = -15.0, copyCovered = 0.5; };

struct Found
{
    int kind = 0, mode = 0;                             // nothing, HAAS or WIDTH; read as the pair or as the copy
    size_t n = 0;                                       // the length of the sound it was read off
    Taps a, b;                                          // the pair: left = a * x, right = b * x (the largest tap of all = 1)
    Vec h;                                              // the copy: side = h * middle, 2 KB + 1 taps, index KB = no delay
    std::vector<float> x;                               // the pair: the one sound behind both sides; the copy: the copy itself (all n samples)
    Spans spans;                                        // where along the sound it holds
    // the numbers behind the verdict
    bool hasPair = false, hasCopy = false, copyChecked = false;
    int way = 0, taps = 0, apart = 0, reach = 0;        // which way round the pair was fitted (0: the right side told from the left); its shape
    double res = 1.0, fits = 0.0, made = -120.0, covered = 0.0, inside = 1.0, beyond = 0.0;
    double share = 0.0, all = 0.0, balance = 0.0, sideDb = 0.0, copyCovered = 0.0, copyInside = 1.0;
    bool found() const { return kind != 0; }
};

// ---------------------------------------------------------------- transforms
struct Spec { Vec re, im; };                            // of a real signal: the points 0 .. NF / 2
struct Work
{
    room::Transform T { NF }; Vec re, im;
    Work() : re ((size_t) NF, 0.0), im ((size_t) NF, 0.0) {}
    void clear() { std::fill (re.begin(), re.end(), 0.0); std::fill (im.begin(), im.end(), 0.0); }
    // re and im hold two real signals: their spectra
    void spectra (Spec& P, Spec& Q)
    {
        T.run (re, im); const int H = NF / 2; P.re.resize ((size_t) H + 1); P.im.resize ((size_t) H + 1); Q.re.resize ((size_t) H + 1); Q.im.resize ((size_t) H + 1);
        for (int k = 0; k <= H; ++k)
        {
            const int m = (NF - k) & (NF - 1); const double zr = re[(size_t) k], zi = im[(size_t) k], cr = re[(size_t) m], ci = -im[(size_t) m];
            P.re[(size_t) k] = 0.5 * (zr + cr); P.im[(size_t) k] = 0.5 * (zi + ci); Q.re[(size_t) k] = 0.5 * (zi - ci); Q.im[(size_t) k] = -0.5 * (zr - cr);
        }
    }
    // p[0 .. len) and q[0 .. len) (either may be missing), put at `at`: their spectra
    void two (const double* p, const double* q, long len, Spec& P, Spec& Q, long at = 0)
    {
        clear();
        if (p != nullptr && len > 0) std::copy (p, p + len, re.begin() + at);
        if (q != nullptr && len > 0) std::copy (q, q + len, im.begin() + at);
        spectra (P, Q);
    }
    // the real signal with the spectrum X: left in re
    void real (const Spec& X)
    {
        const int H = NF / 2;
        for (int k = 0; k <= H; ++k) { const int m = (NF - k) & (NF - 1); re[(size_t) m] = X.re[(size_t) k]; im[(size_t) m] = -X.im[(size_t) k]; re[(size_t) k] = X.re[(size_t) k]; im[(size_t) k] = X.im[(size_t) k]; }
        im[0] = 0.0; im[(size_t) H] = 0.0; T.back (re, im);
    }
};

// ---------------------------------------------------------------- small pieces
// Levinson's recursion on the autocorrelation r[0 .. order], with a little white noise (so the filter stays tame where
// the sound has nothing): w with w[0] = 1; a sound with that autocorrelation comes out flat.
inline Vec levinson (const double* r, int order, double noise)
{
    Vec a ((size_t) order + 1, 0.0); a[0] = 1.0;
    if (! (r[0] > 0.0)) return a;
    Vec rr (r, r + order + 1), t ((size_t) order + 1, 0.0); rr[0] *= 1.0 + noise; double e = rr[0];
    for (int i = 1; i <= order; ++i)
    {
        double acc = rr[(size_t) i]; for (int j = 1; j < i; ++j) acc += a[(size_t) j] * rr[(size_t) (i - j)];
        const double k = -acc / e;
        for (int j = 1; j < i; ++j) t[(size_t) j] = a[(size_t) j] + k * a[(size_t) (i - j)];
        for (int j = 1; j < i; ++j) a[(size_t) j] = t[(size_t) j];
        a[(size_t) i] = k; e *= 1.0 - k * k;
        if (! (e > 0.0)) break;
    }
    return a;
}
// x through the taps w (lags 0 ..), as long as x
inline Vec fir (const Vec& x, const Vec& w)
{
    const long n = (long) x.size(), m = (long) w.size(); Vec out ((size_t) n, 0.0);
    for (long t = 0; t < n; ++t) { double s = 0.0; const long top = std::min (m - 1, t); for (long j = 0; j <= top; ++j) s += w[(size_t) j] * x[(size_t) (t - j)]; out[(size_t) t] = s; }
    return out;
}
// both sides through the taps w, block by block
inline bool firBoth (Work& W, const Vec& w, Vec& l, Vec& r, const Stop& stop)
{
    const long n = (long) l.size(), B = NF - (long) w.size() + 1; Vec wr ((size_t) NF, 0.0), wi ((size_t) NF, 0.0); std::copy (w.begin(), w.end(), wr.begin()); W.T.run (wr, wi);
    Vec ol ((size_t) (n + NF), 0.0), orr ((size_t) (n + NF), 0.0);
    for (long t0 = 0; t0 < n; t0 += B)
    {
        if (stop && stop()) return false;
        const long len = std::min (B, n - t0); W.clear(); std::copy (l.begin() + t0, l.begin() + t0 + len, W.re.begin()); std::copy (r.begin() + t0, r.begin() + t0 + len, W.im.begin()); W.T.run (W.re, W.im);
        for (int k = 0; k < NF; ++k) { const double a = W.re[(size_t) k], b = W.im[(size_t) k]; W.re[(size_t) k] = a * wr[(size_t) k] - b * wi[(size_t) k]; W.im[(size_t) k] = a * wi[(size_t) k] + b * wr[(size_t) k]; }
        W.T.back (W.re, W.im);
        for (int i = 0; i < NF; ++i) { ol[(size_t) (t0 + i)] += W.re[(size_t) i]; orr[(size_t) (t0 + i)] += W.im[(size_t) i]; }
    }
    ol.resize ((size_t) n); orr.resize ((size_t) n); l.swap (ol); r.swap (orr); return true;
}
// rLL, rRR, rLR at lags -K .. K (index K = lag 0), r_XY[k] = sum over t of X(t) Y(t + k); exact, block by block
struct Cor { Vec ll, rr, lr; };
inline bool correlations (Work& W, const Vec& l, const Vec& r, Cor& C, const Stop& stop)
{
    const long n = (long) l.size(); const int H = NF / 2; Spec SLL, SRR, SLR, FL, FR, EL, ER;
    for (Spec* S : { &SLL, &SRR, &SLR }) { S->re.assign ((size_t) H + 1, 0.0); S->im.assign ((size_t) H + 1, 0.0); }
    for (long t0 = 0; t0 < n; t0 += CB)
    {
        if (stop && stop()) return false;
        const long len = std::min ((long) CB, n - t0); W.two (l.data() + t0, r.data() + t0, len, FL, FR);
        const long from = std::max (t0 - K, 0L), to = std::min (t0 + CB + K, n); W.two (l.data() + from, r.data() + from, to - from, EL, ER, from - (t0 - K));      // starts K samples before the block
        for (int k = 0; k <= H; ++k)
        {
            const size_t q = (size_t) k;
            SLL.re[q] += FL.re[q] * EL.re[q] + FL.im[q] * EL.im[q]; SLL.im[q] += FL.re[q] * EL.im[q] - FL.im[q] * EL.re[q];
            SRR.re[q] += FR.re[q] * ER.re[q] + FR.im[q] * ER.im[q]; SRR.im[q] += FR.re[q] * ER.im[q] - FR.im[q] * ER.re[q];
            SLR.re[q] += FL.re[q] * ER.re[q] + FL.im[q] * ER.im[q]; SLR.im[q] += FL.re[q] * ER.im[q] - FL.im[q] * ER.re[q];
        }
    }
    W.real (SLL); C.ll.assign (W.re.begin(), W.re.begin() + 2 * K + 1); W.real (SRR); C.rr.assign (W.re.begin(), W.re.begin() + 2 * K + 1); W.real (SLR); C.lr.assign (W.re.begin(), W.re.begin() + 2 * K + 1);
    return true;
}

// ---------------------------------------------------------------- (A) the pair
// R(t) = sum_j b_j L(t - j) - sum_{j >= 1} a_j R(t - j)  (a_0 = 1), with sums over the whole sound (fast, a little off at
// the file's two ends). Unknown i <= K: b_i; unknown K + j: a_j.
struct Gram
{
    const Vec& rLL; const Vec& rRR; const Vec& rLR; int P = 2 * K + 1; Vec xy, d; double yy = 0.0;
    Gram (const Vec& ll, const Vec& rr, const Vec& lr) : rLL (ll), rRR (rr), rLR (lr), xy ((size_t) (2 * K + 1)), d ((size_t) (2 * K + 1))
    {
        for (int i = 0; i <= K; ++i) { xy[(size_t) i] = rLR[(size_t) (K + i)]; d[(size_t) i] = rLL[(size_t) K]; }
        for (int j = 1; j <= K; ++j) { xy[(size_t) (K + j)] = -rRR[(size_t) (K + j)]; d[(size_t) (K + j)] = rRR[(size_t) K]; }
        yy = rRR[(size_t) K];
    }
    void add (int i, double dw, Vec& grad) const        // grad += dw * (the column of unknown i)
    {
        if (i <= K)
        {
            for (int m = 0; m <= K; ++m) grad[(size_t) m] += dw * rLL[(size_t) (K - i + m)];
            for (int j = 1; j <= K; ++j) grad[(size_t) (K + j)] -= dw * rLR[(size_t) (K + i - j)];
        }
        else
        {
            const int j = i - K;
            for (int m = 0; m <= K; ++m) grad[(size_t) m] -= dw * rLR[(size_t) (K - j + m)];
            for (int l = 1; l <= K; ++l) grad[(size_t) (K + l)] += dw * rRR[(size_t) (K + l - j)];
        }
    }
};
// Coordinate descent on 1/2 |R - X w|^2 + lam |w|_1 through the correlations. The predicted side's own last half
// millisecond is not used: a pair of filters that differ only there is a difference in tone between the sides, not one
// of the effects looked for, and whatever colour the whitening has left in the sound would let a side be explained from
// its own last few samples instead of from the other side.
inline Vec lasso (const Gram& G, double lam, int sweeps = 40, double tol = 1e-5)
{
    const int P = G.P; Vec w ((size_t) P, 0.0), grad ((size_t) P); for (int i = 0; i < P; ++i) grad[(size_t) i] = -G.xy[(size_t) i];
    std::vector<char> fr ((size_t) P, 1); for (int i = K + 1; i < K + NEAR; ++i) fr[(size_t) i] = 0;
    std::vector<int> cand;
    for (int sweep = 0; sweep < sweeps; ++sweep)
    {
        cand.clear(); for (int i = 0; i < P; ++i) if (fr[(size_t) i] && (w[(size_t) i] != 0.0 || std::abs (grad[(size_t) i]) > lam)) cand.push_back (i);
        if (cand.empty()) break;
        std::stable_sort (cand.begin(), cand.end(), [&] (int p, int q) { return std::abs (grad[(size_t) p]) > std::abs (grad[(size_t) q]); });
        if (cand.size() > 300) cand.resize (300);
        double moved = 0.0;
        for (int rep = 0; rep < 6; ++rep)
        {
            double mv = 0.0;
            for (int i : cand)
            {
                const double di = G.d[(size_t) i], z = w[(size_t) i] - grad[(size_t) i] / di, mag = std::max (std::abs (z) - lam / di, 0.0), nw = z > 0.0 ? mag : (z < 0.0 ? -mag : 0.0), dw = nw - w[(size_t) i];
                if (dw != 0.0) { G.add (i, dw, grad); w[(size_t) i] = nw; mv = std::max (mv, std::abs (dw)); }
            }
            moved = std::max (moved, mv);
            if (mv < tol) break;
        }
        if (moved < tol && sweep > 0) break;
    }
    return w;
}
// The regression on the samples K + WORD .. n (every tap inside the file, the whitening filter filled), summed per piece
// of 5 ms: X^T X, X^T y, y^T y. A long sound: PIECES_MOST pieces, evenly spread.
struct Sums { int S = 0, nb = 0; Vec A, b, y; };        // per piece: A (S x S), b (S), y
inline Sums blockSums (const Vec& zL, const Vec& zR, const std::vector<int>& sup)
{
    const long n = (long) zL.size(), t0 = K + WORD, all = std::max (1L, (n - t0) / BLK); const int nb = (int) std::min (all, (long) PIECES_MOST), S = (int) sup.size();
    Sums U; U.S = S; U.nb = nb; U.A.assign ((size_t) nb * S * S, 0.0); U.b.assign ((size_t) nb * S, 0.0); U.y.assign ((size_t) nb, 0.0);
    std::vector<const double*> col ((size_t) S); std::vector<double> sg ((size_t) S);
    for (int q = 0; q < nb; ++q)
    {
        const long k = all <= PIECES_MOST ? q : ((long) q * all) / PIECES_MOST, a = t0 + k * BLK, e = std::min (a + BLK, n), len = e - a;
        if (len <= 0) continue;
        for (int c = 0; c < S; ++c) { const int i = sup[(size_t) c]; if (i <= K) { col[(size_t) c] = zL.data() + a - i; sg[(size_t) c] = 1.0; } else { col[(size_t) c] = zR.data() + a - (i - K); sg[(size_t) c] = -1.0; } }
        const double* y = zR.data() + a; double* A = &U.A[(size_t) q * S * S]; double* b = &U.b[(size_t) q * S];
        for (int c = 0; c < S; ++c)
        {
            const double* pc = col[(size_t) c];
            for (int d = 0; d <= c; ++d) { const double* pd = col[(size_t) d]; double s = 0.0; for (long t = 0; t < len; ++t) s += pc[t] * pd[t]; s *= sg[(size_t) c] * sg[(size_t) d]; A[c * S + d] = s; A[d * S + c] = s; }
            double s = 0.0; for (long t = 0; t < len; ++t) s += pc[t] * y[t]; b[c] = sg[(size_t) c] * s;
        }
        double s = 0.0; for (long t = 0; t < len; ++t) s += y[t] * y[t]; U.y[(size_t) q] = s;
    }
    return U;
}
// Least squares through the Cholesky factor (a little ridge; more if the matrix does not factor): the inverse, the taps, the rest.
inline void solve (const Vec& A, const Vec& b, double yy, int n, Vec& Ai, Vec& w, double& res)
{
    Ai.assign ((size_t) n * n, 0.0); w.assign ((size_t) n, 0.0); res = yy;
    if (n == 0) return;
    double tr = 0.0; for (int i = 0; i < n; ++i) tr += A[(size_t) (i * n + i)];
    double ridge = 1e-10 * std::max (tr, 1e-30) / std::max (n, 1); Vec L ((size_t) n * n, 0.0); bool ok = false;
    for (int tries = 0; tries < 8 && ! ok; ++tries)
    {
        ok = true;
        for (int j = 0; j < n && ok; ++j)
        {
            double s = A[(size_t) (j * n + j)] + ridge; for (int k = 0; k < j; ++k) s -= L[(size_t) (j * n + k)] * L[(size_t) (j * n + k)];
            if (! (s > 0.0) || ! std::isfinite (s)) { ok = false; break; }
            const double dg = std::sqrt (s); L[(size_t) (j * n + j)] = dg;
            for (int i = j + 1; i < n; ++i) { double v = A[(size_t) (i * n + j)]; for (int k = 0; k < j; ++k) v -= L[(size_t) (i * n + k)] * L[(size_t) (j * n + k)]; L[(size_t) (i * n + j)] = v / dg; }
        }
        if (! ok) ridge *= 100.0;
    }
    if (! ok) return;
    Vec Li ((size_t) n * n, 0.0);
    for (int j = 0; j < n; ++j)
    {
        Li[(size_t) (j * n + j)] = 1.0 / L[(size_t) (j * n + j)];
        for (int i = j + 1; i < n; ++i) { double v = 0.0; for (int k = j; k < i; ++k) v -= L[(size_t) (i * n + k)] * Li[(size_t) (k * n + j)]; Li[(size_t) (i * n + j)] = v / L[(size_t) (i * n + i)]; }
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) { double s = 0.0; for (int k = i; k < n; ++k) s += Li[(size_t) (k * n + i)] * Li[(size_t) (k * n + j)]; Ai[(size_t) (i * n + j)] = s; Ai[(size_t) (j * n + i)] = s; }
    double dot = 0.0;
    for (int i = 0; i < n; ++i) { double s = 0.0; for (int j = 0; j < n; ++j) s += Ai[(size_t) (i * n + j)] * b[(size_t) j]; w[(size_t) i] = s; }
    for (int i = 0; i < n; ++i) dot += b[(size_t) i] * w[(size_t) i];
    res = yy - dot;
}
// Backward elimination: the tap whose loss costs least goes while the unexplained rest stays under (1 + grow) * best + slack.
inline std::vector<int> prune (const Vec& A, const Vec& b, double yy, int n, double slack = 1e-4, double grow = 0.10)
{
    std::vector<int> idx ((size_t) n); std::iota (idx.begin(), idx.end(), 0);
    if (n == 0) return idx;
    Vec Ai, w, As, bs; double res = 0.0; solve (A, b, yy, n, Ai, w, res); const double limit = (1.0 + grow) * std::max (res, 0.0) + slack * yy;
    while (! idx.empty())
    {
        const int m = (int) idx.size(); int j = 0; double best = 0.0;
        for (int i = 0; i < m; ++i) { const double c = w[(size_t) i] * w[(size_t) i] / std::max (Ai[(size_t) (i * m + i)], 1e-30); if (i == 0 || c < best) { best = c; j = i; } }
        if (res + best > limit) break;
        idx.erase (idx.begin() + j);
        if (idx.empty()) break;
        const int m2 = (int) idx.size(); As.assign ((size_t) m2 * m2, 0.0); bs.assign ((size_t) m2, 0.0);
        for (int p = 0; p < m2; ++p) { bs[(size_t) p] = b[(size_t) idx[(size_t) p]]; for (int q = 0; q < m2; ++q) As[(size_t) (p * m2 + q)] = A[(size_t) (idx[(size_t) p] * n + idx[(size_t) q])]; }
        solve (As, bs, yy, m2, Ai, w, res);
    }
    return idx;
}
// Trimmed fit: the pieces that fit worst are left out, 5 % at a time (at most half). The trimmed fit is taken only if it
// explains the pieces still in at least `need` times better than the fit over everything explained them all: then most
// of the sound obeys the pair exactly and the rest is something else (a click, a glitch of the plugin). In that case
// every piece that fits about as well as the middle one is in, whatever was left out on the way.
struct Fit { std::vector<char> keep; Vec w; double res = 1.0; };
inline Fit robust (const Sums& U, const std::vector<int>& idx, const std::vector<char>& live, int rounds = 10, double drop = 0.05, double need = 100.0)
{
    const int m = (int) idx.size(), nb = U.nb, S = U.S; Vec As ((size_t) nb * m * m), bs ((size_t) nb * m); double ymax = 0.0; int nlive = 0;
    for (int q = 0; q < nb; ++q)
    {
        ymax = std::max (ymax, U.y[(size_t) q]); nlive += live[(size_t) q] ? 1 : 0;
        for (int i = 0; i < m; ++i) { bs[(size_t) q * m + i] = U.b[(size_t) q * S + idx[(size_t) i]]; for (int j = 0; j < m; ++j) As[((size_t) q * m + i) * m + j] = U.A[((size_t) q * S + idx[(size_t) i]) * S + idx[(size_t) j]]; }
    }
    Vec A, b, Ai;
    auto fit = [&] (const std::vector<char>& k, Vec& w) -> double
    {
        A.assign ((size_t) m * m, 0.0); b.assign ((size_t) m, 0.0); double yy = 0.0;
        for (int q = 0; q < nb; ++q)
        {
            if (! k[(size_t) q]) continue;
            yy += U.y[(size_t) q]; for (int i = 0; i < m; ++i) b[(size_t) i] += bs[(size_t) q * m + i];
            const double* a = &As[(size_t) q * m * m]; for (int i = 0; i < m * m; ++i) A[(size_t) i] += a[i];
        }
        double res = 0.0; solve (A, b, yy, m, Ai, w, res); return std::max (res, 0.0) / std::max (yy, 1e-30);
    };
    auto errors = [&] (const Vec& w, Vec& rel)
    {
        for (int q = 0; q < nb; ++q)
        {
            double lin = 0.0, quad = 0.0; const double* a = m > 0 ? &As[(size_t) q * m * m] : nullptr;
            for (int i = 0; i < m; ++i) { lin += bs[(size_t) q * m + i] * w[(size_t) i]; double s = 0.0; for (int j = 0; j < m; ++j) s += a[i * m + j] * w[(size_t) j]; quad += w[(size_t) i] * s; }
            rel[(size_t) q] = (U.y[(size_t) q] - 2.0 * lin + quad) / std::max (U.y[(size_t) q], 1e-9 * ymax);
        }
    };
    auto any = [] (const std::vector<char>& k) { for (char c : k) if (c) return true; return false; };
    Fit F; Vec w0; const double res0 = fit (live, w0); std::vector<char> k = live, k2; Vec w = w0, rel ((size_t) nb, 0.0); double res = res0; std::vector<int> ord ((size_t) nb);
    for (int r = 0; r < rounds; ++r)
    {
        errors (w, rel); for (int q = 0; q < nb; ++q) if (! k[(size_t) q]) rel[(size_t) q] = -1.0;
        const int nd = std::max (1, (int) (drop * nlive)); std::iota (ord.begin(), ord.end(), 0);
        std::stable_sort (ord.begin(), ord.end(), [&] (int p, int q) { return rel[(size_t) p] > rel[(size_t) q]; });
        k2 = k; for (int i = 0; i < nd && i < nb; ++i) k2[(size_t) ord[(size_t) i]] = 0;
        if (! any (k2)) break;
        k = k2; res = fit (k, w);
        if (res < 1e-9) break;
    }
    if (! (res * need < res0)) { F.keep = live; F.w = w0; F.res = res0; return F; }
    Vec in;
    for (int r = 0; r < 2; ++r)
    {
        errors (w, rel); in.clear(); for (int q = 0; q < nb; ++q) if (k[(size_t) q]) in.push_back (rel[(size_t) q]);
        std::sort (in.begin(), in.end()); const size_t c = in.size(); const double med = c == 0 ? 0.0 : (c % 2 ? in[c / 2] : 0.5 * (in[c / 2 - 1] + in[c / 2])), top = std::max (100.0 * med, 1e-4);
        k2.assign ((size_t) nb, 0); for (int q = 0; q < nb; ++q) k2[(size_t) q] = (live[(size_t) q] && rel[(size_t) q] <= top) ? 1 : 0;
        if (! any (k2)) break;
        k = k2; res = fit (k, w);
    }
    F.keep = k; F.w = w; F.res = res; return F;
}
// One way round: the side zR told from the side zL and from its own past. a (own past, a[0] = 1) and b (the other side): K + 1 taps each.
struct Way { bool ok = false; Vec a, b; double res = 1.0, fits = 0.0; int taps = 0; };
inline Way oneWay (const Vec& zL, const Vec& zR, const Vec& rLL, const Vec& rRR, const Vec& rLR, double mu = 0.01)
{
    Way R; const Gram G (rLL, rRR, rLR); double top = 0.0; for (double v : G.xy) top = std::max (top, std::abs (v));
    const Vec w = lasso (G, mu * top); std::vector<int> sup; for (int i = 0; i < G.P; ++i) if (w[(size_t) i] != 0.0) sup.push_back (i);
    if ((int) sup.size() > MOST / 2) { std::stable_sort (sup.begin(), sup.end(), [&] (int p, int q) { return std::abs (w[(size_t) p]) > std::abs (w[(size_t) q]); }); sup.resize ((size_t) (MOST / 2)); }
    std::set<int> near (sup.begin(), sup.end()); near.insert (0);       // the exact fit may also use the taps next to the found ones and the direct tap of the predicting side
    for (int i : sup) for (int d : { -1, 1 }) { const int j = i + d; if ((i <= K && 0 <= j && j <= K) || (i > K && K + NEAR <= j && j <= 2 * K)) near.insert (j); }
    std::vector<int> s2 (near.begin(), near.end());
    if ((int) s2.size() > MOST)
    {
        std::sort (s2.begin(), s2.end(), [&] (int p, int q) { const double ap = std::abs (w[(size_t) p]), aq = std::abs (w[(size_t) q]); return ap != aq ? ap > aq : p < q; });
        s2.resize ((size_t) MOST); std::sort (s2.begin(), s2.end());
    }
    const Sums U = blockSums (zL, zR, s2); double ymax = 0.0, ysum = 0.0; for (double v : U.y) { ymax = std::max (ymax, v); ysum += v; }
    std::vector<char> live ((size_t) U.nb, 0); bool some = false; for (int q = 0; q < U.nb; ++q) { live[(size_t) q] = U.y[(size_t) q] > 1e-9 * ymax ? 1 : 0; some = some || live[(size_t) q]; }
    if (! some) return R;
    std::vector<int> idx (s2.size()); std::iota (idx.begin(), idx.end(), 0); Fit F = robust (U, idx, live); Vec A, b;
    for (int rep = 0; rep < 2; ++rep)                   // taps that explain next to nothing go; then the pieces are chosen again with the taps left
    {
        const int m = (int) idx.size(); A.assign ((size_t) m * m, 0.0); b.assign ((size_t) m, 0.0); double yy = 0.0;
        for (int q = 0; q < U.nb; ++q)
        {
            if (! F.keep[(size_t) q]) continue;
            yy += U.y[(size_t) q];
            for (int i = 0; i < m; ++i) { b[(size_t) i] += U.b[(size_t) q * U.S + idx[(size_t) i]]; for (int j = 0; j < m; ++j) A[(size_t) (i * m + j)] += U.A[((size_t) q * U.S + idx[(size_t) i]) * U.S + idx[(size_t) j]]; }
        }
        const std::vector<int> sub = prune (A, b, yy, m); std::vector<int> nx; for (int p : sub) nx.push_back (idx[(size_t) p]); idx.swap (nx);
        F = robust (U, idx, live);
    }
    R.a.assign ((size_t) K + 1, 0.0); R.b.assign ((size_t) K + 1, 0.0); R.a[0] = 1.0;
    for (size_t p = 0; p < idx.size(); ++p) { const int i = s2[(size_t) idx[p]]; if (i <= K) R.b[(size_t) i] = F.w[p]; else R.a[(size_t) (i - K)] = F.w[p]; }
    double kept = 0.0; for (int q = 0; q < U.nb; ++q) if (F.keep[(size_t) q]) kept += U.y[(size_t) q];
    R.ok = true; R.res = F.res; R.taps = (int) idx.size(); R.fits = kept / std::max (ysum, 1e-30); return R;
}

// ---------------------------------------------------------------- block work: one filter pair on the whole sound
// p through H0 plus q through H1 (transfer functions on NF / 2 + 1 points, any phase, responses well under a second), by
// overlapping windows: exact up to the responses' far ends. q may be missing.
inline bool filtered (Work& W, const double* p, const double* q, long n, const Spec& H0, const Spec& H1, Vec& out, const Stop& stop, int seg = 1 << 15)
{
    const int hop = seg / 2, off = (NF - seg) / 2, H = NF / 2; Vec win ((size_t) seg); for (int i = 0; i < seg; ++i) win[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * PI * i / seg);
    Vec acc ((size_t) (n + NF), 0.0); Spec P, Q, X; X.re.resize ((size_t) H + 1); X.im.resize ((size_t) H + 1);
    for (long t0 = -hop; t0 < n; t0 += hop)
    {
        const long a = std::max (t0, 0L), e = std::min (t0 + seg, n);
        if (e <= a) continue;
        if (stop && stop()) return false;
        W.clear();
        for (long i = a; i < e; ++i) W.re[(size_t) (off + i - t0)] = p[i] * win[(size_t) (i - t0)];
        if (q != nullptr) for (long i = a; i < e; ++i) W.im[(size_t) (off + i - t0)] = q[i] * win[(size_t) (i - t0)];
        W.spectra (P, Q);
        for (int k = 0; k <= H; ++k)
        {
            const size_t c = (size_t) k; double xr = P.re[c] * H0.re[c] - P.im[c] * H0.im[c], xi = P.re[c] * H0.im[c] + P.im[c] * H0.re[c];
            if (q != nullptr) { xr += Q.re[c] * H1.re[c] - Q.im[c] * H1.im[c]; xi += Q.re[c] * H1.im[c] + Q.im[c] * H1.re[c]; }
            X.re[c] = xr; X.im[c] = xi;
        }
        W.real (X); const long at = t0 - off, i0 = std::max (0L, -at), i1 = std::min ((long) NF, n - at);      // the result's sample i belongs to time at + i
        for (long i = i0; i < i1; ++i) acc[(size_t) (at + i)] += W.re[(size_t) i];
    }
    acc.resize ((size_t) n); out.swap (acc); return true;
}
// x through the taps v
inline Vec sparseConv (const Vec& x, const Taps& v)
{
    const long n = (long) x.size(); Vec out ((size_t) n, 0.0);
    for (const Tap& t : v) { if (t.v == 0.0) continue; for (long i = t.lag; i < n; ++i) out[(size_t) i] += t.v * x[(size_t) (i - t.lag)]; }
    return out;
}
inline Vec sparseConv (const std::vector<float>& x, const Taps& v)
{
    const long n = (long) x.size(); Vec out ((size_t) n, 0.0);
    for (const Tap& t : v) { if (t.v == 0.0) continue; for (long i = t.lag; i < n; ++i) out[(size_t) i] += t.v * (double) x[(size_t) (i - t.lag)]; }
    return out;
}
inline Taps tapsOf (const Vec& v) { Taps t; for (size_t i = 0; i < v.size(); ++i) if (v[i] != 0.0) t.push_back ({ (int) i, v[i] }); return t; }
// The one sound behind both sides: X = (A* L + B* R) / (|A|^2 + |B|^2). With left = a * x, right = b * x it is x itself.
inline bool theOneSound (Work& W, const Stereo& y, const Taps& a, const Taps& b, Vec& x, const Stop& stop, double eps = 1e-4)
{
    Spec A, B, H0, H1; W.clear(); for (const Tap& t : a) W.re[(size_t) t.lag] = t.v; for (const Tap& t : b) W.im[(size_t) t.lag] = t.v; W.spectra (A, B);
    const int H = NF / 2; Vec D ((size_t) H + 1); double top = 0.0;
    for (int k = 0; k <= H; ++k) { const size_t c = (size_t) k; D[c] = A.re[c] * A.re[c] + A.im[c] * A.im[c] + B.re[c] * B.re[c] + B.im[c] * B.im[c]; top = std::max (top, D[c]); }
    H0.re.resize ((size_t) H + 1); H0.im.resize ((size_t) H + 1); H1.re.resize ((size_t) H + 1); H1.im.resize ((size_t) H + 1);
    for (int k = 0; k <= H; ++k) { const size_t c = (size_t) k; const double g = 1.0 / (D[c] + eps * top); H0.re[c] = A.re[c] * g; H0.im[c] = -A.im[c] * g; H1.re[c] = B.re[c] * g; H1.im[c] = -B.im[c] * g; }
    return filtered (W, y.l.data(), y.r.data(), (long) y.size(), H0, H1, x, stop);
}
// Per piece of 5 ms: p its power, u what is unexplained of it. A piece obeys when u <= thr * p. Spans: runs of such
// pieces, holes of up to `close` pieces (quiet or not obeying) closed, runs under `least` pieces dropped.
struct Pieces { std::vector<char> ok; Spans spans; double covered = 0.0, inside = 1.0; };
inline Pieces pieces (const Vec& p, const Vec& u, double thr, int close = 50, int least = 4)
{
    const int nb = (int) p.size(); Pieces R; R.ok.assign ((size_t) nb, 0); double top = 0.0, all = 0.0, pin = 0.0, uin = 0.0; for (double v : p) { top = std::max (top, v); all += v; }
    bool some = false;
    for (int i = 0; i < nb; ++i) { const bool o = top > 0.0 && p[(size_t) i] > 1e-7 * top && u[(size_t) i] <= thr * p[(size_t) i]; R.ok[(size_t) i] = o ? 1 : 0; if (o) { pin += p[(size_t) i]; uin += u[(size_t) i]; some = true; } }
    R.covered = pin / std::max (all, 1e-30); R.inside = some ? uin / std::max (pin, 1e-30) : 1.0;
    int i = 0;
    while (i < nb)
    {
        if (! R.ok[(size_t) i]) { ++i; continue; }
        int j = i;
        while (true)
        {
            while (j < nb && R.ok[(size_t) j]) ++j;
            int g = j;
            while (g < nb && ! R.ok[(size_t) g] && g - j <= close) ++g;
            if (g < nb && R.ok[(size_t) g] && g - j <= close) j = g; else break;
        }
        if (j - i >= least) R.spans.push_back ({ i, j });
        i = j;
    }
    return R;
}
// What the pair explains of the sound itself: x, and per piece how much is unexplained. beyond: what is unexplained in
// the obeying pieces against what one sound at two levels (no filters at all: a sound panned somewhere) leaves
// unexplained there, dB. A sound with little at its sides is explained well by any pair close to that; a pair that is
// really there explains far more than the panned sound does.
struct Check { Vec x; Pieces pc; double beyond = 0.0; };
inline bool checkPair (Work& W, const Stereo& y, const Taps& a, const Taps& b, Check& C, const Stop& stop, double thr = 0.06309573444801933)
{
    if (! theOneSound (W, y, a, b, C.x, stop)) return false;
    const long n = (long) y.size(); const int nb = (int) (n / BLK); C.beyond = 0.0;
    if (nb == 0) { C.pc = Pieces(); return true; }
    Vec p ((size_t) nb, 0.0), u ((size_t) nb, 0.0); const Vec& x = C.x;
    for (int q = 0; q < nb; ++q)
    {
        double pp = 0.0, uu = 0.0;
        for (long t = (long) q * BLK; t < (long) (q + 1) * BLK; ++t)
        {
            double cl = 0.0, cr = 0.0; for (const Tap& tp : a) if (t >= tp.lag) cl += tp.v * x[(size_t) (t - tp.lag)]; for (const Tap& tp : b) if (t >= tp.lag) cr += tp.v * x[(size_t) (t - tp.lag)];
            const double l = y.l[(size_t) t], r = y.r[(size_t) t], rl = l - cl, rr = r - cr; pp += l * l + r * r; uu += rl * rl + rr * rr;
        }
        p[(size_t) q] = pp; u[(size_t) q] = uu;
    }
    C.pc = pieces (p, u, thr); double ll = 0.0, rr = 0.0, lr = 0.0, uin = 0.0; bool some = false;
    for (int q = 0; q < nb; ++q)
    {
        if (! C.pc.ok[(size_t) q]) continue;
        some = true; uin += u[(size_t) q];
        for (long t = (long) q * BLK; t < (long) (q + 1) * BLK; ++t) { const double l = y.l[(size_t) t], r = y.r[(size_t) t]; ll += l * l; rr += r * r; lr += l * r; }
    }
    if (some)
    {
        const double plain = 0.5 * (ll + rr) - std::sqrt ((0.5 * (ll - rr)) * (0.5 * (ll - rr)) + lr * lr);      // the smaller eigenvalue: what the best panned sound leaves
        C.beyond = plain > 1e-12 * (ll + rr) ? 10.0 * std::log10 (std::max (uin, 1e-30) / std::max (plain, 1e-30)) : 0.0;
    }
    return true;
}
// Numbers of the pair: how many taps, how far apart the two sides' main taps are (samples), how far its taps reach, how
// much side it makes of white noise with both sides brought to one level (dB under the middle).
struct Shape { int taps = 0, apart = 0, reach = 0; double made = -120.0; };
inline Shape shape (const Taps& a, const Taps& b)
{
    Shape S; double ma = 0.0, mb = 0.0, ea = 0.0, eb = 0.0; int ia = 0, ib = 0;
    for (const Tap& t : a) { if (std::abs (t.v) > ma) { ma = std::abs (t.v); ia = t.lag; } ea += t.v * t.v; }
    for (const Tap& t : b) { if (std::abs (t.v) > mb) { mb = std::abs (t.v); ib = t.lag; } eb += t.v * t.v; }
    int lo = 1 << 30, hi = -1; for (const Tap& t : a) if (std::abs (t.v) >= 0.1 * ma) { lo = std::min (lo, t.lag); hi = std::max (hi, t.lag); }
    for (const Tap& t : b) if (std::abs (t.v) >= 0.1 * mb) { lo = std::min (lo, t.lag); hi = std::max (hi, t.lag); }
    const double na = std::sqrt (ea), nb = std::sqrt (eb); double side = 0.0, mid = 0.0; size_t i = 0, j = 0;
    while (i < a.size() || j < b.size())                // over the lags either side has
    {
        const int la = i < a.size() ? a[i].lag : 1 << 30, lb = j < b.size() ? b[j].lag : 1 << 30, lag = std::min (la, lb);
        const double va = la == lag ? a[i++].v / na : 0.0, vb = lb == lag ? b[j++].v / nb : 0.0; side += (va - vb) * (va - vb); mid += (va + vb) * (va + vb);
    }
    S.taps = (int) (a.size() + b.size()); S.apart = std::abs (ia - ib); S.reach = hi >= lo ? hi - lo : 0; S.made = 10.0 * std::log10 (std::max (side, 1e-12) / std::max (mid, 1e-12)); return S;
}
// The stretch of `most` samples with the most energy (looked for in steps of half a second): start, length.
inline std::pair<long, long> loudest (const Stereo& y, long most = LONG, long hop = 24000)
{
    const long n = (long) y.size(); if (n <= most) return { 0, n };
    const long nb = n / hop, k = most / hop; Vec c ((size_t) nb + 1, 0.0);
    for (long b = 0; b < nb; ++b) { double e = 0.0; for (long i = b * hop; i < (b + 1) * hop; ++i) e += y.l[(size_t) i] * y.l[(size_t) i] + y.r[(size_t) i] * y.r[(size_t) i]; c[(size_t) b + 1] = c[(size_t) b] + e; }
    long best = 0; double top = 0.0; for (long s = 0; s + k <= nb; ++s) { const double v = c[(size_t) (s + k)] - c[(size_t) s]; if (s == 0 || v > top) { top = v; best = s; } }
    return { best * hop, most };
}
// (A): the two filters (the largest tap of all = 1), fitted both ways round; the way that explains more of the sound
// itself is taken. False: no pair (or given up).
struct Pair { Taps a, b; int way = 0, taps = 0; double res = 1.0, fits = 0.0; Check check; };
inline bool pairOf (Work& W, const Stereo& whole, Pair& best, const Stop& stop, double mu = 0.01)
{
    const auto part = loudest (whole); const bool cut = part.second < (long) whole.size(); Stereo held; if (cut) held = whole.slice ((size_t) part.first, (size_t) part.second);
    const Stereo& y = cut ? held : whole; const long n = (long) y.size(); double eL = 0.0, eR = 0.0;
    for (long i = 0; i < n; ++i) { eL += y.l[(size_t) i] * y.l[(size_t) i]; eR += y.r[(size_t) i] * y.r[(size_t) i]; }
    if (n < LEAST || std::min (eL, eR) < 1e-8 * std::max ({ eL, eR, 1e-30 })) return false;
    const double sl = std::sqrt (std::max (eL, 1e-30)), sr = std::sqrt (std::max (eR, 1e-30)); Vec zl ((size_t) n), zr ((size_t) n);
    for (long i = 0; i < n; ++i) { zl[(size_t) i] = y.l[(size_t) i] / sl; zr[(size_t) i] = y.r[(size_t) i] / sr; }
    Cor C; if (! correlations (W, zl, zr, C, stop)) return false;
    Vec r ((size_t) WORD + 1); for (int k = 0; k <= WORD; ++k) r[(size_t) k] = C.ll[(size_t) (K + k)] + C.rr[(size_t) (K + k)];
    const Vec w = levinson (r.data(), WORD, 1e-4);     // both sides through the same long filter, the one that flattens their average spectrum over 40 ms
    if (! firBoth (W, w, zl, zr, stop) || ! correlations (W, zl, zr, C, stop)) return false;
    Vec back (C.lr.rbegin(), C.lr.rend()); bool have = false; double bestCost = 0.0;
    for (int way = 0; way < 2; ++way)
    {
        if (stop && stop()) return false;
        const Way R = way == 0 ? oneWay (zl, zr, C.ll, C.rr, C.lr, mu) : oneWay (zr, zl, C.rr, C.ll, back, mu);
        if (! R.ok) continue;
        Vec a = way == 0 ? R.a : R.b, b = way == 0 ? R.b : R.a; double k = 0.0, ka = 0.0, kb = 0.0;
        for (double& v : a) { v *= sl; ka = std::max (ka, std::abs (v)); } for (double& v : b) { v *= sr; kb = std::max (kb, std::abs (v)); }
        if (! (ka > 0.0 && kb > 0.0)) continue;        // one side told from its own past alone: no pair
        k = std::max (ka, kb); for (double& v : a) v /= k; for (double& v : b) v /= k;
        Pair P; P.a = tapsOf (a); P.b = tapsOf (b); P.way = way; P.taps = R.taps; P.res = R.res; P.fits = R.fits;
        if (! checkPair (W, y, P.a, P.b, P.check, stop)) return false;
        const double cost = (1.0 - P.check.pc.covered) + P.check.pc.inside + 0.002 * (double) (P.a.size() + P.b.size());
        if (! have || cost < bestCost) { best = std::move (P); bestCost = cost; have = true; }
    }
    if (have && cut && ! checkPair (W, whole, best.a, best.b, best.check, stop)) return false;
    return have;
}

// ---------------------------------------------------------------- (B) the side as a filtered copy of the middle
// Solve T h = rhs for the symmetric Toeplitz matrix T with first column col (Levinson's recursion).
inline Vec levinsonRhs (const Vec& col, const Vec& rhs)
{
    const int n = (int) rhs.size(); Vec f ((size_t) n, 0.0), x ((size_t) n, 0.0), fn ((size_t) n, 0.0); f[0] = 1.0 / col[0]; x[0] = rhs[0] / col[0];
    for (int i = 1; i < n; ++i)
    {
        double ef = 0.0; for (int m = 0; m < i; ++m) ef += col[(size_t) (i - m)] * f[(size_t) m];
        const double den = 1.0 - ef * ef;
        if (! (std::abs (den) > 1e-300)) break;
        for (int m = 0; m <= i; ++m) fn[(size_t) m] = 0.0;
        for (int m = 0; m < i; ++m) fn[(size_t) m] = f[(size_t) m] / den;
        for (int m = 0; m < i; ++m) fn[(size_t) (1 + m)] -= ef / den * f[(size_t) (i - 1 - m)];      // the forward vector of size i + 1; the backward one is its mirror
        for (int m = 0; m <= i; ++m) f[(size_t) m] = fn[(size_t) m];
        double ex = 0.0; for (int m = 0; m < i; ++m) ex += col[(size_t) (i - m)] * x[(size_t) m];
        const double g = rhs[(size_t) i] - ex; for (int m = 0; m <= i; ++m) x[(size_t) m] += g * fn[(size_t) (i - m)];
    }
    return x;
}
// h * m with h's index KB at no delay, as long as m
inline bool convCentered (Work& W, const Vec& m, const Vec& h, Vec& out, const Stop& stop)
{
    Spec H, none; W.clear(); for (int i = 0; i <= KB; ++i) W.re[(size_t) i] = h[(size_t) (KB + i)]; for (int i = 0; i < KB; ++i) W.re[(size_t) (NF - KB + i)] = h[(size_t) i];
    W.spectra (H, none); return filtered (W, m.data(), nullptr, (long) m.size(), H, none, out, stop);
}
// h (2 KB + 1 taps, index KB = no delay) with side ~ h * middle, fitted on whitened signals over the whole sound.
// share: the part of the side's power that is such a copy, without what no delay alone explains (that is balance).
struct Copy { bool ok = false; Vec h; double share = 0.0, all = 0.0, balance = 0.0, sideDb = 0.0; };
inline bool sideCopy (Work& W, const Stereo& y, Copy& G, const Stop& stop)
{
    const long n = (long) y.size(); Vec m ((size_t) n), s ((size_t) n); double em = 0.0, es = 0.0; G = Copy();
    for (long i = 0; i < n; ++i) { m[(size_t) i] = 0.5 * (y.l[(size_t) i] + y.r[(size_t) i]); s[(size_t) i] = 0.5 * (y.l[(size_t) i] - y.r[(size_t) i]); em += m[(size_t) i] * m[(size_t) i]; es += s[(size_t) i] * s[(size_t) i]; }
    if (n < LEAST || em < 1e-12 || es < 1e-10 * em) return true;
    Vec r ((size_t) ORDER + 1, 0.0); for (int k = 0; k <= ORDER; ++k) { double a = 0.0; for (long t = 0; t + k < n; ++t) a += m[(size_t) t] * m[(size_t) (t + k)]; r[(size_t) k] = a; }
    const Vec w = levinson (r.data(), ORDER, 1e-6); const Vec zm = fir (m, w), zs = fir (s, w); Cor C; if (! correlations (W, zm, zs, C, stop)) return false;
    const int P = 2 * KB + 1; Vec col ((size_t) P), rhs ((size_t) P); for (int i = 0; i < P; ++i) { col[(size_t) i] = C.ll[(size_t) (K + i)]; rhs[(size_t) i] = C.lr[(size_t) (K - KB + i)]; }
    col[0] *= 1.0 + 1e-4; G.h = levinsonRhs (col, rhs); const double yy = C.rr[(size_t) K]; double dot = 0.0; for (int i = 0; i < P; ++i) dot += rhs[(size_t) i] * G.h[(size_t) i];
    const double expl = dot / std::max (yy, 1e-30), expl0 = rhs[(size_t) KB] * rhs[(size_t) KB] / col[0] / std::max (yy, 1e-30);
    G.ok = true; G.share = std::max (0.0, expl - expl0); G.all = expl; G.balance = G.h[(size_t) KB]; G.sideDb = 10.0 * std::log10 (es / em); return true;
}
struct CopyCheck { Vec copy; Pieces pc; };
inline bool checkCopy (Work& W, const Stereo& y, const Vec& h, CopyCheck& C, const Stop& stop, double thr = 0.1)
{
    const long n = (long) y.size(); Vec m ((size_t) n), s ((size_t) n); for (long i = 0; i < n; ++i) { m[(size_t) i] = 0.5 * (y.l[(size_t) i] + y.r[(size_t) i]); s[(size_t) i] = 0.5 * (y.l[(size_t) i] - y.r[(size_t) i]); }
    if (! convCentered (W, m, h, C.copy, stop)) return false;
    const int nb = (int) (n / BLK); if (nb == 0) { C.pc = Pieces(); return true; }
    Vec p ((size_t) nb, 0.0), u ((size_t) nb, 0.0);
    for (int q = 0; q < nb; ++q) for (long t = (long) q * BLK; t < (long) (q + 1) * BLK; ++t) { const double v = s[(size_t) t], d = v - C.copy[(size_t) t]; p[(size_t) q] += v * v; u[(size_t) q] += d * d; }
    C.pc = pieces (p, u, thr); return true;
}

// ---------------------------------------------------------------- the verdict
inline Found find (const Stereo& y, const Stop& stop = {}, const Rule& R = {})
{
    Found F; F.n = y.size(); Work W; Pair P;
    if (y.size() == 0) return F;
    if (pairOf (W, y, P, stop))
    {
        const Shape sh = shape (P.a, P.b); F.hasPair = true; F.way = P.way; F.taps = sh.taps; F.apart = sh.apart; F.reach = sh.reach; F.made = sh.made; F.res = P.res; F.fits = P.fits;
        F.covered = P.check.pc.covered; F.inside = P.check.pc.inside; F.beyond = P.check.beyond; F.a = P.a; F.b = P.b;
        if (sh.taps <= R.taps && sh.reach >= R.reach && sh.made >= R.made && F.covered >= R.covered && 10.0 * std::log10 (std::max (F.inside, 1e-30)) <= R.inside && F.beyond <= R.beyond)
        {
            F.mode = PAIR; F.kind = sh.apart >= APART ? HAAS : WIDTH; F.spans = P.check.pc.spans; F.x.assign (P.check.x.begin(), P.check.x.end()); return F;
        }
    }
    if (stop && stop()) return Found();
    Copy G; if (! sideCopy (W, y, G, stop)) return Found();
    if (G.ok)
    {
        F.hasCopy = true; F.share = G.share; F.all = G.all; F.balance = G.balance; F.sideDb = G.sideDb; F.h = G.h;
        if (G.share >= R.share && G.sideDb >= R.side)
        {
            CopyCheck C; if (! checkCopy (W, y, G.h, C, stop)) return Found();
            F.copyChecked = true; F.copyCovered = C.pc.covered; F.copyInside = C.pc.inside;
            if (C.pc.covered >= R.copyCovered) { F.mode = COPY; F.kind = WIDTH; F.spans = C.pc.spans; F.x.assign (C.copy.begin(), C.copy.end()); }
        }
    }
    return F;
}
// The stretches of the sound that have the effect, in seconds.
inline std::vector<std::pair<double, double>> where (const Found& F)
{
    std::vector<std::pair<double, double>> out;
    for (const auto& s : F.spans) out.push_back ({ s.first * BLK / SR, ((size_t) (s.second + 1) * BLK > F.n ? (double) F.n : (double) s.second * BLK) / SR });
    return out;
}

// ---------------------------------------------------------------- the Amount fader
// 1 inside the spans, 0 outside; the way in and out (10 ms, raised cosine) lies outside the span, where the sound is
// quiet or something else: a span's first and last moments are worked on in full.
inline Vec gate (const Spans& spans, long n, int fade = 480)
{
    Vec g ((size_t) n, 0.0), ramp ((size_t) fade); for (int i = 0; i < fade; ++i) ramp[(size_t) i] = 0.5 - 0.5 * std::cos (PI * (i + 0.5) / fade);
    for (const auto& s : spans)
    {
        const long a = std::min ((long) s.first * BLK, n), e = (long) (s.second + 1) * BLK > n ? n : (long) s.second * BLK;
        for (long i = a; i < e; ++i) g[(size_t) i] = 1.0;
        long k = std::min ((long) fade, a); for (long i = 0; i < k; ++i) g[(size_t) (a - k + i)] = std::max (g[(size_t) (a - k + i)], ramp[(size_t) (fade - k + i)]);
        k = std::min ((long) fade, n - e); for (long i = 0; i < k; ++i) g[(size_t) (e + i)] = std::max (g[(size_t) (e + i)], ramp[(size_t) (fade - 1 - i)]);
    }
    return g;
}
// x moved by d samples (later for d > 0, earlier for d < 0; any fraction): a whole shift and a windowed-sinc filter for the rest.
inline Vec shift (const Vec& x, double d, int half = 32)
{
    const long n = (long) x.size(); long k = (long) std::floor (d); double fr = d - (double) k;
    if (fr > 1.0 - 1e-9) { k += 1; fr = 0.0; }
    Vec out ((size_t) n, 0.0);
    if (fr < 1e-9) { for (long i = std::max (0L, k); i < std::min (n, n + k); ++i) out[(size_t) i] = x[(size_t) (i - k)]; return out; }
    Vec h ((size_t) (2 * half)); double sum = 0.0;
    for (int j = 0; j < 2 * half; ++j) { const double t = (double) (j - half + 1) - fr, pt = PI * t, sc = std::abs (pt) < 1e-300 ? 1.0 : std::sin (pt) / pt; h[(size_t) j] = sc * (0.5 + 0.5 * std::cos (PI * t / half)); sum += h[(size_t) j]; }
    for (double& v : h) v /= sum;
    for (int j = 0; j < 2 * half; ++j)                  // out[i] = sum over j of h[j] x[i - k - q], q = j - half + 1
    {
        const long sh = k + (j - half + 1); const double hj = h[(size_t) j];
        for (long i = std::max (0L, sh); i < std::min (n, n + sh); ++i) out[(size_t) i] += hj * x[(size_t) (i - sh)];
    }
    return out;
}
// (delay, gain) if the filter is a single tap, or two neighbours that stand for a tap between two samples.
inline bool oneTap (const Taps& v, double& delay, double& gain)
{
    if (v.size() == 1) { delay = v[0].lag; gain = v[0].v; return true; }
    if (v.size() == 2 && v[1].lag == v[0].lag + 1 && v[0].v * v[1].v > 0.0) { gain = v[0].v + v[1].v; delay = (v[0].lag * v[0].v + v[1].lag * v[1].v) / gain; return true; }
    return false;
}
// The pair with Amount at s (0 .. 2; 1 = as found) against the plain sound at level c on both sides (s = 0).
// One tap on each side, both the same way up (warp): the late side comes nearer in time and the two levels come
// together (to the right: later, up to 40 ms, and further apart); per side: the factor on the found side and how far it
// moves in time, in samples. Anything else: left of the middle a straight line between the plain sound and the found
// pair; right of it everything but each side's direct sound grows; per side: the filter.
struct At { bool warp = false; double factor[2] = { 1.0, 1.0 }, move[2] = { 0.0, 0.0 }; Taps taps[2]; };
inline At pairAt (const Taps& a, const Taps& b, double s, double c)
{
    At R; double da = 0.0, ga = 0.0, db = 0.0, gb = 0.0;
    if (oneTap (a, da, ga) && oneTap (b, db, gb) && ga > 0.0 && gb > 0.0)
    {
        const double d0 = std::min (da, db); R.warp = true;
        R.factor[0] = std::pow (ga / c, s - 1.0); R.move[0] = std::min (s * (da - d0), (double) K) - (da - d0);
        R.factor[1] = std::pow (gb / c, s - 1.0); R.move[1] = std::min (s * (db - d0), (double) K) - (db - d0); return R;
    }
    for (int ch = 0; ch < 2; ++ch)
    {
        const Taps& v = ch == 0 ? a : b; Taps w; bool direct = false;
        for (const Tap& t : v) { if (t.lag == 0) { direct = true; w.push_back ({ 0, s <= 1.0 ? s * t.v + (1.0 - s) * c : t.v }); } else w.push_back ({ t.lag, s * t.v }); }
        if (! direct && s <= 1.0) w.insert (w.begin(), { 0, (1.0 - s) * c });
        R.taps[ch] = w;
    }
    return R;
}
inline double levelOf (const Taps& a, const Taps& b) { double e = 0.0; for (const Tap& t : a) e += t.v * t.v; for (const Tap& t : b) e += t.v * t.v; return std::sqrt (0.5 * e); }
// The sound itself with the group's Amount at s: 0 = the effect out, 1 = untouched, 2 = twice.
inline Stereo own (const Stereo& y, const Found& F, double s)
{
    if (F.mode == 0 || std::abs (s - 1.0) < 1e-12 || F.n != y.size() || F.x.size() != y.size()) return y;
    const long n = (long) y.size(); const Vec g = gate (F.spans, n); Stereo out = y;
    if (F.mode == COPY) { for (long i = 0; i < n; ++i) { const double c = (1.0 - s) * g[(size_t) i] * (double) F.x[(size_t) i]; out.l[(size_t) i] -= c; out.r[(size_t) i] += c; } return out; }
    const At A = pairAt (F.a, F.b, s, levelOf (F.a, F.b));
    for (int ch = 0; ch < 2; ++ch)
    {
        const Vec was = sparseConv (F.x, ch == 0 ? F.a : F.b); Vec& o = ch == 0 ? out.l : out.r;
        if (A.warp) { const Vec now = shift (was, A.move[ch]); for (long i = 0; i < n; ++i) o[(size_t) i] += g[(size_t) i] * (A.factor[ch] * now[(size_t) i] - was[(size_t) i]); }
        else { const Vec now = sparseConv (F.x, A.taps[ch]); for (long i = 0; i < n; ++i) o[(size_t) i] += g[(size_t) i] * (now[(size_t) i] - was[(size_t) i]); }
    }
    return out;
}
// The found effect on another sound: 0 = none, 1 = as found, 2 = twice. Each side goes through its own filter (a sound
// with nothing at its sides gets exactly what the effect does; a wide one keeps what it has at its sides).
inline Stereo onto (const Stereo& z, const Found& F, double s)
{
    if (F.mode == 0 || s < 1e-12) return z;
    const long n = (long) z.size(); Stereo out = z;
    if (F.mode == COPY)
    {
        Work W; Vec m ((size_t) n), c; for (long i = 0; i < n; ++i) m[(size_t) i] = 0.5 * (z.l[(size_t) i] + z.r[(size_t) i]);
        if (! convCentered (W, m, F.h, c, {})) return z;
        for (long i = 0; i < n; ++i) { out.l[(size_t) i] += s * c[(size_t) i]; out.r[(size_t) i] -= s * c[(size_t) i]; }
        return out;
    }
    const double c = levelOf (F.a, F.b); Taps a = F.a, b = F.b; for (Tap& t : a) t.v /= c; for (Tap& t : b) t.v /= c;
    const At A = pairAt (a, b, s, 1.0);
    for (int ch = 0; ch < 2; ++ch)
    {
        const Vec& in = ch == 0 ? z.l : z.r; Vec& o = ch == 0 ? out.l : out.r;
        if (A.warp) { o = shift (sparseConv (in, ch == 0 ? a : b), A.move[ch]); for (double& v : o) v *= A.factor[ch]; }
        else o = sparseConv (in, A.taps[ch]);
    }
    return out;
}

} // namespace sides
} // namespace dr
