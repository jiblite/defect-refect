#pragma once
// The filter engine (FILTER): a fixed low-pass, high-pass or band-pass on a sound, with or without a resonance.
//   find()    reads the filter's curve off one sound (Reader.h: band energies, six curves, the reader's nets), in two
//             looks: the second on the sound as it would be with the first answer undone; it only adds where the
//             first read a deep cut (the reader was taught curves held at -40 dB, and a skirt runs on under that)
//   own()     the sound itself with the group's Amount at s: 0 = the filter out, 1 = untouched, 2 = the filter twice;
//             Cutoff moves the curve along the frequency axis, Resonance scales its peak
//   onto()    the found curve on another sound: 0 = none, 1 = as found, 2 = twice
// The curve is not one of a few textbook shapes: it is 104 numbers, a gain for every twelfth of an octave, so a ladder
// filter, a state-variable one and a pair of two-pole sections are all read as what they did. It is put on and taken
// off as a minimum-phase filter (reader::toIr), so taking it off also undoes the filter's phase.
// Taking a filter out lifts what it cut. The lift ends at 40 dB, and earlier in a band that holds still while the
// rest of the sound moves: such a band is noise that came after the filter, and lifting it would only bring up hiss.
// What it does not do: a filter on one side only; a filter that moves (the moving-tone engine's FILTER SWEEP); a
// shelf or a bell of an equaliser, and a sound that is dull or thin by itself: those are not told from the sound's own
// colour by anyone, and are left to the MOVEMENT group's Spectral fader and to TONE MATCH.
#include "Reader.h"

namespace dr
{
namespace fixed
{
using Stop = room::Stop;
using reader::NBAND;
constexpr int minFrames = 20;                   // under 20 frames that count (0.9 s) nothing is read
constexpr double sureAt = 3.0;                  // the reader's own number (a logit) from which a filter is shown with the Sureness slider where it stands at first
constexpr double lowAt = 0.0;                   // ... and from which it is kept at all: such a filter shows when the slider is turned down
constexpr double deepAt = 6.0;                  // either way its curve must cut this many dB somewhere the sound can be heard
constexpr double skirtFrom = 24.0, skirtOver = 12.0;        // the second look adds nothing where the first read less than 24 dB of cut, all of its answer from 36 dB on
constexpr double maxLift = 40.0, noiseLift = 12.0, maxCut = 80.0, maxBoost = 24.0;
constexpr double shownSure = 0.65;              // what the Sureness slider cuts at by default: a filter just found shows this
constexpr double peakShown = 1.0;               // a peak of this many dB above the pass band is a resonance: the group then has a Resonance fader

enum Kind { NONE = 0, LOW = 1, HIGH = 2, BAND = 3 };
struct Found
{
    bool found = false, looked = false, stopped = false;
    size_t n = 0; int frames = 0;       // the sound it belongs to: its length, and the frames that counted
    double logit = -99.0, sure = 0.0;   // the reader's own number, and the same as a sureness 0 .. 1
    Vec curve;                          // dB per band, 0 in the pass band
    Vec lift;                           // how far each band may be lifted when the filter is taken out (dB)
    Vec apart;                          // how far the reader's nets lay apart per band in the first look (dB)
    int kind = NONE; double cutLo = 0, cutHi = 0;      // corners in Hz, where the curve passes -3 dB: a high-pass has cutLo, a low-pass cutHi, a band-pass both
    double peakDb = 0, peakHz = 0;      // the resonance: the curve's highest point above the pass band
    double depth = 0;                   // the deepest cut where the sound can be heard (dB)
};

inline double sureness (double logit) { return 1.0 / (1.0 + std::exp (-(logit - sureAt) * 0.5 - std::log (shownSure / (1.0 - shownSure)))); }

// What the curve is, in words and numbers: for the group's title line and its faders' ranges.
inline void describe (Found& F)
{
    const Vec& L = F.curve; F.kind = NONE; F.cutLo = F.cutHi = 0; F.peakDb = 0; F.peakHz = 0; if (L.size() != (size_t) NBAND) return;
    int top = 0; for (int b = 1; b < NBAND; ++b) if (L[(size_t) b] > L[(size_t) top]) top = b;
    if (L[(size_t) top] > 0.0) { F.peakDb = L[(size_t) top]; F.peakHz = reader::bandHz (top); }
    auto corner = [&] (int from, int step)      // from the pass band outwards: where the curve first passes -3 dB (between two bands)
    {
        for (int b = from; b + step >= 0 && b + step < NBAND; b += step)
            if (L[(size_t) b] > -3.0 && L[(size_t) (b + step)] <= -3.0)
            {
                const double t = (L[(size_t) b] + 3.0) / std::max (L[(size_t) b] - L[(size_t) (b + step)], 1e-9); return reader::bandHz (b) * std::pow (2.0, (double) step * t / 12.0);
            }
        return 0.0;
    };
    if (L[(size_t) top] <= -3.0) return;        // nothing passes: not a filter of this kind
    const bool lowCut = L[0] <= -6.0, highCut = L[(size_t) NBAND - 1] <= -6.0;
    if (lowCut) F.cutLo = corner (top, -1);
    if (highCut) F.cutHi = corner (top, +1);
    F.kind = lowCut && highCut ? BAND : lowCut ? HIGH : highCut ? LOW : NONE;
}

// How steeply the curve falls beyond a corner, in dB per octave (0 = not told): between where it passes -6 dB and where
// it passes -30 dB, or four fifths of its deepest point where it does not get that far. up: the upper flank (a
// low-pass's), otherwise the lower one.
inline double slopeOf (const Found& F, bool up)
{
    const Vec& L = F.curve; if (L.size() != (size_t) NBAND) return 0.0;
    int top = 0; for (int b = 1; b < NBAND; ++b) if (L[(size_t) b] > L[(size_t) top]) top = b;
    const int step = up ? 1 : -1; double deepest = 0.0; for (int b = top; b >= 0 && b < NBAND; b += step) deepest = std::min (deepest, L[(size_t) b]);
    if (deepest > -12.0) return 0.0;
    const double far = std::max (-30.0, 0.8 * deepest);
    auto pass = [&] (double level)       // from the pass band outwards: where the curve first passes this level (in bands, between two of them)
    {
        for (int b = top; b + step >= 0 && b + step < NBAND; b += step)
            if (L[(size_t) b] > level && L[(size_t) (b + step)] <= level) return (double) b + (double) step * (L[(size_t) b] - level) / std::max (L[(size_t) b] - L[(size_t) (b + step)], 1e-9);
        return -1.0;
    };
    const double a = pass (-6.0), b = pass (far); if (a < 0 || b < 0 || std::abs (b - a) < 0.25) return 0.0;
    return (-6.0 - far) / (std::abs (b - a) / 12.0);
}

// One sound's filter. rd: the reader taught with fixed filters (Resources/filter.bin).
inline Found find (const Stereo& x, const reader::Reader& rd, const Stop& stop = {})
{
    Found F; F.n = x.size(); if (! rd.ready()) return F;
    const reader::Energies E = reader::energies (x, stop); if (stop && stop()) { F.stopped = true; return F; }
    if (E.T < 1) return F;
    const reader::Curves C = reader::curves (E); F.frames = C.frames; F.looked = true; if (C.frames < minFrames) return F;
    Vec first, second; double s1 = -99, s2 = -99; rd.run (C, first, s1, &F.apart);
    Vec undo ((size_t) NBAND); for (int b = 0; b < NBAND; ++b) undo[(size_t) b] = -first[(size_t) b];
    rd.run (reader::curves (E, undo), second, s2);
    F.curve.resize ((size_t) NBAND);
    for (int b = 0; b < NBAND; ++b)
        F.curve[(size_t) b] = std::clamp (first[(size_t) b] + second[(size_t) b] * std::clamp ((-first[(size_t) b] - skirtFrom) / skirtOver, 0.0, 1.0), -maxCut, maxBoost);
    F.logit = s1; F.sure = sureness (s1);
    // where the sound can be heard: bands within 50 dB of its strongest (as it is now)
    double strongest = -1e9; for (int b = 0; b < NBAND; ++b) strongest = std::max (strongest, (double) C.at (0, b));
    // the lift's cap per band: a band that was cut and holds still while the bands that were left alone move is noise
    Vec spread ((size_t) NBAND), passSpread; for (int b = 0; b < NBAND; ++b) spread[(size_t) b] = (double) C.at (4, b) - (double) C.at (2, b);
    for (int b = 0; b < NBAND; ++b) if (std::abs (F.curve[(size_t) b]) < 3.0 && C.at (0, b) > strongest - 50.0) passSpread.push_back (spread[(size_t) b]);
    const double moves = passSpread.empty() ? 0.0 : median (passSpread);
    F.lift.assign ((size_t) NBAND, maxLift);
    if (moves >= 14.0) for (int b = 0; b < NBAND; ++b) if (F.curve[(size_t) b] < -3.0 && spread[(size_t) b] <= 0.5 * moves) F.lift[(size_t) b] = noiseLift;
    // the deepest cut where the clean sound would be heard: where the sound, with the curve undone, is within 50 dB of its strongest
    double cleanTop = -1e9; for (int b = 0; b < NBAND; ++b) cleanTop = std::max (cleanTop, (double) C.at (0, b) - F.curve[(size_t) b]);
    for (int b = 0; b < NBAND; ++b) if ((double) C.at (0, b) - F.curve[(size_t) b] > cleanTop - 50.0) F.depth = std::max (F.depth, -F.curve[(size_t) b]);
    describe (F);
    F.found = F.logit >= lowAt && F.depth >= deepAt && F.kind != NONE;
    return F;
}

// The Cutoff fader (0 .. 2, 1 = as found) as octaves the curve is moved by: two down to two up.
inline double cutoffOctaves (double fader) { return 2.0 * (std::clamp (fader, 0.0, 2.0) - 1.0); }
// The found curve as the faders shape it: moved along the frequency axis, its part above the pass band scaled.
inline Vec shaped (const Found& F, double octaves, double resonance)
{
    Vec L = reader::moved (F.curve, octaves); const double r = std::clamp (resonance, 0.0, 2.0);
    if (r != 1.0) for (auto& v : L) if (v > 0.0) v *= r;
    return L;
}
inline double softCap (double v, double cap) { const double k = 0.75 * cap; return v <= k ? v : k + (cap - k) * std::tanh ((v - k) / std::max (cap - k, 1e-9)); }

// The sound itself: the found filter out and the one the faders give in, as one curve. s: the Amount (0 .. 2).
inline Stereo own (const Stereo& y, const Found& F, double s, double octaves = 0.0, double resonance = 1.0, const Stop& stop = {})
{
    if (! F.found || F.curve.size() != (size_t) NBAND || (s == 1.0 && octaves == 0.0 && resonance == 1.0)) return y;
    const Vec want = shaped (F, octaves, resonance); Vec L ((size_t) NBAND);
    for (int b = 0; b < NBAND; ++b)
    {
        const double v = std::clamp (s, 0.0, 2.0) * want[(size_t) b] - F.curve[(size_t) b];
        L[(size_t) b] = v > 0.0 ? softCap (v, F.lift.size() == (size_t) NBAND ? F.lift[(size_t) b] : maxLift) : v;
    }
    Stereo r = reader::through (y, L, -maxCut, maxLift, stop); return r.size() == y.size() ? r : y;
}
// Another sound through the found filter. s: 0 = none, 1 = as found, 2 = the curve twice.
inline Stereo onto (const Stereo& y, const Found& F, double s, double octaves = 0.0, double resonance = 1.0, const Stop& stop = {})
{
    if (! F.found || F.curve.size() != (size_t) NBAND || s <= 0.0) return y;
    Vec L = shaped (F, octaves, resonance); for (auto& v : L) v *= std::min (s, 2.0);
    Stereo r = reader::through (y, L, -maxCut, maxBoost, stop); return r.size() == y.size() ? r : y;
}

} // namespace fixed
} // namespace dr
