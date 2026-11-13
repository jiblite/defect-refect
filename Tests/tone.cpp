// The tone engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/tone_parity.py):
//   drtone a.wav b.wav prefix [detail]
// writes, as raw float64 files of 4097 values per curve unless said otherwise,
//   prefix.spec.f64      a's long-term spectra: both channels together, left, right (3 curves)
//   prefix.comb.f64      the curve of the reflections found on a (all zero: none); the reflections themselves are printed
//   prefix.match.f64     Tone Match, the curve that gives b the tone of a at the given Detail (default 0.6): one for both channels
//   prefix.match2.f64    the same per channel (2 curves)
//   prefix.ir.f64        prefix.match as an impulse response (16384 values)
//   prefix.flat.wav      a with the found reflections taken out
//   prefix.put.wav       b with the one-curve match on it;  prefix.put2.wav  with the per-channel match
// and prints how far the one sound is a filtered copy of the other (two versions of one recording or not), by how much the
// second is late against the first, and the time each step took.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Wav.h"
#include "../Source/Core/Tone.h"

static bool writeWav (const std::string& p, const dr::Stereo& y)       // float32, level untouched
{
    std::ofstream f (p, std::ios::binary); const uint32_t n = (uint32_t) y.size(), data = n * 2 * 4;
    auto w32 = [&] (uint32_t v) { f.write ((const char*) &v, 4); }; auto w16 = [&] (uint16_t v) { f.write ((const char*) &v, 2); };
    f.write ("RIFF", 4); w32 (36 + data); f.write ("WAVEfmt ", 8); w32 (16); w16 (3); w16 (2); w32 (48000); w32 (48000 * 8); w16 (8); w16 (32); f.write ("data", 4); w32 (data);
    std::vector<float> v ((size_t) n * 2); for (uint32_t i = 0; i < n; ++i) { v[(size_t) i * 2] = (float) y.l[i]; v[(size_t) i * 2 + 1] = (float) y.r[i]; }
    f.write ((const char*) v.data(), (std::streamsize) (v.size() * 4));
    return (bool) f;
}
static void writeRaw (const std::string& p, const std::vector<const dr::Vec*>& vs)
{
    std::ofstream f (p, std::ios::binary);
    for (auto* v : vs) f.write ((const char*) v->data(), (std::streamsize) (v->size() * sizeof (double)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static dr::Stereo sound (const std::string& path)
{
    dr::Stereo y; double sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    if (std::abs (sr - 48000.0) > 0.5) throw std::runtime_error (path + " is not at 48 kHz");
    return y;
}

int main (int argc, char** argv)
{
    try
    {
        if (argc < 4) { std::cout << "drtone a.wav b.wav prefix [detail]\n"; return 1; }
        const dr::Stereo a = sound (argv[1]), b = sound (argv[2]); const std::string pre = argv[3]; const double detail = argc > 4 ? std::atof (argv[4]) : 0.6;
        using clock = std::chrono::steady_clock; auto t0 = clock::now();
        auto lap = [&] (const char* what) { const auto t1 = clock::now(); std::cout << "  " << what << ": " << std::chrono::duration<double> (t1 - t0).count() << " s\n"; t0 = t1; };
        const dr::tone::Found fa = dr::tone::analyse (a); lap ("the first sound's spectra and reflections");
        const dr::tone::Spectra sb = dr::tone::spectra (b); lap ("the second sound's spectra");
        std::cout << "reflections: " << fa.echoes.size(); for (auto& e : fa.echoes) std::printf ("  delay %.4f samples gain %.4f", e.delay, e.gain); std::cout << "\n";
        { long lag = 0; const double c = dr::tone::pairCoherence (a, b, &lag); std::printf ("coherence: %.3f (%s), the second sound is %ld samples late\n", c, c > dr::tone::sameCoherence ? "two versions of one recording" : "two different sounds", lag); lap ("the coherence"); }
        if (fa.spec.silent || sb.silent) { std::cout << "one of the sounds is silent\n"; return 0; }
        writeRaw (pre + ".spec.f64", { &fa.spec.both, &fa.spec.left, &fa.spec.right });
        const dr::Vec comb = dr::tone::combCurve (fa.echoes, fa.spec.both); writeRaw (pre + ".comb.f64", { &comb });
        const dr::Vec m = dr::tone::matchCurve (fa.spec.both, sb.both, detail), ml = dr::tone::matchCurve (fa.spec.left, sb.left, detail), mr = dr::tone::matchCurve (fa.spec.right, sb.right, detail);
        writeRaw (pre + ".match.f64", { &m }); writeRaw (pre + ".match2.f64", { &ml, &mr }); lap ("the curves");
        const dr::Vec h = dr::tone::toIr (m); writeRaw (pre + ".ir.f64", { &h }); lap ("the impulse response");
        dr::Vec neg = comb; for (auto& v : neg) v = -v;
        writeWav (pre + ".flat.wav", dr::tone::gainFilter (a, neg, neg)); lap ("the reflections taken out");
        writeWav (pre + ".put.wav", dr::tone::gainFilter (b, m, m)); lap ("the match put on (one curve)");
        writeWav (pre + ".put2.wav", dr::tone::gainFilter (b, ml, mr)); lap ("the match put on (per channel)");
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drtone: " << e.what() << "\n"; return 1; }
}
