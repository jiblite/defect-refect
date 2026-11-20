// The filter engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/filter_parity.py):
//   drfilter filter.bin a.wav b.wav|- prefix amount cutoff resonance
// One sound (b given as -): the faders work on a. Two sounds: a's filter is put on b. amount 0 .. 2; cutoff 0 .. 2
// (1 = as found, two octaves down to two up); resonance 0 .. 2. Prints what was found on a, and writes as raw float64:
//   prefix.curve.f64   the curve (104), the lift's cap per band (104), how far the nets lay apart (104), then the logit
//   prefix.out.f64     the result, left and right in turn
// The engine itself only acts on a filter it counts as found; with DR_FILTER_ANY set in the environment whatever was
// read is used (for the comparison of the numbers).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Wav.h"
#include "../Source/Core/Filter.h"

static void writeRaw (const std::string& p, const std::vector<dr::Vec>& vs)
{
    std::ofstream f (p, std::ios::binary);
    for (auto& v : vs) f.write ((const char*) v.data(), (std::streamsize) (v.size() * sizeof (double)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static std::vector<char> bytes (const std::string& p)
{
    std::ifstream f (p, std::ios::binary); if (! f) throw std::runtime_error ("cannot read " + p);
    return std::vector<char> ((std::istreambuf_iterator<char> (f)), std::istreambuf_iterator<char>());
}
static dr::Stereo sound (const std::string& path)
{
    dr::Stereo y; double sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    return y;
}
int main (int argc, char** argv)
{
    try
    {
        if (argc < 8) { std::cerr << "usage: drfilter filter.bin a.wav b.wav|- prefix amount cutoff resonance\n"; return 2; }
        const std::vector<char> model = bytes (argv[1]); dr::reader::Reader rd; rd.load (model.data(), model.size());
        const dr::Stereo a = sound (argv[2]); const std::string second = argv[3], prefix = argv[4];
        const double amount = std::atof (argv[5]), cutoff = std::atof (argv[6]), resonance = std::atof (argv[7]);
        const auto t0 = std::chrono::steady_clock::now();
        dr::fixed::Found F = dr::fixed::find (a, rd);
        const auto t1 = std::chrono::steady_clock::now();
        static const char* kinds[] = { "none", "low-pass", "high-pass", "band-pass" };
        std::printf ("found %d logit %.3f sure %.3f frames %d kind %s cutLo %.1f cutHi %.1f peak %.2f dB at %.1f Hz depth %.1f dB\n", (int) F.found, F.logit, F.sure, F.frames, kinds[F.kind], F.cutLo, F.cutHi, F.peakDb, F.peakHz, F.depth);
        if (F.curve.size() != (size_t) dr::reader::NBAND) { F.curve.assign ((size_t) dr::reader::NBAND, 0.0); F.lift.assign ((size_t) dr::reader::NBAND, 0.0); F.apart.assign ((size_t) dr::reader::NBAND, 0.0); }
        writeRaw (prefix + ".curve.f64", { F.curve, F.lift, F.apart, dr::Vec { F.logit } });
        if (std::getenv ("DR_FILTER_ANY") != nullptr) F.found = true;
        const double oct = dr::fixed::cutoffOctaves (cutoff);
        const dr::Stereo out = second == "-" ? dr::fixed::own (a, F, amount, oct, resonance) : dr::fixed::onto (sound (second), F, amount, oct, resonance);
        const auto t2 = std::chrono::steady_clock::now();
        dr::Vec inter (out.size() * 2); for (size_t i = 0; i < out.size(); ++i) { inter[2 * i] = out.l[i]; inter[2 * i + 1] = out.r[i]; }
        writeRaw (prefix + ".out.f64", { inter });
        std::printf ("find %.0f ms, render %.0f ms\n", std::chrono::duration<double, std::milli> (t1 - t0).count(), std::chrono::duration<double, std::milli> (t2 - t1).count());
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drfilter: " << e.what() << "\n"; return 1; }
}
