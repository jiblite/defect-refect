// The bend engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/bend_parity.py):
//   drbend bend.bin a.wav b.wav|- prefix amount
// One sound (b given as -): the fader works on a. Two sounds: a's bend is put on b. amount 0 .. 2. Prints what was
// found on a, and writes as raw float64:
//   prefix.feat.f64    what the meter sees: 184 numbers per half-wave (the upper first), then the two peaks
//   prefix.bend.f64    the squeeze per half-wave and level (2 x 11), then the logit and how far the loudest values were squeezed
//   prefix.out.f64     the result, left and right in turn
// The engine itself only acts on a bend it counts as found; with DR_BEND_ANY set in the environment whatever was read
// is used (for the comparison of the numbers).
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
#include "../Source/Core/Bend.h"

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
        if (argc < 6) { std::cerr << "usage: drbend bend.bin a.wav b.wav|- prefix amount\n"; return 2; }
        const std::vector<char> model = bytes (argv[1]); dr::bend::Meter m; m.load (model.data(), model.size());
        const dr::Stereo a = sound (argv[2]); const std::string second = argv[3], prefix = argv[4]; const double amount = std::atof (argv[5]);
        const auto t0 = std::chrono::steady_clock::now();
        dr::bend::Found F = dr::bend::find (a, m);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf ("found %d logit %.3f sure %.3f top %.2f dB kind %s peaks %.5f %.5f\n", (int) F.found, F.logit, F.sure, F.top, F.hard() ? "distortion" : "saturation", F.vp[0], F.vp[1]);
        {
            std::vector<float> X ((size_t) 2 * (size_t) dr::bend::NF, 0.0f); double pk[2] = { 0, 0 };
            if (a.size() >= dr::bend::minLen) { dr::bend::features (a, 1.0, X.data(), pk[0]); dr::bend::features (a, -1.0, X.data() + dr::bend::NF, pk[1]); }
            dr::Vec feat (X.begin(), X.end()); feat.push_back (pk[0]); feat.push_back (pk[1]); writeRaw (prefix + ".feat.f64", { feat });
        }
        dr::Vec S; for (int h = 0; h < 2; ++h) for (int g = 0; g < dr::bend::NG; ++g) S.push_back (F.S[h][g]);
        S.push_back (F.logit); S.push_back (F.top); writeRaw (prefix + ".bend.f64", { S });
        if (std::getenv ("DR_BEND_ANY") != nullptr && F.looked) F.found = true;
        const dr::Stereo out = second == "-" ? dr::bend::own (a, F, amount) : dr::bend::onto (sound (second), F, amount);
        const auto t2 = std::chrono::steady_clock::now();
        dr::Vec inter (out.size() * 2); for (size_t i = 0; i < out.size(); ++i) { inter[2 * i] = out.l[i]; inter[2 * i + 1] = out.r[i]; }
        writeRaw (prefix + ".out.f64", { inter });
        std::printf ("find %.0f ms, render %.0f ms\n", std::chrono::duration<double, std::milli> (t1 - t0).count(), std::chrono::duration<double, std::milli> (t2 - t1).count());
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drbend: " << e.what() << "\n"; return 1; }
}
