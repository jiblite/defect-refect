// The MOVEMENT faders without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/move_parity.py):
//   drmove a.wav b.wav|- prefix pitch loudness tone stereo spectral [own.cents.f64 [other.cents.f64]]
// One sound (b given as -): the faders work on a. Two sounds: they work on b, and a is the other sound (the one whose
// movement the right half of a fader brings). The five fader values run 0 .. 2 (1 = nothing is done).
// A pitch curve can be given as a raw float64 file (cents per 10 ms) in place of the reader's own; - = the reader's.
// Writes, as raw float64 files, what was found on the sound worked on (prefix.own.*) and on the other (prefix.other.*):
//   .loud.f64    the level per 5 ms, then the sound's own level
//   .space.f64   pan, width and power per 10 ms, then pan0, width0 and whether the sound is mono (1 or 0)
//   .tone.f64    the 18 bands' movement per 10 ms (row after row), then the 18 weights
//   .spec.f64    the long-term spectrum (4097 values)
//   .pitch.f64   the pitch movement in cents per 10 ms, then where it is followed (1 or 0)
// and prefix.out.f64: the result, left and right in turn. Prints the time each step took.
// With DR_FLOW_LOG set in the environment the pitch reader says what it does.
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
#include "../Source/Core/Movement.h"

static void writeRaw (const std::string& p, const std::vector<dr::Vec>& vs)
{
    std::ofstream f (p, std::ios::binary);
    for (auto& v : vs) f.write ((const char*) v.data(), (std::streamsize) (v.size() * sizeof (double)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static dr::Vec readRaw (const std::string& p)
{
    std::ifstream f (p, std::ios::binary | std::ios::ate); if (! f) throw std::runtime_error ("cannot read " + p);
    const std::streamsize n = f.tellg(); f.seekg (0); dr::Vec v ((size_t) n / sizeof (double)); f.read ((char*) v.data(), n); return v;
}
static dr::Stereo sound (const std::string& path)
{
    dr::Stereo y; double sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    if (std::abs (sr - 48000.0) > 0.5) throw std::runtime_error (path + " is not at 48 kHz");
    return y;
}
static void writeFound (const std::string& pre, const dr::move::Found& F)
{
    writeRaw (pre + ".loud.f64", { F.loud.db, { F.loud.ref } });
    writeRaw (pre + ".space.f64", { F.space.pan, F.space.width, F.space.power, { F.space.pan0, F.space.width0, F.space.mono ? 1.0 : 0.0 } });
    writeRaw (pre + ".tone.f64", { F.tone.m, F.tone.w }); writeRaw (pre + ".spec.f64", { F.spec });
    dr::Vec h (F.pitch.held.begin(), F.pitch.held.end()); writeRaw (pre + ".pitch.f64", { F.pitch.cents, h });
}

int main (int argc, char** argv)
{
    try
    {
        if (argc < 9) { std::cout << "drmove a.wav b.wav|- prefix pitch loudness tone stereo spectral [own.cents.f64 [other.cents.f64]]\n"; return 1; }
        const bool two = std::string (argv[2]) != "-"; const dr::Stereo a = sound (argv[1]), b = two ? sound (argv[2]) : dr::Stereo(); const std::string pre = argv[3];
        dr::move::Faders f; f.pitch = std::atof (argv[4]); f.loudness = std::atof (argv[5]); f.tone = std::atof (argv[6]); f.stereo = std::atof (argv[7]); f.spectral = std::atof (argv[8]);
        const std::string ownCents = argc > 9 ? argv[9] : "-", otherCents = argc > 10 ? argv[10] : "-";
        using clock = std::chrono::steady_clock; auto t0 = clock::now();
        auto lap = [&] (const char* what) { const auto t1 = clock::now(); std::printf ("  %s: %.3f s\n", what, std::chrono::duration<double> (t1 - t0).count()); t0 = t1; };
        const dr::Stereo& x = two ? b : a;
        dr::move::Found own = dr::move::findPlain (x); lap ("found on the sound worked on (all but pitch)");
        auto pitchOf = [&] (dr::move::Found& F, const dr::Stereo& s, const std::string& given)
        {
            if (given != "-") { F.pitch.cents = readRaw (given); F.pitch.held.assign (F.pitch.cents.size(), 1); }
            else F.pitch = dr::move::track (s, {}, std::getenv ("DR_FLOW_LOG") != nullptr ? dr::flow::Log ([] (const std::string& l) { std::cout << l << "\n"; }) : dr::flow::Log());
        };
        pitchOf (own, x, ownCents); lap ("its pitch movement");
        writeFound (pre + ".own", own);
        dr::move::Found other;
        if (two) { other = dr::move::findPlain (a); lap ("found on the other sound (all but pitch)"); pitchOf (other, a, otherCents); lap ("its pitch movement"); writeFound (pre + ".other", other); }
        const dr::Stereo y = dr::move::apply (x, own, two ? &other : nullptr, f); lap ("the faders put on");
        dr::Vec out (y.size() * 2); for (size_t i = 0; i < y.size(); ++i) { out[2 * i] = y.l[i]; out[2 * i + 1] = y.r[i]; }
        writeRaw (pre + ".out.f64", { out });
        std::printf ("sound of %.2f s; pitch followed on %.0f %% of it\n", (double) x.size() / 48000.0, own.pitch.held.empty() ? 0.0 : 100.0 * (double) std::count (own.pitch.held.begin(), own.pitch.held.end(), (char) 1) / (double) own.pitch.held.size());
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 2; }
}
