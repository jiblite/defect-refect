// The room engine without the UI (48 kHz WAV in, float32 WAV out with the level untouched):
//   drroom model.bin in.wav out.wav amount               the sound with its room at that amount: 0 = out, 1 = as it is, 2 = twice as loud
//   drroom model.bin in.wav --numbers [prefix]           the room in numbers: found or not, and per band strength, decay time, evidence, width
//   drroom model.bin in.wav --ir out.wav [length bright width [seed]]            the found room as an impulse response
//   drroom model.bin in.wav --onto b.wav out.wav amount [length bright width]    the found room on a second sound
//   drroom model.bin in.wav --one out.wav amount length bright width             the sound with its room at that amount and reshaped
//   drroom model.bin in.wav --dump prefix                what the model sees and says, as raw files for comparison with Python:
//                                                        prefix.LE.f32 .U64.f32 .U256.f32 and the gains .G.f32 are [frame][41 bands] float32,
//                                                        the model's input .IN.f32 is [4 planes][frame][41 bands] (7 planes and the first
//                                                        answer .G1.f32 when there is a second model); prefix.NUM.f64 (also written by
//                                                        --numbers prefix) is float64: found (1 or 0), then 41 each of strength, decay time,
//                                                        evidence and width
// Last arguments, in any order: second=model2.bin (a model for a second pass), piece=N (the models run N frames at a time: the answer
// must not depend on it).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "Wav.h"
#include "../Source/Core/RoomEngine.h"

static bool writeWav (const std::string& p, const dr::Stereo& y)       // float32, level untouched
{
    std::ofstream f (p, std::ios::binary); const uint32_t n = (uint32_t) y.size(), data = n * 2 * 4;
    auto w32 = [&] (uint32_t v) { f.write ((const char*) &v, 4); }; auto w16 = [&] (uint16_t v) { f.write ((const char*) &v, 2); };
    f.write ("RIFF", 4); w32 (36 + data); f.write ("WAVEfmt ", 8); w32 (16); w16 (3); w16 (2); w32 (48000); w32 (48000 * 8); w16 (8); w16 (32); f.write ("data", 4); w32 (data);
    std::vector<float> v ((size_t) n * 2); for (uint32_t i = 0; i < n; ++i) { v[(size_t) i * 2] = (float) y.l[i]; v[(size_t) i * 2 + 1] = (float) y.r[i]; }
    f.write ((const char*) v.data(), (std::streamsize) (v.size() * 4));
    return (bool) f;
}
template <typename Out, typename V> static void writeRaw (const std::string& p, const V& v)
{
    std::vector<Out> o; o.reserve (v.size()); for (const auto x : v) o.push_back ((Out) x);
    std::ofstream f (p, std::ios::binary);
    f.write ((const char*) o.data(), (std::streamsize) (o.size() * sizeof (Out)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static double number (const char* s, const char* what)
{
    char* end = nullptr; const double v = std::strtod (s, &end);
    if (end == s || *end != '\0' || ! std::isfinite (v)) throw std::runtime_error (std::string ("the ") + what + " is not a number: " + s);
    return v;
}
static dr::room::Net model (const std::string& path)
{
    std::ifstream f (path, std::ios::binary); if (! f) throw std::runtime_error ("cannot open " + path);
    const std::vector<char> blob ((std::istreambuf_iterator<char> (f)), {});
    dr::room::Net net; net.load (blob.data(), blob.size()); return net;
}
static dr::Stereo sound (const std::string& path)
{
    dr::Stereo y; double sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    if (sr != 48000.0) throw std::runtime_error ("the sound must be at 48 kHz: " + path);
    return y;
}

int main (int argc, char** argv)
{
    using namespace dr::room; using clk = std::chrono::steady_clock;
    auto ms = [] (clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli> (b - a).count(); };
    int piece = 4096; std::string secondPath;
    for (; argc > 1; --argc)
    {
        const std::string a = argv[argc - 1];
        if (a.rfind ("piece=", 0) == 0) piece = std::atoi (a.c_str() + 6); else if (a.rfind ("second=", 0) == 0) secondPath = a.substr (7); else break;
    }
    const std::string mode = argc > 3 ? argv[3] : ""; const int more = argc - 4;        // arguments after the mode (after out.wav in the first form)
    const bool ok = piece >= 1 && (mode == "--dump" ? more == 1 : mode == "--numbers" ? more <= 1 : mode == "--ir" ? (more == 1 || more == 4 || more == 5)
                                 : mode == "--onto" ? (more == 3 || more == 6) : mode == "--one" ? more == 5 : (argc == 5 && mode.rfind ("--", 0) != 0));
    if (! ok)
    {
        std::cerr << "usage: drroom model.bin in.wav out.wav amount\n       drroom model.bin in.wav --numbers [prefix]\n       drroom model.bin in.wav --ir out.wav [length bright width [seed]]\n"
                     "       drroom model.bin in.wav --onto b.wav out.wav amount [length bright width]\n       drroom model.bin in.wav --one out.wav amount length bright width\n"
                     "       drroom model.bin in.wav --dump prefix\n       last arguments, any of: second=model2.bin piece=N\n";
        return 2;
    }
    try
    {
        // everything that can be wrong with the arguments first
        double amount = 1.0, shape[3] = { 1.0, 1.0, 1.0 }; uint64_t seed = 1; std::string out, prefix; dr::Stereo b;
        auto faders = [&] (int at) { const char* what[3] = { "length", "brightness", "width" }; for (int i = 0; i < 3; ++i) shape[i] = number (argv[at + i], what[i]); };
        if (mode == "--dump") prefix = argv[4];
        else if (mode == "--numbers") prefix = more == 1 ? argv[4] : "";
        else if (mode == "--ir") { out = argv[4]; if (more >= 4) faders (5); if (more == 5) seed = std::strtoull (argv[8], nullptr, 10); }
        else if (mode == "--onto") { out = argv[5]; amount = number (argv[6], "amount"); if (more == 6) faders (7); }
        else if (mode == "--one") { out = argv[4]; amount = number (argv[5], "amount"); faders (6); }
        else { out = argv[3]; amount = number (argv[4], "amount"); }
        const Net net = model (argv[1]); Net second; if (! secondPath.empty()) second = model (secondPath);
        const dr::Stereo y = sound (argv[2]); if (mode == "--onto") b = sound (argv[4]);

        const auto t0 = clk::now();
        const Found f = find (y, net, secondPath.empty() ? nullptr : &second, {}, {}, piece); const auto t1 = clk::now();
        std::cout << (double) y.size() / 48000.0 << " s, " << f.picture.frames << " frames: find " << ms (t0, t1) << " ms (" << (f.found ? "a room" : "no room") << ")";
        auto written = [&] (const char* what, const dr::Stereo& in, const dr::Stereo& res)
        {
            std::cout << ", " << what << " " << ms (t1, clk::now()) << " ms | in rms " << in.rms() << " peak " << in.peak() << ", out rms " << res.rms() << " peak " << res.peak() << "\n";
            if (! writeWav (out, res)) throw std::runtime_error ("cannot write " + out);
        };
        auto numbers = [&]
        {
            dr::Vec v (1, f.found ? 1.0 : 0.0);
            for (const Band* p : { &f.kdb, &f.t60, &f.ev, &f.width }) v.insert (v.end(), p->begin(), p->end());
            writeRaw<double> (prefix + ".NUM.f64", v); writeRaw<float> (prefix + ".G.f32", f.gains);
        };
        if (mode == "--dump")
        {
            std::cout << "\n";
            std::vector<float> in = inputs (f.picture);
            if (! secondPath.empty())
            {
                const std::vector<float> g1 = net.run (in, f.picture.frames, piece), x = secondLook (f.picture.LE, g1, f.picture.frames);
                in.insert (in.end(), x.begin(), x.end()); writeRaw<float> (prefix + ".G1.f32", g1);
            }
            writeRaw<float> (prefix + ".LE.f32", f.picture.LE); writeRaw<float> (prefix + ".U64.f32", f.picture.U64); writeRaw<float> (prefix + ".U256.f32", f.picture.U256); writeRaw<float> (prefix + ".IN.f32", in);
            numbers();
        }
        else if (mode == "--numbers")
        {
            std::cout << "\n"; const Bands B;
            if (f.found)
            {
                std::cout << "band      Hz  strength dB  decay time s  evidence  width\n";
                for (size_t i = 0; i < NB; ++i) { char line[96]; std::snprintf (line, sizeof (line), "%4d %7.0f %12.2f %13.3f %9.0f %6.3f\n", (int) i, B.fc[i], f.kdb[i], f.t60[i], f.ev[i], f.width[i]); std::cout << line; }
            }
            if (! prefix.empty()) numbers();
        }
        else if (mode == "--ir")
        {
            if (! f.found) throw std::runtime_error ("no room shows in this sound: there is no impulse response to write");
            const dr::Stereo h = roomIr (f, shape[0], shape[1], shape[2], seed);
            std::cout << ", impulse response " << ms (t1, clk::now()) << " ms (" << (double) h.size() / 48000.0 << " s, rms " << h.rms() << " peak " << h.peak() << ")\n";
            if (! writeWav (out, h)) throw std::runtime_error ("cannot write " + out);
        }
        else if (mode == "--onto") written ("onto", b, onto (b, f, amount, shape[0], shape[1], shape[2]));
        else if (mode == "--one") written ("one sound", y, oneSound (y, f, amount, shape[0], shape[1], shape[2]));
        else written ("render", y, render (y, f.gains, amount));
    }
    catch (const std::exception& e) { std::cerr << "drroom: " << e.what() << "\n"; return 1; }
    return 0;
}
