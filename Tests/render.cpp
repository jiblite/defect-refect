// render models.bin src.wav [target.wav|-] out.wav [fader=value ...]  : the app's processing without the UI
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include "Wav.h"
#include "../Source/Core/Estimate.h"
static void writeWav (const std::string& p, const dr::Stereo& y)
{
    std::ofstream f (p, std::ios::binary); const uint32_t n = (uint32_t) y.size(), data = n * 2 * 4;
    auto w32 = [&] (uint32_t v) { f.write ((const char*) &v, 4); }; auto w16 = [&] (uint16_t v) { f.write ((const char*) &v, 2); };
    f.write ("RIFF", 4); w32 (36 + data); f.write ("WAVEfmt ", 8); w32 (16); w16 (3); w16 (2); w32 (48000); w32 (48000 * 8); w16 (8); w16 (32); f.write ("data", 4); w32 (data);
    double pk = std::max (y.peak(), 1e-9); const double g = pk > 0.99 ? 0.99 / pk : 1.0;
    for (uint32_t i = 0; i < n; ++i) { float a = (float) (y.l[i] * g), b = (float) (y.r[i] * g); f.write ((const char*) &a, 4); f.write ((const char*) &b, 4); }
}
int main (int argc, char** argv)
{
    using clk = std::chrono::steady_clock; auto t0 = clk::now();
    std::ifstream f (argv[1], std::ios::binary); std::vector<char> b ((std::istreambuf_iterator<char> (f)), {});
    dr::Models M; M.load (b.data(), b.size());
    dr::Stereo src, tgt; double sr; dr::readWav (argv[2], src, sr);
    const bool two = std::string (argv[3]) != "-"; if (two) dr::readWav (argv[3], tgt, sr);
    std::map<std::string, double> fad;
    for (int i = 5; i < argc; ++i) { std::string a = argv[i]; auto e = a.find ('='); fad[a.substr (0, e)] = std::stod (a.substr (e + 1)); }
    auto I = dr::inspect (M, src); auto t1 = clk::now();
    auto found = dr::estimate (I, src); auto t2 = clk::now();
    for (auto& F : found) { std::cout << F.name << " p=" << F.prob; for (auto& [k, v] : F.est) std::cout << " " << k << "=" << v; std::cout << "\n"; }
    auto out = two ? dr::applyTo (tgt, found, fad) : dr::reshape (src, found, fad); auto t3 = clk::now();
    writeWav (argv[4], out);
    auto ms = [] (auto a, auto b) { return (int) std::chrono::duration_cast<std::chrono::milliseconds> (b - a).count(); };
    std::cout << "inspect " << ms (t0, t1) << " ms, estimate " << ms (t1, t2) << " ms, render " << ms (t2, t3) << " ms\n";
}
