// What the room engine says about each sound of a list, for a study of its sureness: one JSON line per sound.
//   build (from this folder): g++ -O2 -std=c++17 -pthread -o rsure rsure.cpp
//   rsure ResourcesDir list.txt      (list: one WAV path per line; a tab and a number after the path: only the first so many seconds)
// Per sound: the engine's answer on the sound as it is and with 3 s of silence after it (the app's two looks): found,
// the app's sureness, its evidence, and per band strength, decay time, evidence, width; plus how the model's answer is
// spread (shares of the live cells under -3, -6, -12, -20 dB; the room part's share of the energy per third of the bands).
#include <fstream>
#include <iostream>
#include <iterator>
#include "../../../Tests/Wav.h"
#include "../../../Source/Core/Estimate.h"
static void look (const dr::Stereo& y, const dr::room::Net& net, const dr::room::Net* net2, const char* tag)
{
    const dr::room::Found f = dr::room::find (y, net, net2); const dr::RoomEvidence e = dr::roomEvidence (f);
    std::cout << "\"" << tag << "\": {\"found\": " << (f.found ? 1 : 0) << ", \"sure\": " << dr::roomSureness (f) << ", \"share\": " << e.share << ", \"deep\": " << e.deep << ", \"bands\": " << e.bands << ", \"moments\": " << e.moments
              << ", \"decay\": " << e.decay << ", \"strength\": " << e.strength << ", \"own\": " << e.own;
    const size_t n = f.gains.size(); const int NB = dr::room::NB, T = f.picture.frames;
    if (n > 0 && f.picture.LE.size() == n)
    {
        const double top = f.picture.top(); size_t live = 0, u3 = 0, u6 = 0, u12 = 0, u20 = 0; double room3[3] = { 0, 0, 0 }, all3[3] = { 0, 0, 0 };
        for (size_t i = 0; i < n; ++i)
        {
            if (! (f.picture.LE[i] > top - 50.0)) continue;
            const double g = f.gains[i], E = std::pow (10.0, (f.picture.LE[i] - top) / 10.0), G = std::pow (10.0, g / 10.0); ++live; u3 += g < -3; u6 += g < -6; u12 += g < -12; u20 += g < -20;
            const int b = (int) (i % (size_t) NB), k = b * 3 / NB; room3[k] += E * (1 - G); all3[k] += E;
        }
        const double L = (double) std::max<size_t> (live, 1);
        std::cout << ", \"frames\": " << T << ", \"live\": " << (double) live / (double) n << ", \"u3\": " << u3 / L << ", \"u6\": " << u6 / L << ", \"u12\": " << u12 / L << ", \"u20\": " << u20 / L;
        for (int k = 0; k < 3; ++k) std::cout << ", \"r" << k << "\": " << (all3[k] > 0 ? room3[k] / all3[k] : 0.0);
    }
    for (int q = 0; q < 4; ++q)
    {
        const dr::room::Band& B = q == 0 ? f.kdb : q == 1 ? f.t60 : q == 2 ? f.ev : f.width; std::cout << ", \"" << (q == 0 ? "kdb" : q == 1 ? "t60" : q == 2 ? "ev" : "width") << "\": [";
        for (int b = 0; b < NB; ++b) std::cout << (b ? "," : "") << (f.found ? (double) B[(size_t) b] : 0.0);
        std::cout << "]";
    }
    std::cout << "}";
}
int main (int argc, char** argv)
{
    if (argc < 3) return 1;
    const std::string dir = argv[1]; dr::room::Net net, net2;
    { std::ifstream f (dir + "/room.bin", std::ios::binary); const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); net.load (b.data(), b.size()); }
    { std::ifstream f (dir + "/room2.bin", std::ios::binary); if (f) { const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); net2.load (b.data(), b.size()); } }
    std::ifstream L (argv[2]); std::string p;
    while (std::getline (L, p))
    {
        if (p.empty()) continue;
        double cut = 0; const auto tab = p.find ('\t'); if (tab != std::string::npos) { cut = std::stod (p.substr (tab + 1)); p = p.substr (0, tab); }
        dr::Stereo y; double sr = 0; if (! dr::readWav (p, y, sr)) { std::cerr << "cannot read " << p << "\n"; continue; }
        if (cut > 0 && (size_t) (cut * 48000.0) < y.size()) { y.l.resize ((size_t) (cut * 48000.0)); y.r.resize ((size_t) (cut * 48000.0)); }
        std::cout << "{\"path\": \"" << p << "\", \"cut\": " << cut << ", \"sec\": " << (double) y.size() / 48000.0 << ", "; look (y, net, net2.ready() ? &net2 : nullptr, "as");
        dr::Stereo z = y; const size_t n = y.size() + (size_t) (dr::roomEndSilence * dr::SR); z.l.resize (n, 0.0); z.r.resize (n, 0.0);
        std::cout << ", "; look (z, net, net2.ready() ? &net2 : nullptr, "end"); std::cout << "}" << std::endl;
    }
    return 0;
}
