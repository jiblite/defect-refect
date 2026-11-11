// Prints the detector, deciders and settings readers for a 48 kHz WAV, for comparison with Python.
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include "Wav.h"
#include "../Source/Core/Inspector.h"
int main (int argc, char** argv)
{
    std::ifstream f (argv[1], std::ios::binary); std::vector<char> b ((std::istreambuf_iterator<char> (f)), {});
    dr::Models M; M.load (b.data(), b.size());
    for (int i = 2; i < argc; ++i)
    {
        dr::Stereo y; double sr; if (! dr::readWav (argv[i], y, sr)) { std::cout << "cannot read " << argv[i] << "\n"; continue; }
        auto w = dr::windows3s (y)[0];
        auto F = dr::features (w); auto o = M.run (F); auto h = dr::handFeatures (w);
        std::cout << "FILE " << argv[i] << "\nfeat0";
        for (int c = 0; c < 4; ++c) { double s = 0; for (int bb = 0; bb < 256; ++bb) for (int t = 0; t < F.frames; ++t) s += F.at (c, bb, t); std::cout << " " << s; }
        std::cout << "\nlogits"; for (float v : o.logits) std::cout << " " << v;
        std::cout << "\nhand"; for (float v : h) std::cout << " " << v;
        auto I = dr::inspect (M, y);
        std::cout << "\nprob"; for (auto& n : M.fx) std::cout << " " << n << "=" << I.prob[n];
        std::cout << "\nset"; for (auto& [fx, p] : I.settings) for (auto& [k, v] : p) std::cout << " " << fx << "." << k << "=" << v;
        std::cout << "\n";
    }
}
