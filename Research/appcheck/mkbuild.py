# Writes two scripts from the flags of the local release build of the app (a ninja build folder):
#   build.sh         the app with its own code (Main.cpp and every header of Source/) under the address and the
#                    undefined-behaviour sanitizer, linked against the JUCE objects of the release build (not instrumented)
#   syntax_main.sh   Main.cpp through two more compilers, syntax only: clang (as on the Mac) and mingw-w64 (the number
#                    sizes of Windows: long is 32 bits)
#   python3 mkbuild.py [ninja build folder] [output folder]
import re, sys
B = sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/drbuild'; O = sys.argv[2] if len(sys.argv) > 2 else '/tmp/claude-0/drasan'
SRC = '/home/claude/defect-refect/Source/App/Main.cpp'; VERSION = re.search(r'project\(DefectRefect VERSION (\S+)', open('/home/claude/defect-refect/CMakeLists.txt').read()).group(1)
nin = open(B + '/build.ninja').read()
i = nin.index('build CMakeFiles/DefectRefect.dir/Source/App/Main.cpp.o:'); blk = nin[i:nin.index('\n\n', i)]
get = lambda k: re.search(r'^  %s = (.*)$' % k, blk, re.M).group(1)
defines = re.sub(r'(JUCE_APPLICATION_VERSION_STRING=\\")[^\\]*', r'\g<1>' + VERSION, get('DEFINES').replace('$ ', ' ').replace('$$', '$')); includes = get('INCLUDES')
j = nin.index('build DefectRefect_artefacts/Release/Defect$ Refect:'); line = nin[j:nin.index('\n', j)]
objs = line.split(': ', 1)[1].split(' | ')[0].split()[1:]; assert objs[0].endswith('Main.cpp.o') and len(objs) == 17, (len(objs), objs[0])
objs = [O + '/Main.o'] + [B + '/' + o for o in objs[1:]]; san = '-fsanitize=address,undefined -fno-omit-frame-pointer'
open(O + '/build.sh', 'w').write('''#!/bin/bash
set -e
cd %s
date -u
nice -n 19 /usr/bin/c++ %s %s -O1 -g1 %s -std=gnu++17 -o %s/Main.o -c %s
nice -n 19 /usr/bin/c++ -O3 -DNDEBUG %s %s -o "%s/Defect Refect" %s/libdr_models.a /usr/lib/x86_64-linux-gnu/libasound.so /usr/lib/x86_64-linux-gnu/libfontconfig.so /usr/lib/x86_64-linux-gnu/libfreetype.so -lrt -ldl -lpthread -flto=2
date -u; ls -la "%s/Defect Refect"
''' % (B, defines, includes, san, O, SRC, san, ' '.join(objs), O, B, O))
win_def = defines.replace(' -DLINUX=1', ''); win_inc = ' '.join(x for x in includes.split() if 'freetype' not in x and 'libpng' not in x and 'LV2_SDK' not in x)
open(O + '/syntax_main.sh', 'w').write('''#!/bin/bash
cd %s
echo "== clang"; nice -n 19 clang++ %s %s -std=gnu++17 -fsyntax-only %s 2>&1 | grep -E "error|Error" | head -20; echo "clang exit ${PIPESTATUS[0]}"
echo "== mingw"; nice -n 19 x86_64-w64-mingw32-g++ %s %s -std=gnu++17 -fsyntax-only -include cstring -include string.h %s 2>&1 | grep -E "error|Error" | head -20; echo "mingw exit ${PIPESTATUS[0]}"
''' % (B, defines, includes, SRC, win_def, win_inc, SRC))
print('written: %s/build.sh, %s/syntax_main.sh (version %s)' % (O, O, VERSION))
