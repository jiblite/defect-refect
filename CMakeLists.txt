cmake_minimum_required(VERSION 3.22)
project(DefectRefect VERSION 0.3.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>") # static runtime: single portable .exe
endif()

# JUCE is fetched at configure time. To use a local copy instead:
#   -DFETCHCONTENT_SOURCE_DIR_JUCE=/path/to/JUCE
include(FetchContent)
FetchContent_Declare(JUCE
  GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
  GIT_TAG 8.0.4
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(JUCE)

# The trained models, built into the exe.
juce_add_binary_data(dr_models SOURCES Resources/models.bin Resources/models27.bin)

juce_add_gui_app(DefectRefect
  PRODUCT_NAME "Defect Refect"
  BUNDLE_ID com.defectrefect.app
  COMPANY_NAME "Defect Refect")

target_sources(DefectRefect PRIVATE Source/App/Main.cpp)
target_include_directories(DefectRefect PRIVATE Source)
target_compile_definitions(DefectRefect PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:DefectRefect,JUCE_PRODUCT_NAME>"
  JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:DefectRefect,JUCE_VERSION>")
if(MSVC)
  target_compile_definitions(DefectRefect PRIVATE NOMINMAX _USE_MATH_DEFINES)
  target_compile_options(DefectRefect PRIVATE /bigobj)
endif()

target_link_libraries(DefectRefect PRIVATE
  dr_models
  juce::juce_gui_extra
  juce::juce_audio_formats
  juce::juce_audio_utils
  juce::juce_audio_devices
  PUBLIC
  juce::juce_recommended_config_flags
  juce::juce_recommended_lto_flags)

# Headless check of the processing (no JUCE): Tests/render.cpp
add_executable(drrender Tests/render.cpp)
find_package(Threads REQUIRED)
target_link_libraries(drrender PRIVATE Threads::Threads)
