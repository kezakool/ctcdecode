cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)


find_package(Python3 REQUIRED COMPONENTS Development)

# Add the Python include directories
include_directories(${Python3_INCLUDE_DIRS})

# TEST EXECUTABLE

# add google test library
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Create an executable for the tests
add_executable(all_tests ${CMAKE_SOURCE_DIR}/tests/cpp/test_build_fst.cpp ${CMAKE_SOURCE_DIR}/tests/cpp/test_decoder.cpp )
# add_executable(decoder_test ${CMAKE_SOURCE_DIR}/tests/cpp/test_decoder.cpp )

# Link Google Test to the test executable
target_link_libraries(all_tests gtest gtest_main build_fst_lib ${Python3_LIBRARIES})
target_sources(all_tests PRIVATE ${CMAKE_SOURCE_DIR}/tools/build_fst.cpp)
target_compile_definitions(all_tests PUBLIC TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")


# Add the tests to CTest
include(GoogleTest)
gtest_discover_tests(all_tests)