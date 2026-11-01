#include <cstdlib>
#include <random>
#include "bench-helper.h"

uint32_t* createInput(uint64_t N, uint64_t maxValue) {
    uint32_t* keys = (uint32_t*) malloc(N * sizeof(uint32_t));
    if (!keys) return nullptr;

    std::mt19937 gen((uint32_t) N);

    for (uint64_t i = 0; i < N; ++i)
        keys[i] = gen() % maxValue + 1;
    return keys;
}