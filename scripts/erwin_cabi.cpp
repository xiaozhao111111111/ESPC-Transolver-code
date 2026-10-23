#include "balltree.h"
// ABI-only adapter. All median partition and tree logic is from balltree.h.
extern "C" __declspec(dllexport) void bt_build(
    const double *data, intptr_t *work, intptr_t *out, bool *mask,
    intptr_t n, intptr_t dims, intptr_t depth) {
    build_tree(data, work, out, mask, 0, n, 0, dims, 0, depth);
}
extern "C" __declspec(dllexport) void bt_partition(
    const double *data, intptr_t *work, intptr_t *out,
    intptr_t n, intptr_t dims, intptr_t depth) {
    partition_ball_tree(data, work, out, 0, n, 0, dims, 0, depth);
}
