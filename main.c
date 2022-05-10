#include "stdint.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "ieee_fp16.h"


struct iree_hal_executable_dispatch_state_v0_t {
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint16_t workgroup_size_z;
    uint16_t push_constant_count;
    uint32_t workgroup_count_x;
    uint32_t workgroup_count_y;
    uint16_t workgroup_count_z;
    uint8_t max_concurrency;
    uint8_t binding_count;
    uint32_t *binding_lengths;
    uint8_t **binding_ptrs;
    uint64_t *push_constants;
};

struct iree_hal_executable_workgroup_state_v0_t {
    uint32_t workgroup_id_x;
    uint32_t workgroup_id_y;
    uint16_t workgroup_id_z;
    uint16_t reserved;
    uint32_t processor_id;
    uint8_t **local_memory;
    uint32_t local_memory_size;
};

// typedef struct iree_hal_executable_dispatch_state_v0_t {
//   // Workgroup size chosen for the dispatch. For compilation modes where the
//   // workgroup size is constant this may be ignored.
//   uint32_t workgroup_size[3];

//   // Total number of available 4 byte push constant values in |push_constants|.
//   uint64_t push_constant_count;

//   // Total workgroup count for the dispatch. This is sourced from either the
//   // original dispatch call (for iree_hal_command_buffer_dispatch) or the
//   // indirection buffer (for iree_hal_command_buffer_dispatch_indirect).
//   uint32_t workgroup_count[3];

//   // Total number of binding base pointers in |binding_ptrs| and
//   // |binding_lengths|. The set is packed densely based on which bindings are
//   // used (known at compile-time).
//   uint64_t binding_count;

//   // |push_constant_count| values.
//   const uint32_t* push_constants;
//   // Base pointers to each binding buffer.
//   void* const* binding_ptrs;
//   // The length of each binding in bytes, 1:1 with |binding_ptrs|.
//   const uint64_t* binding_lengths;

//   // NOTE: the above fields are frequently accessed and should be kept together
//   // to ensure cache-friendly behavior. The first instructions every dispatch
//   // executes are loads from the fields and we want to avoid a cascade of
//   // cache misses. Less-frequently used fields can follow.
// } iree_hal_executable_dispatch_state_v0_t;

int32_t forward_dispatch_0(uint8_t *unused, struct iree_hal_executable_workgroup_state_v0_t *state,
                           struct iree_hal_executable_workgroup_state_v0_t *workgroup_id);

int main (int argc, char *argv[]) {
  float x[4] = {1.1, 2.2, 3.3, 4.4};
  float y[4] = {5.5, 6.6, 7.7, 8.8};
  float z[4] = {0.0, 0.0, 0.0, 0.0};
  void *binding_ptrs[3] = {x, y, z};
  size_t binding_lengths[3] =  {sizeof(x), sizeof(y), sizeof(z)};
  uint32_t push_constants[1] = {0};
  uint32_t workgroup_ids[3] = {0, 0, 0};

    struct iree_hal_executable_dispatch_state_v0_t state = {
        .workgroup_size_x = 1,
        .workgroup_size_y = 1,
        .workgroup_size_z = 1,
        .push_constant_count = 0,
        .workgroup_count_x = 1,
        .workgroup_count_y = 1,
        .workgroup_count_z = 1,
        .binding_count = 3,
        .binding_ptrs = binding_ptrs,
        .push_constants = push_constants,
        .binding_lengths = binding_lengths,
    };
    struct iree_hal_executable_workgroup_state_v0_t wg = {
        .workgroup_id_x = 0,
        .workgroup_id_y = 0,
        .workgroup_id_z = 0,
        .reserved = 0,
        .processor_id = 0,
        .local_memory = 0x0,
        .local_memory_size = 0,
    };

  uint8_t *unused;
  printf("going for it!\n");
  int32_t res = forward_dispatch_0(unused, &state, workgroup_ids);
  printf("failed!\n");
  for (int i = 0; i < 4; i++) {
    printf("Got z[%d] = %f [Expected: %f]\n", i, z[i], x[i] + y[i]);
  }
  return 0;
}
