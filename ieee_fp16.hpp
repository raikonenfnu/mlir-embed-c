#include "stdio.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "iostream"

extern "C" {
  float __gnu_h2f_ieee(short param) {
    unsigned short expHalf16 = param & 0x7C00;
    int exp1 = (int)expHalf16;
    unsigned short mantissa16 = param & 0x03FF;
    int mantissa1 = (int)mantissa16;
    int sign = (int)(param & 0x8000);
    sign = sign << 16;

    // nan or inf
    if (expHalf16 == 0x7C00) {
      // nan
      if (mantissa16 > 0) {
        int res = (0x7FC00000 | sign);
        float fres = *((float*)(&res));
        return fres;
      }
      // inf
      int res = (0x7F800000 | sign);
      float fres = *((float*)(&res));
      return fres;
    }
    if (expHalf16 != 0) {
      exp1 += ((127 - 15) << 10);  // exponents converted to float32 bias
      int res = (exp1 | mantissa1);
      res = res << 13;
      res = (res | sign);
      float fres = *((float*)(&res));
      return fres;
    }

    int xmm1 = exp1 > (1 << 10) ? exp1 : (1 << 10);
    xmm1 = (xmm1 << 13);
    xmm1 += ((127 - 15 - 10) << 23);  // add the bias difference to xmm1
    xmm1 = xmm1 | sign;               // Combine with the sign mask

    float res = (float)mantissa1;  // Convert mantissa to float
    res *= *((float*)(&xmm1));

    return res;
  }

  short __gnu_f2h_ieee(float param) {
    unsigned int param_bit = *((unsigned int*)(&param));
    int sign = param_bit >> 31;
    int mantissa = param_bit & 0x007FFFFF;
    int exp = ((param_bit & 0x7F800000) >> 23) + 15 - 127;
    short res;
    if (exp > 0 && exp < 30) {
      // use rte rounding mode, round the significand, combine sign, exponent and
      // significand into a short.
      res = (sign << 15) | (exp << 10) | ((mantissa + 0x00001000) >> 13);
    } else if (param_bit == 0) {
      res = 0;
    } else {
      if (exp <= 0) {
        if (exp < -10) {
          // value is less than min half float point
          res = 0;
        } else {
          // normalized single, magnitude is less than min normal half float
          // point.
          mantissa = (mantissa | 0x00800000) >> (1 - exp);
          // round to nearest
          if ((mantissa & 0x00001000) > 0) {
            mantissa = mantissa + 0x00002000;
          }
          // combine sign & mantissa (exp is zero to get denormalized number)
          res = (sign << 15) | (mantissa >> 13);
        }
      } else if (exp == (255 - 127 + 15)) {
        if (mantissa == 0) {
          // input float is infinity, return infinity half
          res = (sign << 15) | 0x7C00;
        } else {
          // input float is NaN, return half NaN
          res = (sign << 15) | 0x7C00 | (mantissa >> 13);
        }
      } else {
        // exp > 0, normalized single, round to nearest
        if ((mantissa & 0x00001000) > 0) {
          mantissa = mantissa + 0x00002000;
          if ((mantissa & 0x00800000) > 0) {
            mantissa = 0;
            exp = exp + 1;
          }
        }
        if (exp > 30) {
          // exponent overflow - return infinity half
          res = (sign << 15) | 0x7C00;
        } else {
          // combine sign, exp and mantissa into normalized half
          res = (sign << 15) | (exp << 10) | (mantissa >> 13);
        }
      }
    }
    return res;
  }

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

int32_t forward_dispatch_0(uint8_t *unused, struct iree_hal_executable_dispatch_state_v0_t *state,
                           struct iree_hal_executable_workgroup_state_v0_t *workgroup_id);

void _mlir_ciface_OpName() {
    std::cout<<"wut wut\n";
    printf("hello from cpp side!\n");
}

// TODO: SUpport fp16 with custom Fp16 type.
void _mlir_ciface_mem_promote(UnrankedMemRefType<float> *m) {
  DynamicMemRefType<float> src(*m);
  std::cout<<"ptr:"<<src.basePtr<<"\n";
  std::cout<<"numel:"<<src.sizes[0]<<"\n";
  for (int i = 0; i < src.sizes[0]; i++) {
    std::cout<<src.data[i]<<",";
  }
  std::cout<<"\n";
}

// void _mlir_ciface_mem_promote(int64_t rank, void *ptr) {
//   std::cout<<"rank:"<<rank<<"\n";
//   std::cout<<"received:"<<ptr<<"\n";
//   // printMemrefF32(rank, ptr);
//   // UnrankedMemRefType<float> descriptor = {rank, ptr};
//   // _mlir_ciface_printMemrefF32(&descriptor);
//   // impl::printMemRef(DynamicMemRefType<float>(descriptor));
// }
} // extern "C"
