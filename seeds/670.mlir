module {
  func.func @main(%arg0: tensor<58xi8>, %arg1: tensor<1xi8>, %arg2: tensor<34x94x46x15x77x30xf32>, %arg3: tensor<47x8x79x19xi1>) -> (tensor<1x8x79x19xi1>, tensor<34x94x46x15x77x30xf32>, tensor<34x94x46x15x77x60xf32>, tensor<58xi8>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<58xi8>, tensor<1xi8>) -> tensor<58xi8>
    %1 = tosa.maximum %0, %0 : (tensor<58xi8>, tensor<58xi8>) -> tensor<58xi8>
    %2 = tosa.ceil %arg2 : (tensor<34x94x46x15x77x30xf32>) -> tensor<34x94x46x15x77x30xf32>
    %3 = tosa.rsqrt %2 : (tensor<34x94x46x15x77x30xf32>) -> tensor<34x94x46x15x77x30xf32>
    %4 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<47x8x79x19xi1>) -> tensor<1x8x79x19xi1>
    %5 = tosa.arithmetic_right_shift %4, %4 {round = true} : (tensor<1x8x79x19xi1>, tensor<1x8x79x19xi1>) -> tensor<1x8x79x19xi1>
    %6 = tosa.clamp %2 {min_val = -5.900000e+01 : f32, max_val = 7.500000e+01 : f32} : (tensor<34x94x46x15x77x30xf32>) -> tensor<34x94x46x15x77x30xf32>
    %7 = tosa.concat %6, %3 {axis = 5 : i32} : (tensor<34x94x46x15x77x30xf32>, tensor<34x94x46x15x77x30xf32>) -> tensor<34x94x46x15x77x60xf32>
    %8 = tosa.minimum %6, %2 : (tensor<34x94x46x15x77x30xf32>, tensor<34x94x46x15x77x30xf32>) -> tensor<34x94x46x15x77x30xf32>
    %9 = tosa.logical_left_shift %0, %1 : (tensor<58xi8>, tensor<58xi8>) -> tensor<58xi8>
    %10 = tosa.sub %7, %7 : (tensor<34x94x46x15x77x60xf32>, tensor<34x94x46x15x77x60xf32>) -> tensor<34x94x46x15x77x60xf32>
    %11 = tosa.bitwise_and %1, %9 : (tensor<58xi8>, tensor<58xi8>) -> tensor<58xi8>
    return %5, %8, %10, %11 : tensor<1x8x79x19xi1>, tensor<34x94x46x15x77x30xf32>, tensor<34x94x46x15x77x60xf32>, tensor<58xi8>
  }
}
