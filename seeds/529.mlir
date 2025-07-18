module {
  func.func @main(%arg0: tensor<69x46x81x6x96x62xi64>, %arg1: tensor<69x46x81x6x96x47xi64>, %arg2: tensor<66x58x97x28xf32>) -> (tensor<69x46x81x6x96x109xi64>, tensor<66x58x97x28xf32>, tensor<69x46x81x6x96x109xi64>, tensor<58x2x66x28xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 5 : i32} : (tensor<69x46x81x6x96x62xi64>, tensor<69x46x81x6x96x47xi64>) -> tensor<69x46x81x6x96x109xi64>
    %1 = tosa.clamp %0 {min_val = -46 : i64, max_val = -7 : i64} : (tensor<69x46x81x6x96x109xi64>) -> tensor<69x46x81x6x96x109xi64>
    %2 = tosa.tanh %arg2 : (tensor<66x58x97x28xf32>) -> tensor<66x58x97x28xf32>
    %3 = tosa.tanh %2 : (tensor<66x58x97x28xf32>) -> tensor<66x58x97x28xf32>
    %4 = tosa.reduce_min %2 {axis = 2 : i32} : (tensor<66x58x97x28xf32>) -> tensor<66x58x1x28xf32>
    %5 = "tosa.const"() {values = dense<[1, 2, 0, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
    %6 = tosa.transpose %4 {perms = array<i32: 1, 2, 0, 3>} : (tensor<66x58x1x28xf32>) -> tensor<58x1x66x28xf32>
    %7 = tosa.clamp %3 {min_val = -7.000000e+00 : f32, max_val = -1.000000e+00 : f32} : (tensor<66x58x97x28xf32>) -> tensor<66x58x97x28xf32>
    %in_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %8 = tosa.negate %7, %in_zp_8, %out_zp_8 : (tensor<66x58x97x28xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<66x58x97x28xf32>
    %9 = tosa.concat %6, %6 {axis = 1 : i32} : (tensor<58x1x66x28xf32>, tensor<58x1x66x28xf32>) -> tensor<58x2x66x28xf32>
    %10 = tosa.logical_left_shift %0, %0 : (tensor<69x46x81x6x96x109xi64>, tensor<69x46x81x6x96x109xi64>) -> tensor<69x46x81x6x96x109xi64>
    %11 = tosa.add %9, %9 : (tensor<58x2x66x28xf32>, tensor<58x2x66x28xf32>) -> tensor<58x2x66x28xf32>
    return %1, %8, %10, %11 : tensor<69x46x81x6x96x109xi64>, tensor<66x58x97x28xf32>, tensor<69x46x81x6x96x109xi64>, tensor<58x2x66x28xf32>
  }
}
