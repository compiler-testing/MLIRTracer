module {
  func.func @main(%arg0: tensor<29x70x12xi8>, %arg1: tensor<29x12x60xi8>, %arg2: tensor<84x3x4x71x45x76xf32>) -> (tensor<29x70x60xi8>, tensor<84x3x4x71x45x76xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<29x70x12xi8>, tensor<29x12x60xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<29x70x60xi8>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<29x70x60xi8>, tensor<29x70x60xi8>) -> tensor<29x70x60xi8>
    %2 = tosa.clamp %1 {min_val = 22 : i8, max_val = 91 : i8} : (tensor<29x70x60xi8>) -> tensor<29x70x60xi8>
    %3 = tosa.bitwise_and %2, %1 : (tensor<29x70x60xi8>, tensor<29x70x60xi8>) -> tensor<29x70x60xi8>
    %4 = tosa.rsqrt %arg2 : (tensor<84x3x4x71x45x76xf32>) -> tensor<84x3x4x71x45x76xf32>
    return %3, %4 : tensor<29x70x60xi8>, tensor<84x3x4x71x45x76xf32>
  }
}
