module {
  func.func @main(%arg0: tensor<95x51x15xi64>, %arg1: tensor<95x15x86xi64>, %arg2: tensor<53x3x6x22x25xf32>) -> (tensor<53x3x6x22x25xf32>, tensor<190x102x86xi64>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<95x51x15xi64>, tensor<95x15x86xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<95x51x86xi64>
    %t_1 = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<95x51x86xi64>, !tosa.shape<3>) -> tensor<190x102x86xi64>
    %2 = tosa.sigmoid %arg2 : (tensor<53x3x6x22x25xf32>) -> tensor<53x3x6x22x25xf32>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<190x102x86xi64>, tensor<190x102x86xi64>) -> tensor<190x102x86xi64>
    %4 = tosa.bitwise_and %3, %1 : (tensor<190x102x86xi64>, tensor<190x102x86xi64>) -> tensor<190x102x86xi64>
    %5 = tosa.clamp %4 {min_val = -1 : i64, max_val = 94 : i64} : (tensor<190x102x86xi64>) -> tensor<190x102x86xi64>
    return %2, %5 : tensor<53x3x6x22x25xf32>, tensor<190x102x86xi64>
  }
}
