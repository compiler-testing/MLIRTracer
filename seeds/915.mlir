module {
  func.func @main(%arg0: tensor<40x43xi64>, %arg1: tensor<82x85x55x2xf32>, %arg2: tensor<30x3x66x50xf32>, %arg3: tensor<30xf32>) -> (tensor<82x174x124x30xi1>, tensor<80x43xi64>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<40x43xi64>, !tosa.shape<2>) -> tensor<80x43xi64>
    %1 = tosa.clz %0 : (tensor<80x43xi64>) -> tensor<80x43xi64>
    %2 = tosa.bitwise_or %1, %1 : (tensor<80x43xi64>, tensor<80x43xi64>) -> tensor<80x43xi64>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 82, 174, 124, 30>} : (tensor<82x85x55x2xf32>, tensor<30x3x66x50xf32>, tensor<30xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<82x174x124x30xf32>
    %4 = tosa.equal %3, %3 : (tensor<82x174x124x30xf32>, tensor<82x174x124x30xf32>) -> tensor<82x174x124x30xi1>
    %5 = tosa.arithmetic_right_shift %4, %4 {round = true} : (tensor<82x174x124x30xi1>, tensor<82x174x124x30xi1>) -> tensor<82x174x124x30xi1>
    %6 = tosa.maximum %2, %2 : (tensor<80x43xi64>, tensor<80x43xi64>) -> tensor<80x43xi64>
    return %5, %6 : tensor<82x174x124x30xi1>, tensor<80x43xi64>
  }
}
