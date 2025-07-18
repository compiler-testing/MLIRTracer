module {
  func.func @main(%arg0: tensor<9x11xi64>, %arg1: tensor<97x73x58x10xf32>) -> (tensor<9x11xi64>, tensor<97x73x58x10xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<9x11xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<9x11xi64>
    %1 = tosa.rsqrt %arg1 : (tensor<97x73x58x10xf32>) -> tensor<97x73x58x10xf32>
    return %0, %1 : tensor<9x11xi64>, tensor<97x73x58x10xf32>
  }
}
