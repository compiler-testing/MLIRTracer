module {
  func.func @main(%arg0: tensor<88x26xi64>, %arg1: tensor<88x1xi64>, %arg2: tensor<93x57x32x29x22x44xf32>) -> (tensor<88x26xi1>, tensor<93x57x32x29x22x44xf32>, tensor<93x57x32x29x22x44xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<88x26xi64>, tensor<88x1xi64>) -> tensor<88x26xi1>
    %1 = tosa.sigmoid %arg2 : (tensor<93x57x32x29x22x44xf32>) -> tensor<93x57x32x29x22x44xf32>
    %2 = tosa.logical_and %0, %0 : (tensor<88x26xi1>, tensor<88x26xi1>) -> tensor<88x26xi1>
    %3 = tosa.pow %1, %1 : (tensor<93x57x32x29x22x44xf32>, tensor<93x57x32x29x22x44xf32>) -> tensor<93x57x32x29x22x44xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<93x57x32x29x22x44xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<93x57x32x29x22x44xf32>
    %5 = tosa.rsqrt %1 : (tensor<93x57x32x29x22x44xf32>) -> tensor<93x57x32x29x22x44xf32>
    %6 = tosa.pow %4, %1 : (tensor<93x57x32x29x22x44xf32>, tensor<93x57x32x29x22x44xf32>) -> tensor<93x57x32x29x22x44xf32>
    return %2, %5, %6 : tensor<88x26xi1>, tensor<93x57x32x29x22x44xf32>, tensor<93x57x32x29x22x44xf32>
  }
}
