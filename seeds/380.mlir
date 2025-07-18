module {
  func.func @main(%arg0: tensor<35x69xi32>, %arg1: tensor<1x69xi32>) -> (tensor<35x69xi1>, tensor<35x69xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<35x69xi32>, tensor<1x69xi32>) -> tensor<35x69xi32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<35x69xi32>, tensor<35x69xi32>) -> tensor<35x69xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<35x69xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<35x69xi32>
    %3 = tosa.greater %2, %0 : (tensor<35x69xi32>, tensor<35x69xi32>) -> tensor<35x69xi1>
    %4 = tosa.logical_not %3 : (tensor<35x69xi1>) -> tensor<35x69xi1>
    %5 = tosa.logical_not %4 : (tensor<35x69xi1>) -> tensor<35x69xi1>
    %6 = tosa.greater %2, %2 : (tensor<35x69xi32>, tensor<35x69xi32>) -> tensor<35x69xi1>
    return %5, %6 : tensor<35x69xi1>, tensor<35x69xi1>
  }
}
