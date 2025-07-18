module {
  func.func @main(%arg0: tensor<1x92x74x29x58x90xi8>, %arg1: tensor<1x92x74x1x1x1xi8>) -> tensor<1x92x74x29x58x90xi1> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<1x92x74x29x58x90xi8>, tensor<1x92x74x1x1x1xi8>) -> tensor<1x92x74x29x58x90xi8>
    %1 = tosa.bitwise_not %0 : (tensor<1x92x74x29x58x90xi8>) -> tensor<1x92x74x29x58x90xi8>
    %2 = tosa.bitwise_and %1, %0 : (tensor<1x92x74x29x58x90xi8>, tensor<1x92x74x29x58x90xi8>) -> tensor<1x92x74x29x58x90xi8>
    %3 = tosa.greater %2, %1 : (tensor<1x92x74x29x58x90xi8>, tensor<1x92x74x29x58x90xi8>) -> tensor<1x92x74x29x58x90xi1>
    %4 = tosa.abs %3 : (tensor<1x92x74x29x58x90xi1>) -> tensor<1x92x74x29x58x90xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<1x92x74x29x58x90xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x92x74x29x58x90xi1>
    return %5 : tensor<1x92x74x29x58x90xi1>
  }
}
