module {
  func.func @main(%arg0: tensor<57x37x5x88xi16>, %arg1: tensor<73xi1>, %arg2: tensor<1xi1>, %arg3: tensor<47x74x55x86x7xf32>) -> (tensor<57x37x5x88xi16>, tensor<73xi1>, tensor<47x74x55x86x7xi1>, tensor<47x74x55x86x7xf32>, tensor<47x74x55x86x7xi1>, tensor<47x74x55x86x7xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<57x37x5x88xi16>) -> tensor<57x37x5x88xi16>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<73xi1>, tensor<1xi1>) -> tensor<73xi1>
    %2 = tosa.log %arg3 : (tensor<47x74x55x86x7xf32>) -> tensor<47x74x55x86x7xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<47x74x55x86x7xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<47x74x55x86x7xf32>
    %4 = tosa.sub %3, %3 : (tensor<47x74x55x86x7xf32>, tensor<47x74x55x86x7xf32>) -> tensor<47x74x55x86x7xf32>
    %5 = tosa.equal %3, %2 : (tensor<47x74x55x86x7xf32>, tensor<47x74x55x86x7xf32>) -> tensor<47x74x55x86x7xi1>
    %6 = tosa.minimum %4, %2 : (tensor<47x74x55x86x7xf32>, tensor<47x74x55x86x7xf32>) -> tensor<47x74x55x86x7xf32>
    %7 = tosa.greater %6, %4 : (tensor<47x74x55x86x7xf32>, tensor<47x74x55x86x7xf32>) -> tensor<47x74x55x86x7xi1>
    %8 = tosa.rsqrt %4 : (tensor<47x74x55x86x7xf32>) -> tensor<47x74x55x86x7xf32>
    %9 = tosa.bitwise_and %5, %5 : (tensor<47x74x55x86x7xi1>, tensor<47x74x55x86x7xi1>) -> tensor<47x74x55x86x7xi1>
    %10 = tosa.greater %4, %4 : (tensor<47x74x55x86x7xf32>, tensor<47x74x55x86x7xf32>) -> tensor<47x74x55x86x7xi1>
    return %0, %1, %7, %8, %9, %10 : tensor<57x37x5x88xi16>, tensor<73xi1>, tensor<47x74x55x86x7xi1>, tensor<47x74x55x86x7xf32>, tensor<47x74x55x86x7xi1>, tensor<47x74x55x86x7xi1>
  }
}
