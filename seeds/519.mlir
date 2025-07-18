module {
  func.func @main(%arg0: tensor<20x25x18xi1>, %arg1: tensor<20x18x40xi1>, %arg2: tensor<93xi64>, %arg3: tensor<93xi64>) -> (tensor<20x25x40xi1>, tensor<93xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<20x25x18xi1>, tensor<20x18x40xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<20x25x40xi1>
    %1 = tosa.add %0, %0 : (tensor<20x25x40xi1>, tensor<20x25x40xi1>) -> tensor<20x25x40xi1>
    %2 = tosa.greater_equal %arg2, %arg3 : (tensor<93xi64>, tensor<93xi64>) -> tensor<93xi1>
    return %1, %2 : tensor<20x25x40xi1>, tensor<93xi1>
  }
}
