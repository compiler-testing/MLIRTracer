module {
  func.func @main(%arg0: tensor<89x42x69xi16>, %arg1: tensor<89x69x52xi16>, %arg2: tensor<87xf32>) -> (tensor<89x42x52xi16>, tensor<87xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<89x42x69xi16>, tensor<89x69x52xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<89x42x52xi16>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<89x42x52xi16>, tensor<89x42x52xi16>) -> tensor<89x42x52xi16>
    %2 = tosa.ceil %arg2 : (tensor<87xf32>) -> tensor<87xf32>
    return %1, %2 : tensor<89x42x52xi16>, tensor<87xf32>
  }
}
