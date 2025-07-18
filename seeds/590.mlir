module {
  func.func @main(%arg0: tensor<11x47x27xi16>, %arg1: tensor<11x27x70xi16>, %arg2: tensor<88xf32>) -> (tensor<11x47x70xi16>, tensor<88xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<11x47x27xi16>, tensor<11x27x70xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<11x47x70xi16>
    %1 = tosa.ceil %arg2 : (tensor<88xf32>) -> tensor<88xf32>
    %2 = tosa.ceil %1 : (tensor<88xf32>) -> tensor<88xf32>
    %3 = tosa.clamp %2 {min_val = 3.600000e+01 : f32, max_val = 4.000000e+01 : f32} : (tensor<88xf32>) -> tensor<88xf32>
    return %0, %3 : tensor<11x47x70xi16>, tensor<88xf32>
  }
}
