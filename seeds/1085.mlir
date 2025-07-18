module {
  func.func @main(%arg0: tensor<70xi16>) -> tensor<70xi16> {
    %0 = tosa.clamp %arg0 {min_val = 0 : i16, max_val = 93 : i16} : (tensor<70xi16>) -> tensor<70xi16>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<70xi16>, tensor<70xi16>) -> tensor<70xi16>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<70xi16>) -> tensor<70xi16>
    return %2 : tensor<70xi16>
  }
}
