module {
  func.func @main(%arg0: tensor<74x39x36xi16>, %arg1: tensor<71x2x2x50x31xi1>) -> (tensor<74x39x36xi16>, tensor<71x2x2x50x31xi1>) {
    %0 = tosa.clamp %arg0 {min_val = 2 : i16, max_val = 14 : i16} : (tensor<74x39x36xi16>) -> tensor<74x39x36xi16>
    %1 = tosa.logical_not %arg1 : (tensor<71x2x2x50x31xi1>) -> tensor<71x2x2x50x31xi1>
    return %0, %1 : tensor<74x39x36xi16>, tensor<71x2x2x50x31xi1>
  }
}
