module {
  func.func @main(%arg0: tensor<42x99xi64>, %arg1: tensor<97x17x55x39x74x100xf32>) -> (tensor<42x99xi64>, tensor<97x17x55x39x74x100xf32>) {
    %0 = tosa.clamp %arg0 {min_val = 3 : i64, max_val = 109 : i64} : (tensor<42x99xi64>) -> tensor<42x99xi64>
    %1 = tosa.ceil %arg1 : (tensor<97x17x55x39x74x100xf32>) -> tensor<97x17x55x39x74x100xf32>
    %2 = tosa.sigmoid %1 : (tensor<97x17x55x39x74x100xf32>) -> tensor<97x17x55x39x74x100xf32>
    %3 = tosa.sub %2, %1 : (tensor<97x17x55x39x74x100xf32>, tensor<97x17x55x39x74x100xf32>) -> tensor<97x17x55x39x74x100xf32>
    return %0, %3 : tensor<42x99xi64>, tensor<97x17x55x39x74x100xf32>
  }
}
