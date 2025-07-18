module {
  func.func @main(%arg0: tensor<73x86xi64>, %arg1: tensor<67x3x12x49x93x18xf32>, %arg2: tensor<67x3x12x1x1x1xf32>) -> (tensor<67x3x12x49x93x18xi1>, tensor<73x86xi64>, tensor<67x3x12x49x93x18xf32>) {
    %0 = tosa.clamp %arg0 {min_val = 35 : i64, max_val = 116 : i64} : (tensor<73x86xi64>) -> tensor<73x86xi64>
    %1 = tosa.pow %arg1, %arg2 : (tensor<67x3x12x49x93x18xf32>, tensor<67x3x12x1x1x1xf32>) -> tensor<67x3x12x49x93x18xf32>
    %2 = tosa.greater %1, %1 : (tensor<67x3x12x49x93x18xf32>, tensor<67x3x12x49x93x18xf32>) -> tensor<67x3x12x49x93x18xi1>
    %3 = tosa.maximum %0, %0 : (tensor<73x86xi64>, tensor<73x86xi64>) -> tensor<73x86xi64>
    %4 = tosa.tanh %1 : (tensor<67x3x12x49x93x18xf32>) -> tensor<67x3x12x49x93x18xf32>
    return %2, %3, %4 : tensor<67x3x12x49x93x18xi1>, tensor<73x86xi64>, tensor<67x3x12x49x93x18xf32>
  }
}
