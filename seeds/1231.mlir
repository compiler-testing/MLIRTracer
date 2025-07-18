module {
  func.func @main(%arg0: tensor<56x72xi64>, %arg1: tensor<16x49x28x75xi1>, %arg2: tensor<83xi32>, %arg3: tensor<1xi32>) -> (tensor<56x72xi64>, tensor<16x1x28x75xi1>, tensor<83xi32>) {
    %0 = tosa.clamp %arg0 {min_val = -6 : i64, max_val = 9 : i64} : (tensor<56x72xi64>) -> tensor<56x72xi64>
    %1 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<16x49x28x75xi1>) -> tensor<16x1x28x75xi1>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<83xi32>, tensor<1xi32>) -> tensor<83xi32>
    return %0, %1, %2 : tensor<56x72xi64>, tensor<16x1x28x75xi1>, tensor<83xi32>
  }
}
