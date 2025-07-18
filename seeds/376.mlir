module {
  func.func @main(%arg0: tensor<26xi8>, %arg1: tensor<11x23x57x51x40x5xi64>, %arg2: tensor<1x1x57x1x1x5xi64>, %arg3: tensor<i1>, %arg4: tensor<i1>) -> (tensor<i1>, tensor<i32>, tensor<11x23x57x51x40x5xi64>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<26xi8>) -> tensor<i32>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<11x23x57x51x40x5xi64>, tensor<1x1x57x1x1x5xi64>) -> tensor<11x23x57x51x40x5xi64>
    %2 = tosa.logical_or %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.intdiv %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.clamp %1 {min_val = -55 : i64, max_val = 31 : i64} : (tensor<11x23x57x51x40x5xi64>) -> tensor<11x23x57x51x40x5xi64>
    return %2, %3, %4 : tensor<i1>, tensor<i32>, tensor<11x23x57x51x40x5xi64>
  }
}
