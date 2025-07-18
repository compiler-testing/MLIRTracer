module {
  func.func @main(%arg0: tensor<68x24x25xf32>, %arg1: tensor<25xi32>, %arg2: tensor<25xi32>) -> (tensor<68x24x25xf32>, tensor<25xi32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<68x24x25xf32>) -> tensor<68x24x25xf32>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<25xi32>, tensor<25xi32>) -> tensor<25xi32>
    %2 = tosa.reciprocal %0 : (tensor<68x24x25xf32>) -> tensor<68x24x25xf32>
    %3 = tosa.intdiv %1, %1 : (tensor<25xi32>, tensor<25xi32>) -> tensor<25xi32>
    return %2, %3 : tensor<68x24x25xf32>, tensor<25xi32>
  }
}
