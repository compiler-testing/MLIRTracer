module {
  func.func @main(%arg0: tensor<71x92x57x98xi8>, %arg1: tensor<71x1x57x1xi8>, %arg2: tensor<68xf32>) -> (tensor<68xf32>, tensor<71x92x98xi32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<71x92x57x98xi8>, tensor<71x1x57x1xi8>) -> tensor<71x92x57x98xi1>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<71x92x57x98xi1>) -> tensor<71x92x98xi32>
    %2 = tosa.reverse %1 {axis = 2 : i32} : (tensor<71x92x98xi32>) -> tensor<71x92x98xi32>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<71x92x98xi32>, tensor<71x92x98xi32>) -> tensor<71x92x98xi32>
    %4 = tosa.reciprocal %arg2 : (tensor<68xf32>) -> tensor<68xf32>
    %5 = tosa.bitwise_and %3, %3 : (tensor<71x92x98xi32>, tensor<71x92x98xi32>) -> tensor<71x92x98xi32>
    %6 = tosa.clamp %5 {min_val = 6 : i32, max_val = 132 : i32} : (tensor<71x92x98xi32>) -> tensor<71x92x98xi32>
    return %4, %6 : tensor<68xf32>, tensor<71x92x98xi32>
  }
}
