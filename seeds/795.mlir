module {
  func.func @main(%arg0: tensor<71x91x67xf32>, %arg1: tensor<1x91x1xf32>) -> (tensor<71x91x67xf32>, tensor<71x67xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<71x91x67xf32>, tensor<1x91x1xf32>) -> tensor<71x91x67xf32>
    %1 = tosa.reverse %0 {axis = 2 : i32} : (tensor<71x91x67xf32>) -> tensor<71x91x67xf32>
    %2 = tosa.argmax %1 {axis = 1 : i32} : (tensor<71x91x67xf32>) -> tensor<71x67xi32>
    %3 = tosa.pow %0, %1 : (tensor<71x91x67xf32>, tensor<71x91x67xf32>) -> tensor<71x91x67xf32>
    %4 = tosa.bitwise_xor %2, %2 : (tensor<71x67xi32>, tensor<71x67xi32>) -> tensor<71x67xi32>
    return %3, %4 : tensor<71x91x67xf32>, tensor<71x67xi32>
  }
}
