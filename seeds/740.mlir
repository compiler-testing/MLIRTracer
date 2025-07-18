module {
  func.func @main(%arg0: tensor<5xi32>, %arg1: tensor<1xi32>, %arg2: tensor<87xf32>) -> (tensor<1x1x1xi32>, tensor<87xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<5xi32>, tensor<1xi32>) -> tensor<5xi32>
    %1 = tosa.maximum %0, %0 : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    %2 = tosa.argmax %1 {axis = 0 : i32} : (tensor<5xi32>) -> tensor<i32>
    %r_3 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.reshape %2, %r_3 : (tensor<i32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
    %4 = tosa.exp %arg2 : (tensor<87xf32>) -> tensor<87xf32>
    %5 = tosa.greater %4, %4 : (tensor<87xf32>, tensor<87xf32>) -> tensor<87xi1>
    return %3, %5 : tensor<1x1x1xi32>, tensor<87xi1>
  }
}
