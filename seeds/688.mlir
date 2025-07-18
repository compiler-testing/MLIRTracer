module {
  func.func @main(%arg0: tensor<100x97x78x9x52xi32>) -> tensor<15132x26xi32> {
    %r_0 = tosa.const_shape {values = dense<[ 15132, 26, 900 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<100x97x78x9x52xi32>, !tosa.shape<3>) -> tensor<15132x26x900xi32>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<15132x26x900xi32>) -> tensor<15132x26xi32>
    return %1 : tensor<15132x26xi32>
  }
}
