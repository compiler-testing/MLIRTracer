module {
  func.func @main(%arg0: tensor<3x31x22xi1>) -> tensor<341x2xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<3x31x22xi1>) -> tensor<1x31x22xi1>
    %1 = tosa.logical_not %0 : (tensor<1x31x22xi1>) -> tensor<1x31x22xi1>
    %r_2 = tosa.const_shape {values = dense<[ 341, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<1x31x22xi1>, !tosa.shape<2>) -> tensor<341x2xi1>
    return %2 : tensor<341x2xi1>
  }
}
