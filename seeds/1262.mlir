module {
  func.func @main(%arg0: tensor<73x75x19x26xi64>, %arg1: tensor<1x75x1x26xi64>) -> tensor<14235x1xi64> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<73x75x19x26xi64>, tensor<1x75x1x26xi64>) -> tensor<73x75x19x26xi64>
    %1 = tosa.maximum %0, %0 : (tensor<73x75x19x26xi64>, tensor<73x75x19x26xi64>) -> tensor<73x75x19x26xi64>
    %r_2 = tosa.const_shape {values = dense<[ 14235, 190 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<73x75x19x26xi64>, !tosa.shape<2>) -> tensor<14235x190xi64>
    %3 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<14235x190xi64>) -> tensor<14235x1xi64>
    return %3 : tensor<14235x1xi64>
  }
}
