module {
  func.func @main(%arg0: tensor<10x55x87x7xi8>) -> tensor<10x55x87x1xi8> {
    %0 = tosa.reduce_min %arg0 {axis = 3 : i32} : (tensor<10x55x87x7xi8>) -> tensor<10x55x87x1xi8>
    return %0 : tensor<10x55x87x1xi8>
  }
}
