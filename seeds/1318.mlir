module {
  func.func @main(%arg0: tensor<62x100x49x1xi32>) -> tensor<62x100x49x1xi32> {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<62x100x49x1xi32>) -> tensor<62x100x49x1xi32>
    %1 = tosa.abs %0 : (tensor<62x100x49x1xi32>) -> tensor<62x100x49x1xi32>
    return %1 : tensor<62x100x49x1xi32>
  }
}
