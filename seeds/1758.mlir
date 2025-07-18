module {
  func.func @main(%arg0: tensor<35x85x50x14xi1>) -> tensor<35x85x50x1xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 3 : i32} : (tensor<35x85x50x14xi1>) -> tensor<35x85x50x1xi1>
    return %0 : tensor<35x85x50x1xi1>
  }
}
