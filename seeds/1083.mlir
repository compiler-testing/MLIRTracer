module {
  func.func @main(%arg0: tensor<37x61xi1>) -> tensor<37x1xi1> {
    %0 = tosa.logical_not %arg0 : (tensor<37x61xi1>) -> tensor<37x61xi1>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<37x61xi1>) -> tensor<37x1xi1>
    return %1 : tensor<37x1xi1>
  }
}
