module {
  func.func @main(%arg0: tensor<39xi64>) -> tensor<39xi64> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<39xi64>) -> tensor<39xi64>
    return %0 : tensor<39xi64>
  }
}
