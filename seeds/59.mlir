module {
  func.func @main(%arg0: tensor<6x9x85xi1>) -> tensor<6x9x85xi1> {
    %0 = tosa.bitwise_not %arg0 : (tensor<6x9x85xi1>) -> tensor<6x9x85xi1>
    %1 = tosa.bitwise_not %0 : (tensor<6x9x85xi1>) -> tensor<6x9x85xi1>
    %2 = tosa.reverse %1 {axis = 1 : i32} : (tensor<6x9x85xi1>) -> tensor<6x9x85xi1>
    return %2 : tensor<6x9x85xi1>
  }
}
