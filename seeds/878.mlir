module {
  func.func @main(%arg0: tensor<15x94xi1>) -> tensor<1xi1> {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<15x94xi1>) -> tensor<15xi32>
    %1 = tosa.greater %0, %0 : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<15xi1>) -> tensor<1xi1>
    return %2 : tensor<1xi1>
  }
}
