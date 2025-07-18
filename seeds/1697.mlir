module {
  func.func @main(%arg0: tensor<84x47x55xi1>) -> tensor<84x47x1xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<84x47x55xi1>) -> tensor<84x47x1xi1>
    %1 = tosa.add %0, %0 : (tensor<84x47x1xi1>, tensor<84x47x1xi1>) -> tensor<84x47x1xi1>
    return %1 : tensor<84x47x1xi1>
  }
}
