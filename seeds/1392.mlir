module {
  func.func @main(%arg0: tensor<39x34x48x69xi64>, %arg1: tensor<84x17x23xi1>) -> (tensor<39x34x48x69xi64>, tensor<84x17x23xi1>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<39x34x48x69xi64>) -> tensor<39x34x48x69xi64>
    %1 = tosa.logical_not %arg1 : (tensor<84x17x23xi1>) -> tensor<84x17x23xi1>
    return %0, %1 : tensor<39x34x48x69xi64>, tensor<84x17x23xi1>
  }
}
