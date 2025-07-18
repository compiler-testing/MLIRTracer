module {
  func.func @main(%arg0: tensor<28x32xi1>, %arg1: tensor<47x61x72x56x14xf32>) -> (tensor<28x1xi1>, tensor<28x1xi1>, tensor<28x1xi1>, tensor<47x61x72x56x14xf32>, tensor<47x61x72x56x14xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<28x32xi1>) -> tensor<28x1xi1>
    %1 = tosa.add %0, %0 : (tensor<28x1xi1>, tensor<28x1xi1>) -> tensor<28x1xi1>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<28x1xi1>) -> tensor<28x1xi1>
    %3 = tosa.floor %arg1 : (tensor<47x61x72x56x14xf32>) -> tensor<47x61x72x56x14xf32>
    %4 = tosa.abs %3 : (tensor<47x61x72x56x14xf32>) -> tensor<47x61x72x56x14xf32>
    %5 = tosa.tanh %4 : (tensor<47x61x72x56x14xf32>) -> tensor<47x61x72x56x14xf32>
    %6 = tosa.logical_right_shift %1, %1 : (tensor<28x1xi1>, tensor<28x1xi1>) -> tensor<28x1xi1>
    %7 = tosa.clz %1 : (tensor<28x1xi1>) -> tensor<28x1xi1>
    %8 = tosa.reciprocal %4 : (tensor<47x61x72x56x14xf32>) -> tensor<47x61x72x56x14xf32>
    %9 = tosa.ceil %5 : (tensor<47x61x72x56x14xf32>) -> tensor<47x61x72x56x14xf32>
    return %2, %6, %7, %8, %9 : tensor<28x1xi1>, tensor<28x1xi1>, tensor<28x1xi1>, tensor<47x61x72x56x14xf32>, tensor<47x61x72x56x14xf32>
  }
}
